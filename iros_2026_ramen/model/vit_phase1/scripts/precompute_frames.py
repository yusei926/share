"""labels.parquet の全 (episode, frame) を mp4 から decode して JPEG に落とす.

Issue #18 の I/O bottleneck 対策 (mp4 に対する random access が遅いので、事前に
必要 frame だけ jpg 化してディスクに書く方針)。

- 保存先: `<dst>/ep_{ep_idx:04d}/frame_{frame_idx:05d}.jpg` (Dataset._jpg_path と一致)
- 384x384 に resize 済で保存 (training 時の resize 不要、load 高速化)
- JPEG quality 95 (h264 の lossy 上に更に載る劣化はほぼ無視できる、docs 参照)
- 並列: multiprocessing.Pool で episode 単位に配分、per-worker で mp4 を sequential decode
- resume: 既にある jpg は skip (--force で上書き)

Usage:
    pixi run -e train python -m model.vit_phase1.scripts.precompute_frames \\
        --labels-parquet /path/to/labels.parquet \\
        --lerobot-root data/vit_phase1/hf_cache \\
        --dst data/vit_phase1/frames \\
        --workers 4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CAM_KEY = "observation.images.cam_0"


# ------------------------------------------------------------
# meta load
# ------------------------------------------------------------
def load_ep_video_meta(lerobot_root: Path) -> dict[int, dict]:
    """ep_idx → {'video_file_path', 'from_timestamp'} の辞書."""
    meta_files = sorted((lerobot_root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"no meta parquet under {lerobot_root}/meta/episodes/")
    t = pa.concat_tables([pq.read_table(f) for f in meta_files])
    ep = t["episode_index"].to_numpy()
    chunk = t[f"videos/{CAM_KEY}/chunk_index"].to_numpy()
    file_ = t[f"videos/{CAM_KEY}/file_index"].to_numpy()
    from_ts = t[f"videos/{CAM_KEY}/from_timestamp"].to_numpy()
    result: dict[int, dict] = {}
    for i in range(len(ep)):
        e = int(ep[i])
        vpath = (
            lerobot_root
            / "videos"
            / CAM_KEY
            / f"chunk-{int(chunk[i]):03d}"
            / f"file-{int(file_[i]):03d}.mp4"
        )
        result[e] = {
            "video_file_path": str(vpath),
            "from_timestamp": float(from_ts[i]),
        }
    return result


def load_needed_frames(labels_parquet: Path) -> dict[int, list[int]]:
    """ep_idx → sorted list of frame_index を返す (labels.parquet 全 split 対象)."""
    t = pq.read_table(labels_parquet)
    ep = t["episode_index"].to_numpy()
    fr = t["frame_index"].to_numpy()
    result: dict[int, list[int]] = defaultdict(list)
    for i in range(len(ep)):
        result[int(ep[i])].append(int(fr[i]))
    for k in result:
        result[k] = sorted(set(result[k]))
    return dict(result)


# ------------------------------------------------------------
# worker
# ------------------------------------------------------------
def _worker_process_episode(
    task: tuple[int, list[int], dict, str, int, int, bool],
) -> tuple[int, int, int]:
    """1 episode を sequential decode し jpg で保存。(ep_idx, saved, skipped) を返す."""
    ep_idx, frames, meta, dst_root, image_size, jpg_quality, force = task
    vpath = meta["video_file_path"]
    from_ts = meta["from_timestamp"]

    out_dir = Path(dst_root) / f"ep_{ep_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {vpath}")
    mp4_fps = cap.get(cv2.CAP_PROP_FPS)
    mp4_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts_offset = int(round(from_ts * mp4_fps))
    saved = 0
    skipped = 0

    try:
        for fr in frames:
            out_path = out_dir / f"frame_{fr:05d}.jpg"
            if out_path.exists() and not force:
                skipped += 1
                continue
            abs_frame = ts_offset + fr
            if not 0 <= abs_frame < mp4_total:
                # 想定外だが破損 mp4 のとき silent 落ちしない
                print(
                    f"[warn] ep {ep_idx} frame {fr} maps to {abs_frame}, "
                    f"but mp4 has {mp4_total} frames — skipping",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(abs_frame))
            ok, frame_bgr = cap.read()
            if not ok:
                print(f"[warn] ep {ep_idx} frame {fr} read failed", file=sys.stderr)
                skipped += 1
                continue
            resized = cv2.resize(frame_bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
            saved += 1
    finally:
        cap.release()

    return ep_idx, saved, skipped


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-parquet", type=Path, required=True)
    p.add_argument("--lerobot-root", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True, help="jpg 保存 root")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--force", action="store_true", help="既存 jpg も上書き")
    args = p.parse_args()

    print(f"[precompute] labels={args.labels_parquet}")
    print(f"[precompute] lerobot_root={args.lerobot_root}")
    print(f"[precompute] dst={args.dst}")
    print(f"[precompute] workers={args.workers} image_size={args.image_size} q={args.quality}")

    # ---- meta + labels 統合 ----
    ep_meta = load_ep_video_meta(args.lerobot_root)
    ep_to_frames = load_needed_frames(args.labels_parquet)

    missing_meta = sorted(set(ep_to_frames) - set(ep_meta))
    if missing_meta:
        print(
            f"[FAIL] {len(missing_meta)} episodes in labels but no meta. "
            f"first: {missing_meta[:5]}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_frames = sum(len(v) for v in ep_to_frames.values())
    print(f"[precompute] {len(ep_to_frames)} episodes, {total_frames:,} frames to process")

    # ---- タスク作成 ----
    tasks = [
        (
            ep_idx,
            frames,
            ep_meta[ep_idx],
            str(args.dst),
            args.image_size,
            args.quality,
            args.force,
        )
        for ep_idx, frames in sorted(ep_to_frames.items())
    ]

    # ---- 並列実行 ----
    start = time.time()
    with mp.Pool(processes=args.workers) as pool, tqdm(
        total=total_frames, unit="frame"
    ) as pbar:
        total_saved = 0
        total_skipped = 0
        for ep_idx, saved, skipped in pool.imap_unordered(_worker_process_episode, tasks):
            total_saved += saved
            total_skipped += skipped
            pbar.update(saved + skipped)
            pbar.set_postfix(saved=total_saved, skipped=total_skipped)
    elapsed = time.time() - start

    print(
        f"[precompute] done. saved={total_saved:,} skipped={total_skipped:,} "
        f"elapsed={elapsed:.1f}s ({total_saved/max(elapsed,1):.1f} frames/s)"
    )


if __name__ == "__main__":
    main()
