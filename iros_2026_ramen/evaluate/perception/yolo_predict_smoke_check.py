"""Desktop 側 (pixi runtime env) から Orin publish の CompressedImage topic を
subscribe → YoloObbPerception で inference → OBB を描画した画像 + summary を保存する
smoke check (Issue #64 verify)。

Issue #64 実機 verify の一環で「YOLO が実カメラで何を検出しているか」を目視確認する。
entrypoint.py と同じ Ros2FrameSource (stereo_view="left") + YoloObbPerception を
経由するので、production pipeline と同じ inference 結果が得られる。

Usage:
    pixi run -e runtime python -m evaluate.perception.yolo_predict_smoke_check \\
        --weight model/yolo_obb/runs/m_lowaug_v3/weights/best.pt \\
        --topic /head/camera/color/image_raw/compressed \\
        --n-frames 10 \\
        --out-dir outputs/perception_smoke

    # 実カメラで検出不足の場合、conf を下げて「低信頼度でも検出はできてる」を確認
    pixi run -e runtime python -m evaluate.perception.yolo_predict_smoke_check \\
        --weight ... --conf 0.10 --n-frames 10

Exit code:
    0: --n-frames 分の inference + 画像保存 完了
    1: --timeout 以内に最初の frame が届かなかった (subscribe_smoke_check で切り分け)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="YOLO-OBB weight (.pt) path。entrypoint.py と同じ weight を指定",
    )
    parser.add_argument(
        "--topic",
        default="/head/camera/color/image_raw/compressed",
        help="subscribe 先 topic 名 (Orin 側 usb_cam の namespace と一致させる)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=10,
        help="inference + 保存する frame 数 (default 10)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/perception_smoke"),
        help="annotated jpg + summary.json の保存先",
    )
    parser.add_argument(
        "--stereo-view",
        choices=["packed", "left", "right"],
        default="left",
        help=(
            "packed camera stream の分割方法。default 'left' は entrypoint.py と同じ"
            " (YOLO 学習入力 = LeRobot cam_0 = 左眼 640x480 に合わせる)"
        ),
    )
    parser.add_argument(
        "--first-frame-timeout",
        type=float,
        default=10.0,
        help="最初の camera frame を待つ timeout [s]",
    )
    parser.add_argument(
        "--frame-poll-interval",
        type=float,
        default=0.05,
        help="frame 取得の poll 間隔 [s] (default 0.05 = 20Hz)",
    )
    parser.add_argument(
        "--domain-id",
        type=int,
        default=0,
        help="ROS_DOMAIN_ID (Orin container の compose 側と一致させる)",
    )
    parser.add_argument(
        "--network",
        default="",
        help="cyclonedds network interface (空 = auto、WSL / 別 LAN 時に明示指定)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="YOLO device (cuda / cpu / None は auto)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "YOLO 信頼度 threshold (default 0.25、YoloObbPerception と同じ)。"
            " 実カメラで検出不足の debug 時に --conf 0.10 まで下げて"
            " 「学習 domain 内で低信頼度で検出はできてる」を確認する"
        ),
    )
    args = parser.parse_args()

    if not 0.0 <= args.conf <= 1.0:
        print(f"NG: --conf must be in [0, 1], got {args.conf}", file=sys.stderr)
        return 1

    if not args.weight.exists():
        print(f"NG: weight not found: {args.weight}", file=sys.stderr)
        return 1

    # SDK ChannelFactory を先に Init (Ros2FrameSource が singleton 前提のため、
    # subscribe_smoke_check.py と同 pattern)
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if args.network:
        ChannelFactoryInitialize(args.domain_id, args.network)
    else:
        ChannelFactoryInitialize(args.domain_id)

    # heavy import は SDK init 後 (import cost を明示化)
    import cv2

    from inference.desktop.perception.frame_source import Ros2FrameSource
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    print(f"[smoke] SDK ChannelFactory init (domain={args.domain_id})", file=sys.stderr)

    perception = YoloObbPerception(args.weight, conf=args.conf, device=args.device)
    print(
        f"[smoke] YOLO weight loaded: {args.weight} conf={args.conf}"
        f" (classes={sorted(perception.class_names.values())})",
        file=sys.stderr,
    )

    source = Ros2FrameSource(topic=args.topic, stereo_view=args.stereo_view)
    print(
        f"[smoke] subscribing {args.topic} stereo_view={args.stereo_view}",
        file=sys.stderr,
    )

    # 最初の frame を timeout 付きで待つ (entrypoint と同じ pattern)
    print(
        f"[smoke] waiting for first frame (timeout={args.first_frame_timeout}s)...",
        file=sys.stderr,
    )
    deadline = time.monotonic() + args.first_frame_timeout
    while source.get() is None:
        if time.monotonic() > deadline:
            print(
                f"NG: no camera frame within {args.first_frame_timeout}s."
                " Check --topic / --network / Orin bringup",
                file=sys.stderr,
            )
            source.close()
            return 1
        time.sleep(0.1)
    first = source.get()
    assert first is not None
    print(
        f"[smoke] first frame arrived: shape={first.rgb.shape} t={first.t}",
        file=sys.stderr,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # N frame 分 inference + 保存 (重複 t は skip、latest-only source なので同じ frame を
    # 何度も見る可能性あり)
    summary: list[dict] = []
    seen_t: set[int] = set()
    captured = 0
    while captured < args.n_frames:
        frame = source.get()
        if frame is None or frame.t in seen_t:
            time.sleep(args.frame_poll_interval)
            continue
        seen_t.add(frame.t)

        dets = perception.predict(frame.rgb)
        class_counts: dict[str, int] = {}
        for d in dets:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        # annotated jpg 保存: 元画像に OBB verts (緑 polyline) + class + conf を描画
        annotated = frame.rgb.copy()
        height, width = annotated.shape[:2]
        pixel_scale = np.array([width, height], dtype=np.float32)
        for d in dets:
            # OBBDetection.verts は [0, 1] の正規化座標。OpenCV には必ず
            # pixel座標で渡す。直接 int 化すると全boxが左上数pixelへ潰れる。
            pts = np.rint(d.verts * pixel_scale).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.polylines(
                annotated, [pts.reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 0), thickness=2,
            )
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(
                annotated, label,
                (int(pts[0, 0]), int(pts[0, 1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        out_path = args.out_dir / f"frame_{captured:03d}.jpg"
        cv2.imwrite(str(out_path), annotated)

        print(
            f"[frame {captured}] t={frame.t} n_dets={len(dets)} classes={class_counts}",
            file=sys.stderr,
        )
        summary.append({
            "t": frame.t,
            "n_dets": len(dets),
            "classes": class_counts,
            "dets": [
                {
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "verts": d.verts.tolist(),
                }
                for d in dets
            ],
            "image": out_path.name,
        })
        captured += 1

    # summary.json 保存
    summary_path = args.out_dir / "summary.json"
    total_dets = sum(s["n_dets"] for s in summary)
    all_classes: set[str] = set()
    for s in summary:
        all_classes.update(s["classes"].keys())
    summary_json = {
        "n_frames": captured,
        "total_dets": total_dets,
        "avg_dets_per_frame": total_dets / captured if captured else 0.0,
        "classes_seen": sorted(all_classes),
        "frames": summary,
    }
    summary_path.write_text(json.dumps(summary_json, indent=2))

    print(
        f"[smoke] done: saved {captured} frames to {args.out_dir},"
        f" avg {summary_json['avg_dets_per_frame']:.2f} dets/frame,"
        f" classes seen: {sorted(all_classes)}",
        file=sys.stderr,
    )
    source.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
