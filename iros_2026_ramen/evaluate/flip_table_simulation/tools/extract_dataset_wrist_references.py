#!/usr/bin/env python3
"""Extract real flip-table wrist-camera reference frames from a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOT = os.environ.get("FLIP_TABLE_DATASET_ROOT")
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "flip_table_simulation"
    / "real_flip_table_wrist_frame0010_many_episodes"
)
WRIST_VIDEO_KEYS = {
    "left_wrist": "observation.images.cam_2",
    "right_wrist": "observation.images.cam_3",
}


def _episode_indices(total: int, count: int) -> list[int]:
    if count >= total:
        return list(range(total))
    indices = {round(i * (total - 1) / max(count - 1, 1)) for i in range(count)}
    return sorted(indices)


def _read_video_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return bgr


def _video_path(dataset_root: Path, video_key: str, chunk_index: int, file_index: int) -> Path:
    return (
        dataset_root
        / "videos"
        / video_key
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )


def extract_references(
    dataset_root: Path,
    output_dir: Path,
    frame_index: int,
    episode_count: int,
    fps: float | None,
) -> dict[str, object]:
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if fps is None:
        fps = float(info["fps"])
    episodes = pd.read_parquet(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    selected = _episode_indices(len(episodes), episode_count)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for episode_index in selected:
        row = episodes.iloc[episode_index]
        source_episode_name = str(row.get("source_episode_name", f"episode_{episode_index:06d}"))
        record: dict[str, object] = {
            "episode_index": int(row["episode_index"]),
            "source_episode_name": source_episode_name,
            "frame_index": frame_index,
            "roles": {},
        }
        for role, video_key in WRIST_VIDEO_KEYS.items():
            chunk_index = int(row[f"videos/{video_key}/chunk_index"])
            file_index = int(row[f"videos/{video_key}/file_index"])
            from_timestamp = float(row[f"videos/{video_key}/from_timestamp"])
            absolute_frame = int(round((from_timestamp + frame_index / fps) * fps))
            video_path = _video_path(dataset_root, video_key, chunk_index, file_index)
            bgr = _read_video_frame(video_path, absolute_frame)
            filename = f"{role}_real_ep{int(row['episode_index']):04d}_f{frame_index:04d}.png"
            output_path = output_dir / filename
            if not cv2.imwrite(str(output_path), bgr):
                raise RuntimeError(f"Could not write {output_path}")
            record["roles"][role] = {
                "video_key": video_key,
                "video_path": str(video_path),
                "from_timestamp": from_timestamp,
                "absolute_frame": absolute_frame,
                "path": str(output_path),
            }
        manifest.append(record)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "fps": fps,
        "frame_index": frame_index,
        "episode_count": len(selected),
        "selected_episode_indices": selected,
        "wrist_video_keys": WRIST_VIDEO_KEYS,
        "manifest": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(DEFAULT_DATASET_ROOT) if DEFAULT_DATASET_ROOT else None,
        required=DEFAULT_DATASET_ROOT is None,
        help="LeRobot dataset root; defaults to FLIP_TABLE_DATASET_ROOT.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--episode-count", type=int, default=48)
    parser.add_argument("--fps", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_references(
        args.dataset_root,
        args.output_dir,
        args.frame_index,
        args.episode_count,
        args.fps,
    )
    if not math.isfinite(float(summary["fps"])):
        raise RuntimeError(f"Invalid fps: {summary['fps']}")
    print(
        f"wrote {summary['episode_count']} paired wrist references to {summary['output_dir']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
