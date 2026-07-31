"""Convert captured JPEG generations into compact review MP4 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import cv2

from .recording import CAMERA_ROLES


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def package_recording(root: Path, *, delete_frames: bool = False) -> dict[str, Any]:
    root = root.expanduser().resolve()
    grouped = {role: [] for role in CAMERA_ROLES}
    for row in _rows(root / "camera_frames.jsonl"):
        role = row.get("role")
        if role in grouped:
            grouped[role].append(row)
    videos = {}
    errors = []
    output_root = root / "videos"
    output_root.mkdir(parents=True, exist_ok=True)
    for role, rows in grouped.items():
        rows.sort(key=lambda value: int(value["frame_index"]))
        if len(rows) < 2:
            errors.append(f"{role}: fewer than two frames")
            continue
        timestamps = [int(value["capture_monotonic_ns"]) for value in rows]
        span_s = (timestamps[-1] - timestamps[0]) / 1e9
        fps = 30.0 if span_s <= 0 else (len(rows) - 1) / span_s
        fps = min(60.0, max(1.0, fps))
        first = cv2.imread(str(root / rows[0]["relative_path"]))
        if first is None:
            errors.append(f"{role}: first JPEG did not decode")
            continue
        height, width = first.shape[:2]
        path = output_root / f"{role}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            errors.append(f"{role}: OpenCV MP4 writer did not open")
            continue
        written = 0
        try:
            for row in rows:
                frame = cv2.imread(str(root / row["relative_path"]))
                if frame is None or frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        f"frame {row['frame_index']} failed decode/shape validation"
                    )
                writer.write(frame)
                written += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{role}: {type(exc).__name__}: {exc}")
        finally:
            writer.release()
        if written == len(rows) and path.stat().st_size > 0:
            videos[role] = {
                "relative_path": path.relative_to(root).as_posix(),
                "frame_count": written,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_s": span_s,
            }
    complete = len(videos) == len(CAMERA_ROLES) and not errors
    if complete and delete_frames:
        shutil.rmtree(root / "frames", ignore_errors=True)
    report = {"complete": complete, "videos": videos, "errors": errors}
    (root / "video_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--delete-frames", action="store_true")
    args = parser.parse_args()
    report = package_recording(args.capture_dir, delete_frames=args.delete_frames)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
