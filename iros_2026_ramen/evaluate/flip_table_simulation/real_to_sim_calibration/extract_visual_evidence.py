#!/usr/bin/env python3
"""Extract synchronized RGB evidence from selected real flip-table episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json

from .contracts import CAMERA_ROLES, SOURCE_FPS


def frame_indices(frame_count: int) -> tuple[int, ...]:
    if frame_count < 2:
        raise ValueError("episode must contain at least two frames")
    return tuple(sorted({0, 10, round((frame_count - 1) * 0.25), round((frame_count - 1) * 0.50), round((frame_count - 1) * 0.75), frame_count - 1}))


def _read_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok or image is None:
        raise RuntimeError(f"cannot decode source video frame {frame}: {path}")
    if image.shape != (480, 640, 3):
        raise ValueError(f"source camera must decode 640x480 RGB, got {image.shape}: {path}")
    return image


def _metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "mean_luma": float(gray.mean()),
        "luma_std": float(gray.std()),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_saturation": float(hsv[:, :, 1].mean()),
        "near_black_fraction": float((gray < 12).mean()),
        "near_white_fraction": float((gray > 240).mean()),
    }


def _video_path(dataset_root: Path, camera_key: str, metadata: dict[str, Any]) -> tuple[Path, int]:
    prefix = f"videos/{camera_key}/"
    chunk = int(metadata[prefix + "chunk_index"])
    file = int(metadata[prefix + "file_index"])
    start = float(metadata[prefix + "from_timestamp"])
    return dataset_root / "videos" / camera_key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4", int(round(start * SOURCE_FPS))


def extract(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest["source"]
    dataset_root = Path(source["dataset_root"])
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    evidence: list[dict[str, Any]] = []
    for role, raw_path in manifest["episode_bundles"].items():
        bundle = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        count = len(bundle["timestamps_s"])
        for local_frame in frame_indices(count):
            for camera_key, camera_role in CAMERA_ROLES.items():
                video, start_frame = _video_path(dataset_root, camera_key, bundle["video_metadata"])
                image = _read_frame(video, start_frame + local_frame)
                destination = output_dir / role / f"frame_{local_frame:04d}" / f"{camera_role}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(destination), image):
                    raise RuntimeError(f"cannot write {destination}")
                evidence.append(
                    {
                        "bundle_role": role,
                        "episode_index": int(bundle["source_episode_index"]),
                        "local_frame": local_frame,
                        "camera_key": camera_key,
                        "camera_role": camera_role,
                        "source_video": str(video),
                        "source_video_frame": start_frame + local_frame,
                        "output": str(destination),
                        "metrics": _metrics(image),
                    }
                )
    report = {
        "schema_version": "team_ramen_flip_table_real_visual_evidence/v1",
        "manifest": str(manifest_path),
        "source_revision": source["revision"],
        "frame_count": len(evidence),
        "camera_roles": CAMERA_ROLES,
        "evidence": evidence,
    }
    atomic_write_json(output_dir / "visual_evidence_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = extract(args.manifest.expanduser().resolve(), args.output_dir.expanduser().resolve())
    print(f"wrote {report['frame_count']} synchronized 640x480 RGB evidence frames")


if __name__ == "__main__":
    main()
