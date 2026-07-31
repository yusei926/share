#!/usr/bin/env python3
"""Extract table-top image-plane rotation for progress-label refinement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
DEFAULT_REVISION = "0dc47877dfb2efbea796a059c81290c649bc773c"
DEFAULT_WEIGHT_REPO = "Team-RAMEN/IROS2026_RAMEN_Hara_yoloobb_upperpolicy"
DEFAULT_WEIGHT_FILE = "runs/m_lowaug_v3/weights/best.pt"
EXPECTED_WEIGHT_SHA256 = "37aa511a01883edc03d92b7564fc09f95a7be804b678b3212ac2a2255f5d4a21"
VIDEO_KEY = "observation.images.cam_0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--weight", type=Path)
    parser.add_argument("--weight-repo", default=DEFAULT_WEIGHT_REPO)
    parser.add_argument("--weight-file", default=DEFAULT_WEIGHT_FILE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if not 0.0 < args.confidence <= 1.0:
        raise ValueError("--confidence must be in (0,1]")

    source_root = args.source_root or Path(
        snapshot_download(
            args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=[
                "meta/info.json",
                "meta/episodes/**/*.parquet",
                f"videos/{VIDEO_KEY}/**/*.mp4",
            ],
        )
    )
    weight = args.weight or Path(
        hf_hub_download(
            args.weight_repo,
            filename=args.weight_file,
            repo_type="model",
        )
    )
    if _sha256(weight) != EXPECTED_WEIGHT_SHA256:
        raise ValueError(
            f"YOLO weight hash mismatch for {weight}; expected {EXPECTED_WEIGHT_SHA256}"
        )
    for secret_name in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "WANDB_API_KEY",
        "ULTRALYTICS_API_KEY",
    ):
        os.environ.pop(secret_name, None)

    output_root = args.output_root.resolve()
    output_path = output_root / "visual_rotation.jsonl"
    manifest_path = output_root / "visual_rotation_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(f"{output_root} already contains visual-rotation output")
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("ULTRALYTICS_OFFLINE", "true")
    from ultralytics import YOLO, settings

    settings.update({"sync": False})

    model = YOLO(str(weight))
    rows = _episode_rows(source_root)
    records: list[dict[str, Any]] = []
    thumbnails: list[tuple[int, int, float, np.ndarray]] = []
    captures: dict[Path, cv2.VideoCapture] = {}
    try:
        for position, row in enumerate(rows, start=1):
            record, thumbnail = _extract_episode(
                source_root,
                row,
                captures=captures,
                model=model,
                stride=args.stride,
                confidence=args.confidence,
            )
            records.append(record)
            thumbnails.append(
                (
                    int(row["episode_index"]),
                    int(row["curation_orientation_cluster"]),
                    float(record["detection_fraction"]),
                    thumbnail,
                )
            )
            if position % 10 == 0 or position == len(rows):
                print(f"[visual-rotation] {position}/{len(rows)}", flush=True)
    finally:
        for capture in captures.values():
            capture.release()

    output_path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    contact_sheet_path = output_root / "orientation_contact_sheet.jpg"
    _write_contact_sheet(thumbnails, contact_sheet_path)
    manifest = {
        "schema_version": "flip_table_visual_rotation_manifest_v1",
        "dataset_repo_id": args.repo_id,
        "dataset_revision": args.revision,
        "source_root": source_root.resolve().as_posix(),
        "video_key": VIDEO_KEY,
        "weight_repo": args.weight_repo,
        "weight_file": args.weight_file,
        "weight_sha256": _sha256(weight),
        "stride": args.stride,
        "confidence_threshold": args.confidence,
        "episode_count": len(records),
        "mean_detection_fraction": float(
            np.mean([record["detection_fraction"] for record in records])
        ),
        "low_confidence_episodes": [
            record["episode_index"]
            for record in records
            if record["detection_fraction"] < 0.5
        ],
        "sidecar_sha256": _sha256(output_path),
        "contact_sheet": contact_sheet_path.name,
        "contact_sheet_sha256": _sha256(contact_sheet_path),
        "contact_sheet_human_review_required": True,
        "policy_input": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _episode_rows(source_root: Path) -> list[dict[str, Any]]:
    paths = sorted((source_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"episode metadata is missing under {source_root}")
    columns = [
        "episode_index",
        "length",
        "curation_orientation_cluster",
        f"videos/{VIDEO_KEY}/chunk_index",
        f"videos/{VIDEO_KEY}/file_index",
        f"videos/{VIDEO_KEY}/from_timestamp",
    ]
    rows = [row for path in paths for row in pq.read_table(path, columns=columns).to_pylist()]
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(174)):
        raise ValueError("visual extraction requires exactly episodes 0..173")
    return rows


def _extract_episode(
    source_root: Path,
    row: dict[str, Any],
    *,
    captures: dict[Path, cv2.VideoCapture],
    model: Any,
    stride: int,
    confidence: float,
) -> tuple[dict[str, Any], np.ndarray]:
    chunk = int(row[f"videos/{VIDEO_KEY}/chunk_index"])
    file_index = int(row[f"videos/{VIDEO_KEY}/file_index"])
    video_path = (
        source_root
        / "videos"
        / VIDEO_KEY
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    capture = captures.get(video_path)
    if capture is None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {video_path}")
        captures[video_path] = capture

    length = int(row["length"])
    start_seconds = float(row[f"videos/{VIDEO_KEY}/from_timestamp"])
    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)
    sample_indices = set(range(0, length, stride))
    sample_indices.add(length - 1)
    sampled_frames: list[np.ndarray] = []
    sampled_frame_indices: list[int] = []
    thumbnail: np.ndarray | None = None
    for frame_index in range(length):
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"cannot decode episode={row['episode_index']} frame={frame_index}"
            )
        if thumbnail is None:
            thumbnail = frame.copy()
        if frame_index in sample_indices:
            sampled_frame_indices.append(frame_index)
            sampled_frames.append(frame)
    assert thumbnail is not None

    predictions = model.predict(
        source=sampled_frames,
        conf=confidence,
        verbose=False,
        device=0,
    )
    detected_indices: list[int] = []
    detected_angles: list[float] = []
    detected_confidences: list[float] = []
    for frame_index, prediction in zip(sampled_frame_indices, predictions, strict=True):
        detection = _best_table_top(prediction)
        if detection is None:
            continue
        angle, score = detection
        detected_indices.append(frame_index)
        detected_angles.append(angle)
        detected_confidences.append(score)

    rotation = np.zeros(length, dtype=np.float32)
    confidence_values = np.zeros(length, dtype=np.float32)
    if len(detected_indices) >= 2:
        indices = np.asarray(detected_indices, dtype=np.float64)
        unwrapped = np.unwrap(2.0 * np.asarray(detected_angles, dtype=np.float64)) / 2.0
        all_indices = np.arange(length, dtype=np.float64)
        rotation[:] = np.interp(all_indices, indices, unwrapped).astype(np.float32)
        confidence_values[:] = np.interp(
            all_indices,
            indices,
            np.asarray(detected_confidences, dtype=np.float64),
            left=0.0,
            right=0.0,
        ).astype(np.float32)

    detection_fraction = len(detected_indices) / len(sampled_frame_indices)
    return (
        {
            "schema_version": "flip_table_visual_rotation_v1",
            "episode_index": int(row["episode_index"]),
            "length": length,
            "rotation_rad": rotation.tolist(),
            "confidence": confidence_values.tolist(),
            "detection_fraction": detection_fraction,
            "sampled_frame_count": len(sampled_frame_indices),
            "detected_frame_count": len(detected_indices),
        },
        thumbnail,
    )


def _best_table_top(result: Any) -> tuple[float, float] | None:
    if result.obb is None or len(result.obb) == 0:
        return None
    candidates: list[tuple[float, float]] = []
    for class_index, score, xywhr in zip(
        result.obb.cls.tolist(),
        result.obb.conf.tolist(),
        result.obb.xywhr.cpu().numpy(),
        strict=True,
    ):
        if str(result.names[int(class_index)]) != "table_top":
            continue
        _, _, width, height, angle = (float(value) for value in xywhr)
        if height > width:
            angle += np.pi / 2.0
        candidates.append((float(score), angle))
    if not candidates:
        return None
    score, angle = max(candidates)
    return angle, score


def _write_contact_sheet(
    thumbnails: list[tuple[int, int, float, np.ndarray]],
    output: Path,
) -> None:
    tile_width, tile_height = 192, 144
    columns = 10
    rows = (len(thumbnails) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (episode, cluster, fraction, image) in enumerate(thumbnails):
        tile = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (tile_width, 24), (0, 0, 0), thickness=-1)
        cv2.putText(
            tile,
            f"ep {episode:03d} c{cluster} det {fraction:.2f}",
            (4, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y = (index // columns) * tile_height
        x = (index % columns) * tile_width
        sheet[y : y + tile_height, x : x + tile_width] = tile
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write {output}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
