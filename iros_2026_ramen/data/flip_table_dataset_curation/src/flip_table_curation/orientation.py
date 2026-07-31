from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .source import SourceSnapshot


@dataclass(frozen=True)
class OrientationSample:
    episode_index: int
    valid: bool
    descriptor: list[float] | None
    detection_fraction: float
    thumbnail: str | None


def _frame_descriptor(result: Any, width: int, height: int) -> np.ndarray | None:
    if result.obb is None or len(result.obb) == 0:
        return None
    names = result.names
    tables: list[tuple[float, np.ndarray]] = []
    legs: list[np.ndarray] = []
    for cls, confidence, xywhr in zip(
        result.obb.cls.tolist(),
        result.obb.conf.tolist(),
        result.obb.xywhr.cpu().numpy(),
        strict=True,
    ):
        name = str(names[int(cls)])
        if name == "table_top":
            tables.append((float(confidence), np.asarray(xywhr, dtype=np.float64)))
        elif name in {"leg", "leg_tip"}:
            legs.append(np.asarray(xywhr, dtype=np.float64))
    if not tables:
        return None
    confidence, table = max(tables, key=lambda item: item[0])
    cx, cy, box_width, box_height, angle = table
    leg_centers = (
        np.asarray([[item[0] / width, item[1] / height] for item in legs])
        if legs
        else np.empty((0, 2))
    )
    if len(leg_centers):
        leg_mean = np.mean(leg_centers, axis=0)
        leg_spread = np.ptp(leg_centers, axis=0)
    else:
        leg_mean = np.zeros(2)
        leg_spread = np.zeros(2)
    return np.asarray(
        [
            confidence,
            cx / width,
            cy / height,
            box_width / width,
            box_height / height,
            np.sin(2.0 * angle),
            np.cos(2.0 * angle),
            min(len(legs), 8) / 8.0,
            leg_mean[0],
            leg_mean[1],
            leg_spread[0],
            leg_spread[1],
        ],
        dtype=np.float64,
    )


def _read_samples(
    captures: dict[tuple[str, Path], cv2.VideoCapture],
    snapshot: SourceSnapshot,
    row: dict[str, Any],
    key: str,
    frame_offsets: np.ndarray,
) -> list[np.ndarray]:
    path = snapshot.video_path(row, key)
    cache_key = (key, path)
    capture = captures.get(cache_key)
    if capture is None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open source video {path}")
        captures[cache_key] = capture
    start = snapshot.video_offset(row, key)
    frames: list[np.ndarray] = []
    for offset in frame_offsets:
        capture.set(cv2.CAP_PROP_POS_MSEC, (start + float(offset) / snapshot.fps) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"cannot decode episode={row['episode_index']} key={key} offset={offset}"
            )
        frames.append(frame)
    return frames


def extract_orientation_samples(
    snapshot: SourceSnapshot,
    *,
    rows: list[dict[str, Any]],
    weight_path: Path,
    output_dir: Path,
    sample_frames: int,
    sample_window_frames: int,
    confidence: float,
    minimum_detection_fraction: float,
) -> list[OrientationSample]:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weight_path))
    device: str | int = 0
    try:
        import torch

        if not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"
    offsets = np.rint(
        np.linspace(0, max(0, sample_window_frames - 1), sample_frames)
    ).astype(int)
    captures: dict[tuple[str, Path], cv2.VideoCapture] = {}
    samples: list[OrientationSample] = []
    try:
        for position, row in enumerate(rows):
            episode = int(row["episode_index"])
            images: list[np.ndarray] = []
            first_left: np.ndarray | None = None
            for key in ("observation.images.cam_0", "observation.images.cam_1"):
                frames = _read_samples(captures, snapshot, row, key, offsets)
                if key.endswith("cam_0"):
                    first_left = frames[0]
                images.extend(frames)
            predictions = model.predict(
                source=images, conf=confidence, verbose=False, device=device
            )
            descriptors = [
                descriptor
                for image, prediction in zip(images, predictions, strict=True)
                if (
                    descriptor := _frame_descriptor(
                        prediction, image.shape[1], image.shape[0]
                    )
                )
                is not None
            ]
            detection_fraction = len(descriptors) / len(images)
            thumbnail_path: Path | None = None
            if first_left is not None:
                thumbnail_path = output_dir / f"episode_{episode:06d}.jpg"
                cv2.imwrite(
                    str(thumbnail_path),
                    first_left,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
            valid = detection_fraction >= minimum_detection_fraction
            if valid:
                matrix = np.stack(descriptors)
                descriptor_value = np.concatenate(
                    [np.mean(matrix, axis=0), np.std(matrix, axis=0)]
                ).tolist()
            else:
                descriptor_value = None
            samples.append(
                OrientationSample(
                    episode_index=episode,
                    valid=valid,
                    descriptor=descriptor_value,
                    detection_fraction=detection_fraction,
                    thumbnail=(
                        str(thumbnail_path.relative_to(output_dir.parent))
                        if thumbnail_path
                        else None
                    ),
                )
            )
            if (position + 1) % 25 == 0:
                print(f"[orientation] {position + 1}/{len(rows)} episodes")
    finally:
        for capture in captures.values():
            capture.release()
    return samples

