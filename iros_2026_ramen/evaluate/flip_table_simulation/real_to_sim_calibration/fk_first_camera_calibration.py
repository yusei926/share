#!/usr/bin/env python3
"""Calibrate recorded flip-table cameras in an FK-first order.

The real dataset contains calibrated image intrinsics and encoder states, but
does not contain camera-to-link transforms.  This program deliberately solves
the observable part first: a single head-left mount correction is constrained
by G1 arm geometry across several *static* poses.  It only unlocks table,
workbench, and wrist-camera fitting after a held-out head-image gate passes.

No simulator state, object pose, contact, segmentation, or global camera is
used by the head stage.  The output is offline calibration evidence and must
not be used as a policy feature or runtime branch.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import combinations, product
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.source_dataset import SourceDatasetIndex, extract_video_frame
from evaluate.flip_table_simulation.real_to_sim_calibration.head_arm_motion_alignment import (
    ArmProjector,
)
from evaluate.flip_table_simulation.real_to_sim_calibration.source_cad_alignment import (
    PoseEstimate,
    TABLE_YAW_180_SYMMETRY,
    align_source_episode,
    robust_fixed_pose,
)


SCHEMA_VERSION = "team_ramen_flip_table_fk_first_camera_calibration/v1"
IMAGE_SHAPE = (480, 640)
HEAD_CAMERA_KEY = "observation.images.cam_0"
ARM_SLICE = slice(22, 36)
LANDMARK_INDEX = {"elbow": 3, "wrist": 5, "hand_base": 6}
LANDMARKS = tuple(LANDMARK_INDEX)
MAX_STATIC_SPEED_RAD_S = 0.18
MAX_OFFSET_FRAMES = 4
MAX_OFFSET_M = 0.030
MAX_OFFSET_DEG = 4.0
MAX_AUTOMATIC_DISTANCE_PX = 18.0
MIN_AUTOMATIC_CONFIDENCE = 0.55
MIN_HOLDOUT_POINTS = 12
HEAD_MEDIAN_GATE_PX = 3.0
HEAD_P95_GATE_PX = 8.0
SCENE_HOLDOUT_TRANSLATION_GATE_M = 0.050
SCENE_HOLDOUT_ROTATION_GATE_DEG = 6.0
SCENE_FRAME_MAXIMUM_DEFAULT = 50
SCENE_FRAME_STRIDE_DEFAULT = 5


@dataclass(frozen=True)
class SourceFrame:
    episode_index: int
    frame_index: int
    robot_q_current: np.ndarray
    image_bgr: np.ndarray
    split: str


@dataclass(frozen=True)
class ImageFeature:
    episode_index: int
    frame_index: int
    split: str
    side: str
    landmark: str
    pixel_xy: np.ndarray
    confidence: float
    origin: str


def _require_image(image: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(image)
    if value.shape != (*IMAGE_SHAPE, 3) or value.dtype != np.uint8:
        raise ValueError(f"{label} must be an uint8 640x480 BGR image")
    return value


def _require_q(q: Any, label: str) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (36,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite 36D robot_q_current")
    return value


def static_frame_indices(
    robot_q_current: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    count: int = 4,
    maximum_speed_rad_s: float = MAX_STATIC_SPEED_RAD_S,
) -> tuple[int, ...]:
    """Choose distinct, static, arm-diverse frames without using scene pixels."""

    q = np.asarray(robot_q_current, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 36 or len(q) < count:
        raise ValueError("robot_q_current must be [T,36] with at least count rows")
    if timestamps.shape != (len(q),) or not np.isfinite(q).all() or not np.isfinite(timestamps).all():
        raise ValueError("source state/timestamps must be finite and aligned")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    arms = q[:, ARM_SLICE]
    speed = np.zeros(len(q), dtype=np.float64)
    if len(q) > 1:
        delta = np.linalg.norm(np.diff(arms, axis=0), axis=1) / np.diff(timestamps)
        speed[0] = delta[0]
        speed[-1] = delta[-1]
        if len(q) > 2:
            speed[1:-1] = np.maximum(delta[:-1], delta[1:])
    candidates = np.flatnonzero(speed <= maximum_speed_rad_s)
    if len(candidates) < count:
        # A data capture may never be perfectly still.  Keep the rule explicit
        # and select the least-moving states rather than silently using motion.
        candidates = np.argsort(speed)[: max(count, min(len(q), count * 8))]
    candidates = np.asarray(sorted(set(int(value) for value in candidates)), dtype=np.int64)
    selected = [int(candidates[np.argmin(speed[candidates])])]
    while len(selected) < count:
        remaining = np.asarray([value for value in candidates if int(value) not in selected], dtype=np.int64)
        if len(remaining) == 0:
            raise ValueError("could not select enough distinct static frames")
        diversity = np.min(
            np.linalg.norm(arms[remaining, None, :] - arms[np.asarray(selected)][None, :, :], axis=2),
            axis=1,
        )
        # Prefer configuration diversity, then lower velocity, then earlier frames.
        order = np.lexsort((remaining, speed[remaining], -diversity))
        selected.append(int(remaining[order[0]]))
    return tuple(sorted(selected))


def can_accept_head_calibration(
    *,
    median_px: float | None,
    p95_px: float | None,
    holdout_points: int,
    manual_points: int,
) -> tuple[bool, str]:
    """Return an auditable hold-out decision for one shared mount estimate."""

    if median_px is None or p95_px is None or not np.isfinite((median_px, p95_px)).all():
        return False, "hold-out reprojection metrics are unavailable"
    if holdout_points < MIN_HOLDOUT_POINTS:
        return False, f"fewer than {MIN_HOLDOUT_POINTS} hold-out arm features"
    if median_px > HEAD_MEDIAN_GATE_PX or p95_px > HEAD_P95_GATE_PX:
        return False, "held-out reprojection gate failed"
    return True, "passed shared-mount held-out reprojection gate"


def _edge_distance_and_nearest(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = _require_image(image_bgr, "image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.0), 40, 120)
    # `distanceTransformWithLabels` labels non-zero pixels in its input.  Set
    # Canny edges to zero to obtain the closest edge location for every pixel.
    distance, labels = cv2.distanceTransformWithLabels(
        (edges == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    edge_y, edge_x = np.nonzero(edges)
    if len(edge_x) == 0:
        raise ValueError("automatic arm feature extraction found no image edges")
    lookup = np.column_stack((edge_x, edge_y)).astype(np.float64)
    # OpenCV's pixel labels are one-based.  The defensive clip protects a
    # backend-specific zero label at an edge pixel without fabricating a point.
    labels = np.clip(labels.astype(np.int64) - 1, 0, len(lookup) - 1)
    return distance.astype(np.float64), lookup[labels]


def _clip_pixel(pixel: np.ndarray) -> tuple[int, int] | None:
    xy = np.asarray(pixel, dtype=np.float64)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        return None
    x, y = np.rint(xy).astype(int)
    if not 0 <= x < IMAGE_SHAPE[1] or not 0 <= y < IMAGE_SHAPE[0]:
        return None
    return int(x), int(y)


def automatic_features(
    source_frame: SourceFrame,
    projector: ArmProjector,
) -> tuple[ImageFeature, ...]:
    """Extract conservative arm-edge candidates around nominal FK landmarks."""

    distance, nearest = _edge_distance_and_nearest(source_frame.image_bgr)
    pixels = projector.project(
        source_frame.robot_q_current, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    )
    output: list[ImageFeature] = []
    for side in ("left", "right"):
        for landmark, index in LANDMARK_INDEX.items():
            candidate = _clip_pixel(pixels[side][index])
            if candidate is None:
                continue
            x, y = candidate
            residual = float(distance[y, x])
            if residual > MAX_AUTOMATIC_DISTANCE_PX:
                continue
            confidence = float(np.clip(1.0 - residual / MAX_AUTOMATIC_DISTANCE_PX, 0.0, 1.0))
            output.append(
                ImageFeature(
                    source_frame.episode_index,
                    source_frame.frame_index,
                    source_frame.split,
                    side,
                    landmark,
                    np.asarray(nearest[y, x], dtype=np.float64),
                    confidence,
                    "automatic_edge",
                )
            )
    return tuple(output)


def _manual_feature_key(feature: ImageFeature) -> tuple[int, int, str, str]:
    return feature.episode_index, feature.frame_index, feature.side, feature.landmark


def load_manual_features(path: Path | None) -> dict[tuple[int, int, str, str], ImageFeature]:
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("manual keypoints schema differs from FK-first calibration schema")
    values = document.get("manual_features")
    if not isinstance(values, list):
        raise ValueError("manual_features must be a list")
    output: dict[tuple[int, int, str, str], ImageFeature] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("manual feature must be an object")
        side = value.get("side")
        landmark = value.get("landmark")
        split = value.get("split")
        if side not in ("left", "right") or landmark not in LANDMARKS or split not in ("fit", "holdout"):
            raise ValueError("manual feature side, landmark, or split is invalid")
        pixel = np.asarray(value.get("pixel_xy"), dtype=np.float64)
        if _clip_pixel(pixel) is None:
            raise ValueError("manual feature pixel must be within the 640x480 image")
        feature = ImageFeature(
            int(value["episode_index"]), int(value["frame_index"]), str(split), str(side), str(landmark), pixel,
            1.0, "manual",
        )
        key = _manual_feature_key(feature)
        if key in output:
            raise ValueError(f"duplicate manual feature: {key}")
        output[key] = feature
    return output


def _read_episode_states(source_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq

    episode = SourceDatasetIndex(source_root).episode(episode_index)
    rows = pq.read_table(
        episode.data_path,
        columns=["frame_index", "timestamp", "observation.state.robot_q_current"],
        filters=[("episode_index", "=", episode_index)],
    ).to_pylist()
    rows.sort(key=lambda item: int(item["frame_index"]))
    if [int(row["frame_index"]) for row in rows] != list(range(episode.frame_count)):
        raise ValueError(f"episode {episode_index} frame index is not contiguous")
    q = np.asarray([row["observation.state.robot_q_current"] for row in rows], dtype=np.float64)
    timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    if q.shape != (episode.frame_count, 36):
        raise ValueError(f"episode {episode_index} robot state has unexpected shape {q.shape}")
    return q, timestamps


def selected_source_frames(
    source_root: Path,
    episodes: Iterable[int],
    *,
    count_per_episode: int = 4,
) -> dict[int, tuple[int, ...]]:
    return {
        int(episode): static_frame_indices(*_read_episode_states(source_root, int(episode)), count=count_per_episode)
        for episode in episodes
    }


def _source_frames(
    *, source_root: Path, selected: Mapping[int, Iterable[int]], output_dir: Path
) -> tuple[SourceFrame, ...]:
    source = SourceDatasetIndex(source_root)
    config = load_pipeline_config(DEFAULT_CONFIG_PATH)
    camera = next(item for item in config.cameras if item.source_key == HEAD_CAMERA_KEY)
    output: list[SourceFrame] = []
    for episode_index, frame_values in sorted(selected.items()):
        q, _ = _read_episode_states(source_root, episode_index)
        episode = source.episode(episode_index)
        video = episode.video_slice(HEAD_CAMERA_KEY)
        frames = tuple(frame_values)
        for frame_index in frames:
            if not 0 <= frame_index < len(q):
                raise ValueError(f"frame {frame_index} is outside episode {episode_index}")
            path = output_dir / "real_head_left" / f"episode_{episode_index:04d}" / f"frame_{frame_index:06d}.png"
            extract_video_frame(video.path, video.timestamp_for_frame(frame_index, camera.fps, episode.frame_count), path)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            _require_image(image, str(path))
            output.append(SourceFrame(
                episode_index, frame_index, q[frame_index], image,
                # The final fit/hold-out assignment happens after automatic
                # visibility scoring; do not confuse temporal order with a
                # visible, independent robot-pose observation.
                "candidate",
            ))
    return tuple(output)


def _select_visible_frames(
    candidates: tuple[SourceFrame, ...],
    features: tuple[ImageFeature, ...],
    *,
    count_per_episode: int,
) -> tuple[SourceFrame, ...]:
    """Keep arm-visible, joint-diverse static poses and split them 2/2."""

    feature_count = defaultdict(int)
    for feature in features:
        if feature.confidence >= MIN_AUTOMATIC_CONFIDENCE:
            feature_count[(feature.episode_index, feature.frame_index)] += 1
    by_episode: dict[int, list[SourceFrame]] = defaultdict(list)
    for frame in candidates:
        by_episode[frame.episode_index].append(frame)
    selected: list[SourceFrame] = []
    for episode, values in sorted(by_episode.items()):
        if len(values) < count_per_episode:
            raise ValueError(f"episode {episode} lacks enough static candidates")
        chosen: list[SourceFrame] = []
        while len(chosen) < count_per_episode:
            remaining = [item for item in values if item not in chosen]
            best = max(
                remaining,
                key=lambda item: (
                    feature_count[(item.episode_index, item.frame_index)] * 10.0
                    + (0.0 if not chosen else min(
                        float(np.linalg.norm(item.robot_q_current[ARM_SLICE] - prior.robot_q_current[ARM_SLICE]))
                        for prior in chosen
                    )),
                    -item.frame_index,
                ),
            )
            chosen.append(best)
        selected.extend(chosen)
    if any(sum(item.episode_index == episode for item in selected) != count_per_episode for episode in by_episode):
        raise RuntimeError("selected frame count per episode is inconsistent")

    # Two hold-outs in each calibration episode prevent one episode's scene
    # layout from carrying the gate.  Choose the allocation with the strongest
    # weaker side (fit vs. hold-out) of automatically visible arm landmarks.
    per_episode_pairs = []
    for episode in sorted(by_episode):
        values = [item for item in selected if item.episode_index == episode]
        per_episode_pairs.append(tuple(combinations(values, 2)))
    best_holdout: tuple[SourceFrame, ...] | None = None
    best_score: tuple[int, int] | None = None
    for groups in product(*per_episode_pairs):
        holdout = tuple(item for group in groups for item in group)
        fit = tuple(item for item in selected if item not in holdout)
        holdout_count = sum(feature_count[(item.episode_index, item.frame_index)] for item in holdout)
        fit_count = sum(feature_count[(item.episode_index, item.frame_index)] for item in fit)
        score = (min(fit_count, holdout_count), fit_count + holdout_count)
        if best_score is None or score > best_score:
            best_holdout, best_score = holdout, score
    if best_holdout is None:
        raise RuntimeError("could not assign visible frames to fit/hold-out splits")
    holdout_keys = {(item.episode_index, item.frame_index) for item in best_holdout}
    return tuple(
        replace(item, split="holdout" if (item.episode_index, item.frame_index) in holdout_keys else "fit")
        for item in sorted(selected, key=lambda item: (item.episode_index, item.frame_index))
    )


def _project_landmark(
    projector: ArmProjector,
    q: np.ndarray,
    side: str,
    landmark: str,
    correction: np.ndarray,
) -> np.ndarray | None:
    pixels = projector.project(q, correction[:3], correction[3:])
    return pixels[side][LANDMARK_INDEX[landmark]] if side in pixels else None


def _features_with_manual_overrides(
    automatic: Iterable[ImageFeature], manual: Mapping[tuple[int, int, str, str], ImageFeature]
) -> tuple[ImageFeature, ...]:
    values = {_manual_feature_key(value): value for value in automatic if value.confidence >= MIN_AUTOMATIC_CONFIDENCE}
    values.update(manual)
    return tuple(sorted(values.values(), key=lambda value: _manual_feature_key(value)))


def _residuals(
    projector: ArmProjector,
    states: Mapping[tuple[int, int], SourceFrame],
    features: Iterable[ImageFeature],
    correction: np.ndarray,
    offset_frames: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    residuals: list[float] = []
    records: list[dict[str, Any]] = []
    grouped: dict[tuple[int, int], list[ImageFeature]] = defaultdict(list)
    for feature in features:
        grouped[(feature.episode_index, feature.frame_index)].append(feature)
    q_cache: dict[tuple[int, int], np.ndarray] = {}
    for (episode_index, frame_index), candidates in grouped.items():
        frame = states[(episode_index, frame_index)]
        q_key = (episode_index, frame_index + offset_frames)
        if q_key not in q_cache:
            q, _ = _read_episode_states_for_cache(states, episode_index, frame_index + offset_frames)
            q_cache[q_key] = q
        for feature in candidates:
            predicted = _project_landmark(projector, q_cache[q_key], feature.side, feature.landmark, correction)
            if predicted is None or _clip_pixel(predicted) is None:
                continue
            error = float(np.linalg.norm(predicted - feature.pixel_xy))
            weight = 1.0 if feature.origin == "manual" else 0.45 + 0.55 * feature.confidence
            residuals.append(error * weight)
            records.append({
                "episode_index": episode_index, "frame_index": frame_index, "split": feature.split,
                "side": feature.side, "landmark": feature.landmark, "origin": feature.origin,
                "confidence": feature.confidence, "observed_pixel_xy": feature.pixel_xy.tolist(),
                "projected_pixel_xy": np.asarray(predicted).tolist(), "error_px": error,
            })
    return np.asarray(residuals, dtype=np.float64), records


def _read_episode_states_for_cache(
    states: Mapping[tuple[int, int], SourceFrame], episode_index: int, requested_frame: int
) -> tuple[np.ndarray, bool]:
    """Return a source q for an offset frame without touching image alignment."""

    # `_run_head` registers full matrices on this attribute to keep per-offset
    # source state lookup explicit and prevent shifted RGB/labels.
    q_by_episode = getattr(_read_episode_states_for_cache, "q_by_episode", None)
    if not isinstance(q_by_episode, dict) or episode_index not in q_by_episode:
        raise RuntimeError("internal source-state cache is unavailable")
    q = q_by_episode[episode_index]
    if not 0 <= requested_frame < len(q):
        raise ValueError("q_current offset leaves the selected source episode")
    return _require_q(q[requested_frame], "offset robot_q_current"), True


def _fit_for_offset(
    projector: ArmProjector,
    states: Mapping[tuple[int, int], SourceFrame],
    features: tuple[ImageFeature, ...],
    offset_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from scipy.optimize import least_squares

    fit_features = tuple(feature for feature in features if feature.split == "fit")
    if len(fit_features) < 12:
        raise ValueError("at least 12 fit arm features are required")

    def residual(value: np.ndarray) -> np.ndarray:
        result, _ = _residuals(projector, states, fit_features, np.asarray(value, dtype=np.float64), offset_frames)
        if len(result) < 12:
            return np.full(12, 1.0e4, dtype=np.float64)
        return result

    bounds = np.asarray(([-MAX_OFFSET_M] * 3 + [-MAX_OFFSET_DEG] * 3, [MAX_OFFSET_M] * 3 + [MAX_OFFSET_DEG] * 3), dtype=np.float64)
    result = least_squares(residual, x0=np.zeros(6), bounds=(bounds[0], bounds[1]), loss="huber", f_scale=3.0, max_nfev=300)
    holdout_residuals, records = _residuals(projector, states, features, result.x, offset_frames)
    return np.asarray(result.x, dtype=np.float64), holdout_residuals, records


def _draw_review(image: np.ndarray, records: Iterable[dict[str, Any]]) -> np.ndarray:
    output = _require_image(image, "review image").copy()
    colors = {"left": (40, 220, 40), "right": (40, 40, 220)}
    for item in records:
        observed = tuple(np.rint(item["observed_pixel_xy"]).astype(int))
        predicted = tuple(np.rint(item["projected_pixel_xy"]).astype(int))
        color = colors[item["side"]]
        cv2.circle(output, observed, 5, color, 1, cv2.LINE_AA)
        cv2.drawMarker(output, predicted, color, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.line(output, observed, predicted, color, 1, cv2.LINE_AA)
    return output


def run_head_calibration(
    *, source_root: Path, episodes: tuple[int, ...], urdf: Path, output_dir: Path,
    manual_keypoints: Path | None = None, count_per_episode: int = 4,
) -> dict[str, Any]:
    """Fit a shared head mount from static real arm configurations."""

    if len(set(episodes)) != len(episodes) or len(episodes) < 2:
        raise ValueError("at least two distinct calibration episodes are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_count = max(count_per_episode * 3, 12)
    candidate_frames_by_episode = selected_source_frames(source_root, episodes, count_per_episode=candidate_count)
    candidate_source_frames = _source_frames(
        source_root=source_root, selected=candidate_frames_by_episode, output_dir=output_dir
    )
    q_by_episode = {episode: _read_episode_states(source_root, episode)[0] for episode in episodes}
    setattr(_read_episode_states_for_cache, "q_by_episode", q_by_episode)
    config = load_pipeline_config(DEFAULT_CONFIG_PATH)
    camera = next(item for item in config.cameras if item.source_key == HEAD_CAMERA_KEY)
    projector = ArmProjector(urdf, camera)
    candidate_automatic = tuple(
        feature for frame in candidate_source_frames for feature in automatic_features(frame, projector)
    )
    source_frames = _select_visible_frames(
        candidate_source_frames, candidate_automatic, count_per_episode=count_per_episode
    )
    selected_keys = {(item.episode_index, item.frame_index): item.split for item in source_frames}
    automatic = tuple(
        replace(feature, split=selected_keys[(feature.episode_index, feature.frame_index)])
        for feature in candidate_automatic
        if (feature.episode_index, feature.frame_index) in selected_keys
    )
    state_by_key = {(item.episode_index, item.frame_index): item for item in source_frames}
    manual = load_manual_features(manual_keypoints)
    features = _features_with_manual_overrides(automatic, manual)
    if not features:
        raise ValueError("automatic extraction found no valid arm features; provide manual keypoints")

    candidates: list[dict[str, Any]] = []
    for offset in range(-MAX_OFFSET_FRAMES, MAX_OFFSET_FRAMES + 1):
        try:
            correction, _, records = _fit_for_offset(projector, state_by_key, features, offset)
        except ValueError as error:
            candidates.append({"offset_frames": offset, "valid": False, "reason": str(error)})
            continue
        holdout = [item["error_px"] for item in records if item["split"] == "holdout"]
        fit = [item["error_px"] for item in records if item["split"] == "fit"]
        candidates.append({
            "offset_frames": offset, "valid": True, "correction": correction.tolist(), "fit_median_px": float(np.median(fit)) if fit else None,
            "holdout_median_px": float(np.median(holdout)) if holdout else None,
            "holdout_p95_px": float(np.quantile(holdout, .95)) if holdout else None,
            "records": records,
        })
    valid = [item for item in candidates if item.get("valid")]
    if not valid:
        raise ValueError("no time-offset candidate had enough visible arm features")
    best = min(valid, key=lambda item: (float(item["holdout_p95_px"] or np.inf), abs(int(item["offset_frames"]))))
    correction = np.asarray(best["correction"], dtype=np.float64)
    report_records = best.pop("records")
    holdout_points = sum(item["split"] == "holdout" for item in report_records)
    manual_points = sum(item["origin"] == "manual" for item in report_records)
    accepted, decision = can_accept_head_calibration(
        median_px=best["holdout_median_px"], p95_px=best["holdout_p95_px"], holdout_points=holdout_points,
        manual_points=manual_points,
    )
    for frame in source_frames:
        rows = [item for item in report_records if item["episode_index"] == frame.episode_index and item["frame_index"] == frame.frame_index]
        preview = _draw_review(frame.image_bgr, rows)
        path = output_dir / "review" / f"episode_{frame.episode_index:04d}" / f"frame_{frame.frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), preview):
            raise OSError(f"could not write {path}")
    manual_candidates = []
    for frame in source_frames:
        projected = projector.project(
            frame.robot_q_current, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
        )
        automatic_by_key = {_manual_feature_key(item): item for item in automatic}
        for side in ("left", "right"):
            for landmark, index in LANDMARK_INDEX.items():
                pixel = np.asarray(projected[side][index], dtype=np.float64)
                if _clip_pixel(pixel) is None:
                    continue
                key = (frame.episode_index, frame.frame_index, side, landmark)
                automatic_feature = automatic_by_key.get(key)
                manual_candidates.append({
                    "episode_index": frame.episode_index, "frame_index": frame.frame_index, "split": frame.split,
                    "side": side, "landmark": landmark, "pixel_xy": pixel.tolist(),
                    "automatic_confidence": None if automatic_feature is None else automatic_feature.confidence,
                    "note": "Keep only a visible elbow, wrist, or hand-base point. Edit pixel_xy to its real RGB location.",
                })
    template = {
        "schema_version": SCHEMA_VERSION,
        "manual_features": manual_candidates,
    }
    atomic_write_json(output_dir / "manual_keypoints.template.json", template)
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera calibration only",
        "source": {"dataset_root": str(source_root), "episode_indices": list(episodes), "camera": HEAD_CAMERA_KEY,
                   "intrinsics": "fixed raw calibration from pipeline_v1.json", "stereo_baseline": "fixed raw calibration"},
        "selection": {"static_speed_limit_rad_s": MAX_STATIC_SPEED_RAD_S, "count_per_episode": count_per_episode,
                      "candidate_frames_by_episode": {str(key): list(value) for key, value in candidate_frames_by_episode.items()},
                      "frames_by_episode": {str(episode): [item.frame_index for item in source_frames if item.episode_index == episode] for episode in episodes},
                      "split_rule": "two arm-visible poses per calibration episode are held out after joint-diversity selection"},
        "feature_method": {"automatic": "Canny edge association local to nominal encoder-FK landmarks", "manual_override": str(manual_keypoints) if manual_keypoints else None,
                           "manual_template": "manual_keypoints.template.json", "automatic_features": len(automatic), "features_used": len(features)},
        "optimization": {"shared_parameters": ["torso_from_head_left_translation_m", "torso_from_head_left_rotation_xyz_deg", "rgb_to_q_current_offset_frames"],
                         "bounds": {"translation_m": [-MAX_OFFSET_M, MAX_OFFSET_M], "rotation_deg": [-MAX_OFFSET_DEG, MAX_OFFSET_DEG], "time_offset_frames": [-MAX_OFFSET_FRAMES, MAX_OFFSET_FRAMES]},
                         "candidates": candidates, "selected": best},
        "shared_head_left_correction": {"torso_offset_m": correction[:3].tolist(), "camera_rotation_rpy_deg": correction[3:].tolist(), "rgb_to_q_current_offset_frames": int(best["offset_frames"])},
        "heldout": {"points": holdout_points, "manual_points": manual_points, "median_px": best["holdout_median_px"], "p95_px": best["holdout_p95_px"],
                    "gate": {"median_px_max": HEAD_MEDIAN_GATE_PX, "p95_px_max": HEAD_P95_GATE_PX,
                             "manual_review": "required only for low-confidence or visibly incorrect automatic features"}},
        "accepted_for_scene_calibration": accepted,
        "decision": decision,
        "feature_records": report_records,
        "limitations": [
            "Automatic edge associations are provisional; they are not semantic robot keypoints.",
            "Low-confidence or visibly incorrect automatic features must be corrected through the generated manual template before re-running.",
            "The dataset has no independent camera-to-link transform; this result retains residual uncertainty.",
            "This program never shifts RGB, rewrites labels, changes simulator defaults, or exposes calibration to a policy.",
        ],
    }
    atomic_write_json(output_dir / "head_camera_calibration.json", report)
    return report


def _accepted_head_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("head report schema is unexpected")
    if report.get("accepted_for_scene_calibration") is not True:
        raise ValueError("head calibration did not pass its held-out gate")
    return report


def _scene_pose_delta(fitted: np.ndarray, heldout: np.ndarray) -> tuple[float, float]:
    """Compare table poses while respecting the physical 180-degree symmetry."""

    fit = np.asarray(fitted, dtype=np.float64)
    validation = np.asarray(heldout, dtype=np.float64)
    alternatives = (validation, validation @ TABLE_YAW_180_SYMMETRY)
    selected = min(
        alternatives,
        key=lambda value: Rotation.from_matrix(fit[:3, :3].T @ value[:3, :3]).magnitude(),
    )
    translation = float(np.linalg.norm(fit[:3, 3] - selected[:3, 3]))
    rotation = float(np.degrees(Rotation.from_matrix(fit[:3, :3].T @ selected[:3, :3]).magnitude()))
    return translation, rotation


def _workbench_observability(scene: Mapping[str, Any]) -> dict[str, Any]:
    """State exactly what the head-stereo evidence can identify about the bench.

    The source table CAD establishes the support plane through its contact
    geometry, but it cannot supply an arbitrary workbench origin or yaw when
    no metrically known bench edge is visible in both eyes.  Reporting that
    ambiguity is safer than inventing a workbench transform and accidentally
    treating it as a simulator reset target.
    """

    source = scene.get("fitted")
    if not isinstance(source, Mapping) or not isinstance(source.get("fixed_scene_root_from_table"), list):
        return {
            "status": "unavailable",
            "reason": "table CAD fit was not accepted; no workbench plane can be constrained",
        }
    return {
        "status": "not_fully_identifiable_from_table_only",
        "identified_from_table_cad": [
            "support plane normal and height in robot-root coordinates, subject to the fixed CAD contact datum",
        ],
        "not_identified": [
            "root_from_workbench translation parallel to the support plane",
            "root_from_workbench yaw",
        ],
        "required_to_promote_to_root_from_workbench": [
            "a metrically known workbench outer edge or fiducial visible in both head cameras",
        ],
        "policy_use": "forbidden: offline scene-identifiability record only",
    }


def _scene_fit_frame_selection(screened: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Select fit frames using stereo *and* multi-frame fixed-scene agreement.

    Per-frame left/right agreement alone can select three mutually displaced
    CAD registrations.  That would silently force the D405 mount fit to
    explain table-pose noise.  Score every three-frame combination using the
    same robust table-pose aggregation used by the persisted source artifact.
    """

    candidates: list[tuple[int, tuple[PoseEstimate, PoseEstimate], Mapping[str, Any]]] = []
    for frame in screened.get("frames", []):
        stereo = frame.get("stereo_agreement")
        eyes = frame.get("eyes")
        if not isinstance(stereo, Mapping) or stereo.get("accepted") is not True or not isinstance(eyes, Mapping):
            continue
        left, right = eyes.get("head_left"), eyes.get("head_right")
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        try:
            frame_index = int(frame["frame_index"])
            pair = (
                PoseEstimate(frame_index, "head_left", np.asarray(left["root_from_table"], dtype=np.float64),
                             float(left["confidence"]), float(left["cad_edge_error_px"])),
                PoseEstimate(frame_index, "head_right", np.asarray(right["root_from_table"], dtype=np.float64),
                             float(right["confidence"]), float(right["cad_edge_error_px"])),
            )
            if any(value.root_from_table.shape != (4, 4) for value in pair):
                continue
            candidates.append((frame_index, pair, stereo))
        except (KeyError, TypeError, ValueError):
            continue
    if len(candidates) < 6:
        raise ValueError("fewer than six screened stereo-consistent scene frames for fit/hold-out")

    def score(indices: tuple[int, int, int]) -> tuple[float, float, float, float, tuple[int, ...]]:
        selected = [candidates[index] for index in indices]
        estimates = tuple(estimate for _, pair, _ in selected for estimate in pair)
        _, temporal = robust_fixed_pose(estimates)
        pair_translation = [float(item[2]["translation_m"]) for item in selected]
        pair_rotation = [float(item[2]["rotation_deg"]) for item in selected]
        return (
            float(temporal["translation_spread_p95_m"]),
            float(temporal["rotation_spread_p95_deg"]),
            float(np.quantile(pair_translation, 0.95)),
            float(np.quantile(pair_rotation, 0.95)),
            tuple(item[0] for item in selected),
        )

    selected_indices = min(combinations(range(len(candidates)), 3), key=score)
    fit_frames = tuple(candidates[index][0] for index in selected_indices)
    heldout_frames = tuple(
        item[0] for index, item in enumerate(candidates) if index not in selected_indices
    )
    return fit_frames, heldout_frames


def run_scene_calibration(
    *,
    source_root: Path,
    head_report_path: Path,
    urdf: Path,
    stereo_calibration: Path,
    output_dir: Path,
    scene_frame_maximum: int = SCENE_FRAME_MAXIMUM_DEFAULT,
    scene_frame_stride: int = SCENE_FRAME_STRIDE_DEFAULT,
) -> dict[str, Any]:
    """Run table/workbench source-CAD fitting only after the head gate passes."""

    if scene_frame_maximum < 5:
        raise ValueError("scene_frame_maximum must be at least 5")
    if scene_frame_stride <= 0:
        raise ValueError("scene_frame_stride must be positive")
    head = _accepted_head_report(head_report_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    correction = head["shared_head_left_correction"]
    results: dict[str, Any] = {}
    # The arm-mount stage deliberately chooses visually diverse arm poses.
    # Scene geometry instead needs a stationary, pre-manipulation table.  Use
    # a dense early interval and let the strict stereo-CAD gate reject it if
    # that episode is not actually static/visible.
    offset = int(head["shared_head_left_correction"]["rgb_to_q_current_offset_frames"])
    scene_frames = tuple(range(max(0, -offset), scene_frame_maximum + 1, scene_frame_stride))
    for episode_text in head["selection"]["frames_by_episode"]:
        episode = int(episode_text)
        try:
            screened = align_source_episode(
                source_root=source_root, episode_index=episode, frames=scene_frames, urdf=urdf,
                stereo_calibration=stereo_calibration, output_dir=output_dir / f"episode_{episode:04d}" / "screen",
                head_correction=correction, q_current_offset_frames=offset,
            )
            fit_frames, heldout_frames = _scene_fit_frame_selection(screened)
            fitted = align_source_episode(
                source_root=source_root, episode_index=episode, frames=fit_frames, urdf=urdf,
                stereo_calibration=stereo_calibration, output_dir=output_dir / f"episode_{episode:04d}" / "fit",
                head_correction=correction, q_current_offset_frames=offset,
            )
            heldout = align_source_episode(
                source_root=source_root, episode_index=episode, frames=heldout_frames, urdf=urdf,
                stereo_calibration=stereo_calibration, output_dir=output_dir / f"episode_{episode:04d}" / "heldout",
                head_correction=correction, q_current_offset_frames=offset,
            )
            translation, rotation = _scene_pose_delta(
                np.asarray(fitted["fixed_scene_root_from_table"]), np.asarray(heldout["fixed_scene_root_from_table"])
            )
            accepted = bool(
                fitted.get("accepted_for_fixed_scene_proposal")
                and heldout.get("accepted_for_fixed_scene_proposal")
                and translation <= SCENE_HOLDOUT_TRANSLATION_GATE_M
                and rotation <= SCENE_HOLDOUT_ROTATION_GATE_DEG
            )
            results[str(episode)] = {
                "accepted_for_fixed_scene_proposal": accepted,
                "screened": screened,
                "fitted": fitted,
                "heldout": heldout,
                "heldout_pose_delta": {"translation_m": translation, "rotation_deg": rotation,
                                        "translation_gate_m": SCENE_HOLDOUT_TRANSLATION_GATE_M,
                                        "rotation_gate_deg": SCENE_HOLDOUT_ROTATION_GATE_DEG},
                "workbench_observability": _workbench_observability({"fitted": fitted}),
            }
        except (OSError, RuntimeError, ValueError) as error:
            results[str(episode)] = {"accepted_for_fixed_scene_proposal": False, "rejection_reason": str(error)}
    report = {"schema_version": SCHEMA_VERSION, "policy_use": "forbidden: offline scene calibration only", "head_report": str(head_report_path),
              "head_gate_passed": True, "scene_frame_candidates": list(scene_frames),
              "scene_frame_selection": {"maximum_frame": scene_frame_maximum, "stride": scene_frame_stride}, "episodes": results,
              "accepted_for_wrist_calibration": all(item.get("accepted_for_fixed_scene_proposal") for item in results.values())}
    atomic_write_json(output_dir / "scene_calibration.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("head", "scene"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=(184, 250))
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-keypoints", type=Path)
    parser.add_argument("--head-report", type=Path)
    parser.add_argument("--stereo-calibration", type=Path)
    parser.add_argument("--scene-frame-maximum", type=int, default=SCENE_FRAME_MAXIMUM_DEFAULT)
    parser.add_argument("--scene-frame-stride", type=int, default=SCENE_FRAME_STRIDE_DEFAULT)
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    if args.stage == "head":
        report = run_head_calibration(source_root=source_root, episodes=tuple(args.episodes), urdf=args.urdf.expanduser().resolve(),
                                      output_dir=args.output_dir.expanduser().resolve(), manual_keypoints=args.manual_keypoints.expanduser().resolve() if args.manual_keypoints else None)
        print(json.dumps({"accepted": report["accepted_for_scene_calibration"], "decision": report["decision"], "heldout": report["heldout"]}, indent=2))
        return
    if args.head_report is None or args.stereo_calibration is None:
        raise ValueError("scene stage requires --head-report and --stereo-calibration")
    report = run_scene_calibration(source_root=source_root, head_report_path=args.head_report.expanduser().resolve(), urdf=args.urdf.expanduser().resolve(),
                                   stereo_calibration=args.stereo_calibration.expanduser().resolve(), output_dir=args.output_dir.expanduser().resolve(),
                                   scene_frame_maximum=args.scene_frame_maximum, scene_frame_stride=args.scene_frame_stride)
    print(json.dumps({"accepted_for_wrist_calibration": report["accepted_for_wrist_calibration"]}, indent=2))


if __name__ == "__main__":
    main()
