"""Validated object-pose and multi-EEF subtask annotations.

Annotations are offline generation evidence. They are never policy inputs.
Every accepted pose and phase annotation is tied to immutable evidence by SHA-256.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .config import EXPECTED_SUBTASKS


ANNOTATION_SCHEMA_VERSION = "team_ramen_flip_table_source_annotation/v2"
EXPECTED_EEFS = ("left", "right")
_METRIC_UNITS = {"m", "rad", "px"}


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def _non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    result = _non_empty_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class QualityMetric:
    name: str
    value: float
    unit: str
    maximum: float

    @classmethod
    def from_json(cls, value: Any, label: str) -> "QualityMetric":
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        name = _non_empty_string(value.get("name"), f"{label}.name")
        unit = _non_empty_string(value.get("unit"), f"{label}.unit")
        if unit not in _METRIC_UNITS:
            raise ValueError(f"{label}.unit must be one of {sorted(_METRIC_UNITS)}")
        measured = float(value.get("value"))
        maximum = float(value.get("maximum"))
        if not math.isfinite(measured) or measured < 0.0:
            raise ValueError(f"{label}.value must be finite and non-negative")
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError(f"{label}.maximum must be finite and positive")
        if measured > maximum:
            raise ValueError(
                f"{label} failed its acceptance gate: {measured} {unit} > {maximum} {unit}"
            )
        return cls(name=name, value=measured, unit=unit, maximum=maximum)


@dataclass(frozen=True)
class PoseEvidence:
    method: str
    reviewer: str
    calibration_artifact_sha256: str
    metrics: tuple[QualityMetric, ...]

    @classmethod
    def from_json(cls, value: Any) -> "PoseEvidence":
        if not isinstance(value, dict):
            raise ValueError("pose_evidence must be an object")
        method = _non_empty_string(value.get("method"), "pose_evidence.method")
        if method.lower() in {"guess", "guessed", "manual_guess"}:
            raise ValueError("pose_evidence.method must identify a measured calibration method")
        reviewer = _non_empty_string(value.get("reviewer"), "pose_evidence.reviewer")
        artifact_sha = _sha256(
            value.get("calibration_artifact_sha256"),
            "pose_evidence.calibration_artifact_sha256",
        )
        raw_metrics = value.get("quality_metrics")
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ValueError("pose_evidence.quality_metrics must be a non-empty list")
        metrics = tuple(
            QualityMetric.from_json(metric, f"pose_evidence.quality_metrics[{index}]")
            for index, metric in enumerate(raw_metrics)
        )
        names = [metric.name for metric in metrics]
        if len(names) != len(set(names)):
            raise ValueError("pose_evidence quality metric names must be unique")
        return cls(
            method=method,
            reviewer=reviewer,
            calibration_artifact_sha256=artifact_sha,
            metrics=metrics,
        )


@dataclass(frozen=True)
class FrameRange:
    start: int
    end: int

    @classmethod
    def from_json(cls, value: Any, label: str) -> "FrameRange":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"{label} must be [start, end)")
        start, end = value
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"{label} values must be integers")
        if start < 0 or end <= start:
            raise ValueError(f"{label} must satisfy 0 <= start < end")
        return cls(start=start, end=end)


@dataclass(frozen=True)
class SourceEpisodeAnnotation:
    episode_index: int
    frame_count: int
    table_pose_trajectory_robot_root_xyzw: tuple[tuple[float, ...], ...]
    pose_evidence: PoseEvidence
    subtask_reviewer: str
    subtask_evidence_sha256: str
    subtasks: dict[str, dict[str, FrameRange]]

    @classmethod
    def from_json(cls, value: Any) -> "SourceEpisodeAnnotation":
        if not isinstance(value, dict):
            raise ValueError("episode annotation must be an object")
        episode_index = value.get("episode_index")
        frame_count = value.get("frame_count")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError("episode_index must be a non-negative integer")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")
        raw_trajectory = value.get("table_pose_trajectory_robot_root_xyzw")
        if not isinstance(raw_trajectory, list) or len(raw_trajectory) != frame_count:
            raise ValueError(f"table pose trajectory must contain exactly {frame_count} poses")
        trajectory = tuple(
            _finite_vector(pose, 7, f"table pose trajectory[{index}]")
            for index, pose in enumerate(raw_trajectory)
        )
        for index, pose in enumerate(trajectory):
            quat_norm = math.sqrt(sum(item * item for item in pose[3:]))
            if not math.isclose(quat_norm, 1.0, abs_tol=1e-4):
                raise ValueError(
                    f"table pose trajectory[{index}] quaternion must be unit length, got {quat_norm}"
                )
        pose_evidence = PoseEvidence.from_json(value.get("pose_evidence"))
        subtask_reviewer = _non_empty_string(value.get("subtask_reviewer"), "subtask_reviewer")
        subtask_evidence_sha256 = _sha256(
            value.get("subtask_evidence_sha256"), "subtask_evidence_sha256"
        )

        parsed = parse_subtasks(value.get("subtasks"), frame_count)
        return cls(
            episode_index=episode_index,
            frame_count=frame_count,
            table_pose_trajectory_robot_root_xyzw=trajectory,
            pose_evidence=pose_evidence,
            subtask_reviewer=subtask_reviewer,
            subtask_evidence_sha256=subtask_evidence_sha256,
            subtasks=parsed,
        )


def parse_subtasks(value: Any, frame_count: int) -> dict[str, dict[str, FrameRange]]:
    """Validate complete, monotonic, bimanually coordinated phase ranges."""

    if not isinstance(value, dict) or set(value) != set(EXPECTED_EEFS):
        raise ValueError(f"subtasks must contain EEF keys {EXPECTED_EEFS}")
    parsed: dict[str, dict[str, FrameRange]] = {}
    for eef in EXPECTED_EEFS:
        eef_values = value[eef]
        if not isinstance(eef_values, dict) or set(eef_values) != set(EXPECTED_SUBTASKS):
            raise ValueError(f"{eef} must contain subtasks {EXPECTED_SUBTASKS}")
        parsed[eef] = {
            name: FrameRange.from_json(eef_values[name], f"{eef}.{name}")
            for name in EXPECTED_SUBTASKS
        }
        ranges = list(parsed[eef].values())
        if ranges[0].start != 0 or ranges[-1].end != frame_count:
            raise ValueError(f"{eef} subtasks must cover the complete episode")
        for previous, current in zip(ranges, ranges[1:]):
            if previous.end != current.start:
                raise ValueError(f"{eef} subtasks must be contiguous and monotonic")
    for name in ("grasp", "lift", "rotate_180", "settle", "release"):
        left, right = parsed["left"][name], parsed["right"][name]
        if left.start != right.start or left.end != right.end:
            raise ValueError(f"bimanual {name} boundaries must be synchronized")
    return parsed


def load_annotations(path: str | Path) -> tuple[SourceEpisodeAnnotation, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported annotation schema")
    values = payload.get("episodes")
    if not isinstance(values, list) or not values:
        raise ValueError("annotations must contain at least one episode")
    episodes = tuple(SourceEpisodeAnnotation.from_json(value) for value in values)
    indices = [episode.episode_index for episode in episodes]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("episode annotations must be unique and sorted")
    return episodes
