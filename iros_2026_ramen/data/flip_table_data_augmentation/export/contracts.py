"""Validated contracts for accepted, rendered synthetic episodes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ..config import EXPECTED_CAMERA_KEYS
from ..fk_audit import SYNTHETIC_ACTION_FK_SCHEMA_VERSION
from ..io_utils import sha256_file
from ..source_contract import NUMERIC_FEATURES


RENDER_MANIFEST_SCHEMA_VERSION = "team_ramen_flip_table_rendered_episode/v1"
LINEAGE_SCHEMA_VERSION = "team_ramen_flip_table_episode_lineage/v1"
NUMERIC_KEYS = tuple(NUMERIC_FEATURES)
TASK = "flip table"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _relative_file(root: Path, value: Any, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a safe relative path")
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"{label} does not resolve to a file under the episode root")
    return path


@dataclass(frozen=True)
class RenderedEpisode:
    root: Path
    manifest_path: Path
    candidate_id: str
    trajectory_kind: str
    source_kind: str
    appearance_variant: int
    source_episode_indices: tuple[int, ...]
    source_trajectory_lineage: str
    frame_count: int
    fps: int
    task: str
    numeric_trace: Path
    camera_dirs: dict[str, Path]
    trajectory_sha256: str
    runtime_manifest_sha256: str
    config_sha256: str
    randomization: dict[str, Any]
    success_report: dict[str, Any]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "RenderedEpisode":
        manifest = Path(manifest_path).expanduser().resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != RENDER_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported rendered episode manifest: {manifest}")
        root = manifest.parent.resolve()
        candidate_id = _text(payload.get("candidate_id"), "candidate_id")
        trajectory_kind = _text(payload.get("trajectory_kind"), "trajectory_kind")
        source_kind = _text(payload.get("source_kind"), "source_kind")
        if trajectory_kind not in {"direct_sim_teleop", "mimic"}:
            raise ValueError("trajectory_kind must be direct_sim_teleop or mimic")
        if source_kind not in {"real_demo", "sim_teleop"}:
            raise ValueError("source_kind must be real_demo or sim_teleop")
        if trajectory_kind == "direct_sim_teleop" and source_kind != "sim_teleop":
            raise ValueError("a direct sim teleop trajectory must have sim_teleop source")
        appearance_variant = payload.get("appearance_variant")
        frame_count = payload.get("frame_count")
        fps = payload.get("fps")
        if (
            isinstance(appearance_variant, bool)
            or not isinstance(appearance_variant, int)
            or appearance_variant < 0
        ):
            raise ValueError("appearance_variant must be a non-negative integer")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")
        if fps != 30:
            raise ValueError("rendered policy images must be synchronized at 30 fps")

        raw_source_indices = payload.get("source_episode_indices")
        if not isinstance(raw_source_indices, list) or not raw_source_indices:
            raise ValueError("source_episode_indices must be a non-empty list")
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in raw_source_indices):
            raise ValueError("source_episode_indices must contain non-negative integers")
        source_indices = tuple(sorted(set(raw_source_indices)))
        if len(source_indices) != len(raw_source_indices):
            raise ValueError("source_episode_indices must be sorted and unique")

        success_report = payload.get("success_report")
        if not isinstance(success_report, dict):
            raise ValueError("success_report must be an object")
        if success_report.get("accepted") is not True or success_report.get("strict_v1_contract") is not True:
            raise ValueError("only strict-V1 accepted trajectories may be exported")
        reasons = success_report.get("rejection_reasons")
        if reasons not in (None, []):
            raise ValueError("accepted trajectory cannot contain rejection reasons")
        action_fk_report = success_report.get("action_fk_report")
        if (
            not isinstance(action_fk_report, dict)
            or action_fk_report.get("schema_version") != SYNTHETIC_ACTION_FK_SCHEMA_VERSION
            or action_fk_report.get("pass") is not True
            or isinstance(action_fk_report.get("frame_count"), bool)
            or not isinstance(action_fk_report.get("frame_count"), int)
            or action_fk_report["frame_count"] <= 0
        ):
            raise ValueError("accepted trajectory lacks a passing full-rate action FK audit")

        randomization = payload.get("randomization")
        if not isinstance(randomization, dict) or not randomization:
            raise ValueError("randomization must record the applied appearance and physical parameters")
        trajectory_sampling = randomization.get("trajectory_sampling")
        if (
            not isinstance(trajectory_sampling, dict)
            or trajectory_sampling.get("source_frame_count")
            != action_fk_report["frame_count"]
        ):
            raise ValueError("render sampling and full-rate action FK audit frame counts differ")
        forbidden = {"object_pose_observation", "contact_observation", "segmentation_observation"}
        if forbidden.intersection(payload):
            raise ValueError("sim-only teacher signals cannot be serialized as policy features")

        numeric_trace = _relative_file(root, payload.get("numeric_trace"), "numeric_trace")
        if sha256_file(numeric_trace) != _sha256(
            payload.get("numeric_trace_sha256"), "numeric_trace_sha256"
        ):
            raise ValueError("numeric trace SHA-256 does not match the rendered episode manifest")

        cameras = payload.get("cameras")
        if not isinstance(cameras, dict) or tuple(cameras) != EXPECTED_CAMERA_KEYS:
            raise ValueError(f"cameras must contain ordered keys {EXPECTED_CAMERA_KEYS}")
        camera_dirs: dict[str, Path] = {}
        for key in EXPECTED_CAMERA_KEYS:
            relative = Path(_text(cameras[key], f"cameras.{key}"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"cameras.{key} must be a safe relative path")
            directory = (root / relative).resolve()
            if root not in directory.parents or not directory.is_dir():
                raise ValueError(f"cameras.{key} does not resolve to a directory under the episode root")
            expected = [directory / f"frame_{index:06d}.png" for index in range(frame_count)]
            if any(not path.is_file() for path in expected):
                raise ValueError(f"cameras.{key} is missing one or more contiguous rendered frames")
            extras = [path for path in directory.glob("frame_*.png") if path not in set(expected)]
            if extras:
                raise ValueError(f"cameras.{key} contains frames outside [0,{frame_count})")
            camera_dirs[key] = directory

        task = _text(payload.get("task"), "task")
        if task != TASK:
            raise ValueError(f"rendered episode task must be {TASK!r}")

        return cls(
            root=root,
            manifest_path=manifest,
            candidate_id=candidate_id,
            trajectory_kind=trajectory_kind,
            source_kind=source_kind,
            appearance_variant=appearance_variant,
            source_episode_indices=source_indices,
            source_trajectory_lineage=_text(
                payload.get("source_trajectory_lineage"), "source_trajectory_lineage"
            ),
            frame_count=frame_count,
            fps=fps,
            task=task,
            numeric_trace=numeric_trace,
            camera_dirs=camera_dirs,
            trajectory_sha256=_sha256(payload.get("trajectory_sha256"), "trajectory_sha256"),
            runtime_manifest_sha256=_sha256(
                payload.get("runtime_manifest_sha256"), "runtime_manifest_sha256"
            ),
            config_sha256=_sha256(payload.get("config_sha256"), "config_sha256"),
            randomization=randomization,
            success_report=success_report,
        )


def lineage_split(lineage: str, weights: dict[str, float]) -> str:
    """Assign one immutable lineage to a deterministic dataset split."""

    ordered = ("train", "validation", "test")
    if tuple(weights) != ordered:
        raise ValueError(f"split weights must contain ordered keys {ordered}")
    values = tuple(float(weights[name]) for name in ordered)
    if any(not math.isfinite(value) or value <= 0.0 for value in values) or not math.isclose(
        sum(values), 1.0, abs_tol=1.0e-9
    ):
        raise ValueError("split weights must be finite, positive, and sum to one")
    bucket = int.from_bytes(hashlib.sha256(lineage.encode("utf-8")).digest()[:8], "big") / 2**64
    cumulative = 0.0
    for name, weight in zip(ordered, values, strict=True):
        cumulative += weight
        if bucket < cumulative:
            return name
    return "test"
