#!/usr/bin/env python3
"""Audit the 30 successful DR-stratified simulator teleoperation demos."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from data.flip_table_data_augmentation.teleop.config import load_teleop_config
from data.flip_table_data_augmentation.teleop.raw_episode import (
    DEFAULT_MAXIMUM_CAMERA_AGE_S,
    POLICY_CAMERA_ROLE_TO_KEY,
    RAW_EPISODE_SCHEMA_VERSION,
)


REQUIRED_RANDOMIZATION_KEYS = {
    "table",
    "robot",
    "contact_materials",
    "camera_mounts",
    "camera_image",
    "control",
    "lighting",
    "room",
}
_MIN_CAMERA_INTERVAL_NS = 15_000_000
_MAX_CAMERA_INTERVAL_NS = 50_000_000
_MIN_MEAN_CAMERA_INTERVAL_NS = 25_000_000
_MAX_MEAN_CAMERA_INTERVAL_NS = 42_000_000
_MAX_COMMAND_ALIGNMENT_NS = 70_000_000
_MAX_CAMERA_AGE_NS = int(DEFAULT_MAXIMUM_CAMERA_AGE_S * 1.0e9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _first_policy_images(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for role in POLICY_CAMERA_ROLE_TO_KEY:
        image = root / "policy_cameras" / role / "000000.jpg"
        if not image.is_file():
            raise FileNotFoundError(f"missing first policy image: {image}")
        from PIL import Image

        with Image.open(image) as decoded:
            if decoded.size != (640, 480) or decoded.mode != "RGB":
                raise ValueError(f"{image} must be RGB 640x480, got {decoded.mode} {decoded.size}")
        result[role] = sha256_file(image)
    return result


def _timestamp(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer nanosecond timestamp")
    return value


def _audit_trace(root: Path, trace_path: Path, *, expected_count: int) -> dict[str, float]:
    """Check every recorded sample, rather than trusting a manifest summary."""

    observations: list[int] = []
    command_sequences: list[int] = []
    expected_cameras = set(POLICY_CAMERA_ROLE_TO_KEY.values())
    with trace_path.open(encoding="utf-8") as stream:
        for expected_index, line in enumerate(stream):
            if not line.strip():
                raise ValueError("raw trace contains a blank frame record")
            frame = json.loads(line)
            if not isinstance(frame, dict) or frame.get("frame_index") != expected_index:
                raise ValueError("raw trace frame indices must be contiguous from zero")
            observation_ns = _timestamp(
                frame.get("observation_monotonic_ns"),
                f"frame {expected_index} observation timestamp",
            )
            command_ns = _timestamp(
                frame.get("command_monotonic_ns"),
                f"frame {expected_index} command timestamp",
            )
            if abs(command_ns - observation_ns) > _MAX_COMMAND_ALIGNMENT_NS:
                raise ValueError("command and observation are not locally timestamp-aligned")
            observation_sequence = frame.get("observation_sequence")
            command_sequence = frame.get("command_sequence")
            if (
                isinstance(observation_sequence, bool)
                or not isinstance(observation_sequence, int)
                or isinstance(command_sequence, bool)
                or not isinstance(command_sequence, int)
            ):
                raise ValueError("raw trace sequences must be integers")
            if observations and observation_ns <= observations[-1]:
                raise ValueError("observation timestamps must be strictly increasing")
            if command_sequences and command_sequence <= command_sequences[-1]:
                raise ValueError("command sequences must be strictly increasing")
            camera_records = frame.get("policy_cameras")
            if not isinstance(camera_records, dict) or set(camera_records) != expected_cameras:
                raise ValueError("raw trace policy camera schema is not exactly cam_0/cam_2/cam_3")
            for key, record in camera_records.items():
                if not isinstance(record, dict):
                    raise ValueError(f"policy camera record is invalid: {key}")
                relative = record.get("path")
                if not isinstance(relative, str) or Path(relative).is_absolute():
                    raise ValueError(f"policy camera path is invalid: {key}")
                image = root / relative
                if not image.is_file() or sha256_file(image) != record.get("sha256"):
                    raise ValueError(f"policy camera file/hash does not match trace: {key}")
                camera_ns = _timestamp(record.get("capture_monotonic_ns"), f"{key} timestamp")
                if camera_ns > observation_ns or observation_ns - camera_ns > _MAX_CAMERA_AGE_NS:
                    raise ValueError(f"policy camera timestamp is outside the permitted observation window: {key}")
            observations.append(observation_ns)
            command_sequences.append(command_sequence)
    if len(observations) != expected_count:
        raise ValueError("trace frame count differs from manifest")
    if len(observations) < 2:
        raise ValueError("teleop episode needs at least two trace records")
    intervals = [later - earlier for earlier, later in zip(observations, observations[1:])]
    if min(intervals) < _MIN_CAMERA_INTERVAL_NS or max(intervals) > _MAX_CAMERA_INTERVAL_NS:
        raise ValueError("camera/record cadence is outside the 30 Hz 50 Hz-servo schedule")
    mean_interval = sum(intervals) / len(intervals)
    if not _MIN_MEAN_CAMERA_INTERVAL_NS <= mean_interval <= _MAX_MEAN_CAMERA_INTERVAL_NS:
        raise ValueError("mean camera/record cadence is not 30 Hz")
    return {
        "record_interval_mean_ms": mean_interval / 1.0e6,
        "record_interval_min_ms": min(intervals) / 1.0e6,
        "record_interval_max_ms": max(intervals) / 1.0e6,
    }


def _audit_episode(root: Path, *, config_digest: str, runtime_digest: str) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    trace_path = root / "frames.jsonl"
    manifest = _object(manifest_path)
    if manifest.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION:
        raise ValueError("unsupported raw episode schema")
    if manifest.get("backend") != "sim" or manifest.get("success") is not True:
        raise ValueError("only successful simulator demos are eligible")
    if manifest.get("config_sha256") != config_digest:
        raise ValueError("teleop config digest differs from the active pinned config")
    if manifest.get("runtime_digest") != runtime_digest:
        raise ValueError("RoboFinals runtime digest differs from the pinned V1 image")
    if manifest.get("fps") != 30 or int(manifest.get("frame_count", 0)) < 2:
        raise ValueError("episode must contain at least two 30 Hz frames")
    if manifest.get("policy_camera_keys") != list(POLICY_CAMERA_ROLE_TO_KEY.values()):
        raise ValueError("policy camera schema is not exactly cam_0/cam_2/cam_3")
    if manifest.get("operator_only_cameras") != ["head_right"]:
        raise ValueError("head-right must remain operator-only")
    if manifest.get("privileged_policy_features") != []:
        raise ValueError("privileged simulator data leaked into policy features")
    diagnostics = manifest.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("raw episode lacks diagnostics")
    if diagnostics.get("success_source") != "simulator_validation":
        raise ValueError("sim demo must have simulator-validation success provenance")
    randomization = diagnostics.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("raw episode lacks randomization sidecar")
    missing = sorted(REQUIRED_RANDOMIZATION_KEYS - set(randomization))
    if missing:
        raise ValueError(f"raw episode randomization lacks: {missing}")
    if randomization.get("policy_uses_privileged_values") is not False:
        raise ValueError("randomization sidecar does not prove policy isolation")
    profile = manifest.get("dr_profile")
    level = _finite(randomization.get("profile_level"), "profile_level")
    table = randomization["table"]
    if not isinstance(table, dict):
        raise ValueError("table randomization must be an object")
    yaw = _finite(table.get("yaw_delta_rad"), "table.yaw_delta_rad")
    contacts = randomization["contact_materials"]
    if not isinstance(contacts, dict) or not isinstance(contacts.get("pairs"), dict):
        raise ValueError("contact randomization is incomplete")
    camera = randomization["camera_image"]
    if not isinstance(camera, dict) or set(camera.get("rigs", {})) != {
        "head_stereo",
        "left_wrist",
        "right_wrist",
    }:
        raise ValueError("camera image randomization lacks an episode-fixed three-rig profile")
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    trace_timing = _audit_trace(
        root,
        trace_path,
        expected_count=int(manifest["frame_count"]),
    )
    return {
        "episode_id": manifest["episode_id"],
        "dr_profile": profile,
        "profile_level": level,
        "frame_count": int(manifest["frame_count"]),
        "table_yaw_delta_rad": yaw,
        "manifest_sha256": sha256_file(manifest_path),
        "trace_sha256": sha256_file(trace_path),
        "first_policy_image_sha256": _first_policy_images(root),
        **trace_timing,
    }


def audit_collection(raw_root: str | Path, *, config_path: str | Path | None = None) -> dict[str, Any]:
    """Return a complete collection report without writing or exiting.

    Keeping the inspection separately callable makes the release condition
    testable and prevents a partial collection from being mistaken for a
    passing one merely because the CLI produced a report file.
    """

    config = load_teleop_config() if config_path is None else load_teleop_config(config_path)
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    expected = {
        profile.name: profile.successful_demos for profile in config.collection.profiles
    }
    reports: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for episode_root in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        try:
            reports.append(
                _audit_episode(
                    episode_root,
                    config_digest=config.digest,
                    runtime_digest=config.runtime.robofinals_digest,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors[episode_root.name] = f"{type(exc).__name__}: {exc}"
    profile_counts = Counter(report["dr_profile"] for report in reports)
    yaw_values = {round(report["table_yaw_delta_rad"], 6) for report in reports}
    status = "passed"
    if errors or dict(profile_counts) != expected or len(yaw_values) < len(reports):
        status = "failed"
    return {
        "schema_version": "team_ramen_flip_table_teleop_collection_audit/v1",
        "status": status,
        "raw_root": str(root),
        "teleop_config_sha256": config.digest,
        "runtime_digest": config.runtime.robofinals_digest,
        "expected_successes_by_profile": expected,
        "actual_successes_by_profile": dict(sorted(profile_counts.items())),
        "unique_table_yaw_values": len(yaw_values),
        "episodes": reports,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    report = audit_collection(args.raw_root, config_path=args.config)
    atomic_write_json(args.output, report)
    if report["status"] != "passed":
        raise SystemExit(
            "teleop collection audit failed; see "
            f"{args.output}: errors={len(report['errors'])}, "
            f"profile_counts={report['actual_successes_by_profile']}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
