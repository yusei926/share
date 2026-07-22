#!/usr/bin/env python3
"""Fail-closed acceptance report for fixed-parameter real-to-sim validation.

Each supplied episode report must identify the held-out source episode, the
frozen shared-parameter digest, and measurements derived from recorded real
evidence and its simulator replay. Missing measurements fail the release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_heldout_validation/v1"
REQUIRED_METRICS = {
    "camera_reprojection_median_px": (3.0, "max"),
    "camera_reprojection_p95_px": (8.0, "max"),
    "upper_body_joint_rmse_rad": (0.03, "max"),
    "table_translation_rmse_m": (0.020, "max"),
    "table_rotation_rmse_deg": (3.0, "max"),
    "phase_timing_max_error_s": (0.100, "max"),
    "mask_iou": (0.90, "min"),
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _selected_validation_indices(calibration_manifest: dict[str, Any]) -> tuple[int, ...]:
    bundles = calibration_manifest.get("episode_bundles")
    if not isinstance(bundles, dict):
        raise ValueError("calibration manifest omits episode_bundles")
    result = []
    for name, path_value in sorted(bundles.items()):
        if name.startswith("validation_"):
            result.append(int(_load_object(Path(str(path_value)))["source_episode_index"]))
    if len(result) != 5 or len(set(result)) != 5:
        raise ValueError("calibration manifest must reserve exactly five unique validation episodes")
    return tuple(sorted(result))


def _metric_gate(metrics: dict[str, Any], metric_sources: dict[str, Any]) -> dict[str, Any]:
    outcomes: dict[str, dict[str, Any]] = {}
    for name, (threshold, direction) in REQUIRED_METRICS.items():
        value = metrics.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            outcomes[name] = {"status": "missing", "threshold": threshold, "direction": direction}
            continue
        source = metric_sources.get(name)
        if not isinstance(source, str) or not source.strip():
            outcomes[name] = {
                "status": "missing_source",
                "value": float(value),
                "threshold": threshold,
                "direction": direction,
            }
            continue
        observed = float(value)
        passed = observed <= threshold if direction == "max" else observed >= threshold
        outcomes[name] = {
            "status": "pass" if passed else "fail",
            "value": observed,
            "threshold": threshold,
            "direction": direction,
        }
    return {"passed": all(value["status"] == "pass" for value in outcomes.values()), "metrics": outcomes}


def evaluate(calibration_manifest_path: Path, episode_report_paths: tuple[Path, ...]) -> dict[str, Any]:
    calibration_manifest = _load_object(calibration_manifest_path)
    expected_indices = _selected_validation_indices(calibration_manifest)
    reports: dict[int, dict[str, Any]] = {}
    duplicate_indices: list[int] = []
    invalid_reports: list[dict[str, Any]] = []
    parameter_digests: set[str] = set()

    for path in episode_report_paths:
        report = _load_object(path)
        if report.get("schema_version") != SCHEMA_VERSION:
            invalid_reports.append({"path": str(path), "reason": "schema_version"})
            continue
        try:
            index = int(report["source_episode_index"])
        except (KeyError, TypeError, ValueError):
            invalid_reports.append({"path": str(path), "reason": "source_episode_index"})
            continue
        if index in reports:
            duplicate_indices.append(index)
            continue
        digest = report.get("shared_parameter_sha256")
        parameter_path_value = report.get("shared_parameters_path")
        if not isinstance(digest, str) or len(digest) != 64:
            invalid_reports.append({"path": str(path), "reason": "shared_parameter_sha256"})
            continue
        if not isinstance(parameter_path_value, str) or not parameter_path_value:
            invalid_reports.append({"path": str(path), "reason": "shared_parameters_path"})
            continue
        parameter_path = Path(parameter_path_value)
        if not parameter_path.is_absolute():
            parameter_path = path.parent / parameter_path
        if not parameter_path.is_file() or hashlib.sha256(parameter_path.read_bytes()).hexdigest() != digest:
            invalid_reports.append({"path": str(path), "reason": "shared_parameter_snapshot"})
            continue
        parameter_digests.add(digest)
        reports[index] = {**report, "report_path": str(path)}

    episodes = []
    for index in expected_indices:
        report = reports.get(index)
        if report is None:
            episodes.append(
                {
                    "source_episode_index": index,
                    "status": "missing_report",
                    "gate": _metric_gate({}, {}),
                }
            )
            continue
        metrics = report.get("metrics")
        metric_sources = report.get("metric_sources")
        episodes.append(
            {
                "source_episode_index": index,
                "status": "evaluated",
                "report_path": report["report_path"],
                "gate": _metric_gate(
                    metrics if isinstance(metrics, dict) else {},
                    metric_sources if isinstance(metric_sources, dict) else {},
                ),
                "metric_sources": metric_sources if isinstance(metric_sources, dict) else {},
            }
        )

    unexpected = sorted(set(reports) - set(expected_indices))
    shared_parameters_frozen = len(parameter_digests) == 1 and len(reports) == len(expected_indices)
    passed = (
        not invalid_reports
        and not duplicate_indices
        and not unexpected
        and shared_parameters_frozen
        and all(episode["status"] == "evaluated" and episode["gate"]["passed"] for episode in episodes)
    )
    return {
        "schema_version": "team_ramen_flip_table_heldout_acceptance/v1",
        "calibration_manifest": str(calibration_manifest_path),
        "calibration_manifest_sha256": hashlib.sha256(calibration_manifest_path.read_bytes()).hexdigest(),
        "required_validation_episode_indices": list(expected_indices),
        "shared_parameter_sha256": next(iter(parameter_digests), None) if len(parameter_digests) == 1 else None,
        "shared_parameters_frozen": shared_parameters_frozen,
        "unexpected_episode_indices": unexpected,
        "duplicate_episode_indices": sorted(duplicate_indices),
        "invalid_reports": invalid_reports,
        "episodes": episodes,
        "passed": passed,
        "decision": "accepted" if passed else "rejected_or_incomplete",
        "policy_use": "forbidden: offline calibration acceptance only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--episode-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.calibration_manifest.expanduser().resolve(),
        tuple(path.expanduser().resolve() for path in args.episode_report),
    )
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"decision": result["decision"], "passed": result["passed"]}, indent=2))


if __name__ == "__main__":
    main()
