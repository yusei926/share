#!/usr/bin/env python3
"""Create one fail-closed held-out replay report from immutable artifacts.

This utility measures only what its inputs establish.  In particular, a
recorded joint replay trace yields the upper-body tracking metric, but it does
not reveal real table pose, contact, phase timing, or camera reprojection.
Those values remain absent until supplied by their dedicated comparison tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from data.flip_table_data_augmentation.io_utils import atomic_write_json

from .heldout_validation import REQUIRED_METRICS, SCHEMA_VERSION
from .replay import analyze_trace


DERIVED_COMPARISON_SCHEMAS = {
    "head": "team_ramen_multiframe_head_geometry/v1",
    "table_motion": "team_ramen_flip_table_table_motion_comparison/v1",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _metric_overrides(path: Path | None) -> tuple[dict[str, float], dict[str, str]]:
    """Load explicit comparison measurements without guessing any field names."""

    if path is None:
        return {}, {}
    payload = _load_object(path)
    metrics = payload.get("metrics")
    sources = payload.get("metric_sources")
    if not isinstance(metrics, dict) or not isinstance(sources, dict):
        raise ValueError(
            f"comparison metrics must have object metrics and metric_sources fields: {path}"
        )
    allowed = set(REQUIRED_METRICS) - {"upper_body_joint_rmse_rad"}
    selected = {
        key: float(value)
        for key, value in metrics.items()
        if key in allowed and isinstance(value, (int, float))
    }
    selected_sources = {
        key: str(value)
        for key, value in sources.items()
        if key in selected and isinstance(value, str) and value.strip()
    }
    return selected, selected_sources


def _derived_metrics(
    *,
    path: Path | None,
    expected_schema: str,
    source_episode_index: int,
    allowed: set[str],
) -> tuple[dict[str, float], dict[str, str]]:
    """Load metrics only from a typed, episode-bound comparison artifact."""

    if path is None:
        return {}, {}
    payload = _load_object(path)
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"comparison schema differs from {expected_schema}: {path}")
    if payload.get("source_episode_index") != source_episode_index:
        raise ValueError(f"comparison episode differs from held-out bundle: {path}")
    metrics = payload.get("metrics")
    sources = payload.get("metric_sources")
    if not isinstance(metrics, dict) or not isinstance(sources, dict):
        raise ValueError(f"comparison omits metrics/metric_sources: {path}")
    selected = {
        key: float(value)
        for key, value in metrics.items()
        if key in allowed and isinstance(value, (int, float))
    }
    selected_sources = {
        key: str(value)
        for key, value in sources.items()
        if key in selected and isinstance(value, str) and value.strip()
    }
    return selected, selected_sources


def build_report(
    *,
    episode_bundle_path: Path,
    trace_path: Path,
    shared_parameters_path: Path,
    comparison_metrics_path: Path | None,
    head_comparison_path: Path | None,
    table_motion_comparison_path: Path | None,
    trace_analysis_path: Path,
) -> dict[str, Any]:
    bundle = _load_object(episode_bundle_path)
    index = int(bundle["source_episode_index"])
    analysis = analyze_trace(trace_path, trace_analysis_path)
    metrics: dict[str, float] = {
        "upper_body_joint_rmse_rad": float(
            analysis["replay_observation_matching"]["upper_body_rmse_rad"]
        )
    }
    metric_sources: dict[str, str] = {
        "upper_body_joint_rmse_rad": (
            "recorded robot_q_current versus simulator state_after from fixed-base replay"
        )
    }
    comparison_metrics, comparison_sources = _metric_overrides(comparison_metrics_path)
    metrics.update(comparison_metrics)
    metric_sources.update(comparison_sources)
    head_metrics, head_sources = _derived_metrics(
        path=head_comparison_path,
        expected_schema=DERIVED_COMPARISON_SCHEMAS["head"],
        source_episode_index=index,
        allowed={"camera_reprojection_median_px", "camera_reprojection_p95_px", "mask_iou"},
    )
    table_metrics, table_sources = _derived_metrics(
        path=table_motion_comparison_path,
        expected_schema=DERIVED_COMPARISON_SCHEMAS["table_motion"],
        source_episode_index=index,
        allowed={"table_translation_rmse_m", "table_rotation_rmse_deg", "phase_timing_max_error_s"},
    )
    metrics.update(head_metrics)
    metrics.update(table_metrics)
    metric_sources.update(head_sources)
    metric_sources.update(table_sources)
    snapshot = shared_parameters_path.read_bytes()
    return {
        "schema_version": SCHEMA_VERSION,
        "source_episode_index": index,
        "episode_bundle_path": str(episode_bundle_path),
        "trace_path": str(trace_path),
        "shared_parameters_path": str(shared_parameters_path),
        "shared_parameter_sha256": hashlib.sha256(snapshot).hexdigest(),
        "metrics": metrics,
        "metric_sources": metric_sources,
        "trace_analysis": analysis,
        "comparison_metrics_path": str(comparison_metrics_path) if comparison_metrics_path else None,
        "head_comparison_path": str(head_comparison_path) if head_comparison_path else None,
        "table_motion_comparison_path": (
            str(table_motion_comparison_path) if table_motion_comparison_path else None
        ),
        "policy_use": "forbidden: offline calibration acceptance only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--shared-parameters", type=Path, required=True)
    parser.add_argument("--comparison-metrics", type=Path)
    parser.add_argument("--head-comparison", type=Path)
    parser.add_argument("--table-motion-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        episode_bundle_path=args.episode_bundle.expanduser().resolve(),
        trace_path=args.trace.expanduser().resolve(),
        shared_parameters_path=args.shared_parameters.expanduser().resolve(),
        comparison_metrics_path=(
            args.comparison_metrics.expanduser().resolve() if args.comparison_metrics else None
        ),
        head_comparison_path=(
            args.head_comparison.expanduser().resolve() if args.head_comparison else None
        ),
        table_motion_comparison_path=(
            args.table_motion_comparison.expanduser().resolve()
            if args.table_motion_comparison
            else None
        ),
        trace_analysis_path=(args.output.expanduser().resolve().with_suffix(".trace_analysis.json")),
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"source_episode_index": report["source_episode_index"], "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
