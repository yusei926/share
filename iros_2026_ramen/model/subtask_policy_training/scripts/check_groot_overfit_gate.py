"""Reject a GR00T overfit run that is static, numerically invalid, or grossly inaccurate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def check_report(report: dict[str, Any]) -> dict[str, Any]:
    episode_metrics = list(report["episodes_report"].values())
    if not episode_metrics:
        raise ValueError("overfit report contains no episodes")
    aggregate = report["aggregate"]
    numeric = [
        float(aggregate["physical_arm_rmse_rad"]),
        float(aggregate["dex1_mae"]),
        float(aggregate["predicted_arm_range_rad"]),
        float(aggregate["predicted_dex1_range"]),
    ]
    finite = all(math.isfinite(value) for value in numeric)
    arm_amplitude = all(
        float(item["initial_chunk_arm_max_abs_displacement_rad"])
        >= min(
            0.02,
            0.25 * float(item["initial_chunk_target_arm_max_abs_displacement_rad"]),
        )
        for item in episode_metrics
    )
    dex_target_active = any(
        float(item["initial_chunk_target_dex1_range"]) >= 0.2 for item in episode_metrics
    )
    dex_amplitude = (
        any(float(item["initial_chunk_dex1_range"]) >= 0.1 for item in episode_metrics)
        if dex_target_active
        else True
    )
    noncollapsed = float(aggregate["stationary_frame_fraction"]) < 0.9
    arm_accuracy = float(aggregate["physical_arm_rmse_rad"]) < 0.6
    dex_accuracy = float(aggregate["dex1_mae"]) < 1.5
    checks = {
        "finite": finite,
        "initial_arm_amplitude": arm_amplitude,
        "initial_dex1_amplitude_when_target_active": dex_amplitude,
        "not_stationary_collapse": noncollapsed,
        "arm_rmse_below_0_6_rad": arm_accuracy,
        "dex1_mae_below_1_5": dex_accuracy,
    }
    return {
        "schema_version": "groot_n17_overfit_gate_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = check_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("GR00T overfit gate failed; full training was not started")


if __name__ == "__main__":
    main()
