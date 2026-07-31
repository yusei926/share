"""Summarize and select a GR00T temporal-ensemble simulator candidate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path)
    parser.add_argument("--scripted-dir", type=Path)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        help="Validate and summarize one fixed-scene or DR policy run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _vector(value: Any, expected: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (expected,) or not np.isfinite(array).all():
        return None
    return array


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("test_*/action_state_trace.jsonl")):
        with path.open(encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())
    return rows


def policy_episode_seeds(
    directory: Path,
    base_seed: int,
    expected_count: int,
) -> list[int]:
    episode_seeds: list[int] = []
    episode_directories = sorted(
        directory.glob("test_*"),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if len(episode_directories) != expected_count:
        raise ValueError(
            f"expected {expected_count} episode trace directories under {directory}, "
            f"found {len(episode_directories)}"
        )
    for episode_index, episode_dir in enumerate(episode_directories):
        recorded_index = int(episode_dir.name.rsplit("_", 1)[-1])
        if recorded_index != episode_index:
            raise ValueError(
                f"expected contiguous test_{episode_index}, found {episode_dir.name}"
            )
        trace = episode_dir / "action_state_trace.jsonl"
        if not trace.is_file():
            raise ValueError(f"GR00T action trace is missing: {trace}")
        rows = [
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows or any("policy_inference_seed" not in row for row in rows):
            raise ValueError(
                f"{episode_dir.name} lacks complete policy inference-seed evidence"
            )
        seeds = {int(row["policy_inference_seed"]) for row in rows}
        expected = base_seed + episode_index
        if seeds != {expected}:
            raise ValueError(
                f"{episode_dir.name} used inference seeds {sorted(seeds)}, "
                f"expected only {expected}"
            )
        episode_seeds.append(expected)
    return episode_seeds


def policy_runtime_labels(directory: Path) -> dict[str, str]:
    rows = _trace_rows(directory)
    if not rows:
        raise ValueError(f"GR00T action trace is missing under {directory}")
    modes = {str(row.get("evaluation_mode", "")) for row in rows}
    profiles = {
        str(row.get("domain_randomization_profile", "")) for row in rows
    }
    if len(modes) != 1 or "" in modes:
        raise ValueError(f"runtime evaluation mode is missing or inconsistent in {directory}")
    if len(profiles) != 1 or "" in profiles:
        raise ValueError(
            f"runtime domain-randomization profile is missing or inconsistent in {directory}"
        )
    return {
        "evaluation_mode": modes.pop(),
        "domain_randomization_profile": profiles.pop(),
    }


def scripted_tracking_metrics(directory: Path) -> dict[str, float | bool]:
    errors: list[np.ndarray] = []
    actual: list[np.ndarray] = []
    for row in _trace_rows(directory):
        error = _vector(row.get("tracking_error"), 14)
        measured = _vector(row.get("actual_joint_position"), 14)
        if error is not None and measured is not None:
            errors.append(error)
            actual.append(measured)
    if not errors:
        raise ValueError(f"scripted tracking trace is missing under {directory}")
    error_array = np.stack(errors)
    actual_array = np.stack(actual)
    rmse = float(np.sqrt(np.mean(np.square(error_array))))
    p95 = float(np.quantile(np.abs(error_array), 0.95))
    motion_range = float(np.max(np.ptp(actual_array, axis=0)))
    passed = rmse <= 0.08 and p95 <= 0.16 and motion_range >= 0.05
    return {
        "arm_rmse_rad": rmse,
        "arm_p95_abs_error_rad": p95,
        "actual_arm_range_rad": motion_range,
        "passed": passed,
    }


def policy_trace_metrics(directory: Path) -> dict[str, float | int | None]:
    targets: list[np.ndarray] = []
    tracking_errors: list[np.ndarray] = []
    for row in _trace_rows(directory):
        if not bool(row.get("action_advanced")):
            continue
        target = _vector(row.get("safe_joint_target_16d"), 16)
        state = _vector(row.get("joint_state_after_19d"), 19)
        if target is None:
            continue
        targets.append(target)
        if state is not None:
            measured = np.concatenate((state[3:17], state[17:19]))
            tracking_errors.append(target - measured)
    if not targets:
        raise ValueError(f"GR00T action trace is missing under {directory}")
    target_array = np.stack(targets)
    velocity = np.diff(target_array, axis=0) * 30.0
    acceleration = np.diff(velocity, axis=0) * 30.0
    jerk = np.diff(acceleration, axis=0) * 30.0
    return {
        "action_advances": len(targets),
        "target_step_rmse": (
            float(np.sqrt(np.mean(np.square(np.diff(target_array, axis=0)))))
            if len(target_array) > 1
            else 0.0
        ),
        "target_acceleration_rms": (
            float(np.sqrt(np.mean(np.square(acceleration))))
            if len(acceleration)
            else 0.0
        ),
        "target_jerk_rms": (
            float(np.sqrt(np.mean(np.square(jerk)))) if len(jerk) else 0.0
        ),
        "tracking_rmse": (
            float(np.sqrt(np.mean(np.square(np.stack(tracking_errors)))))
            if tracking_errors
            else None
        ),
    }


def _candidate_preference(decay_lambda: str, execution_steps: int) -> int:
    order = [
        ("-0.1", 10),
        ("-0.1", 5),
        ("-0.1", 20),
        ("-0.25", 10),
        ("-0.25", 5),
        ("-0.25", 20),
        ("0", 10),
        ("0", 5),
        ("0", 20),
        ("none", 10),
        ("none", 5),
        ("none", 20),
    ]
    try:
        return order.index((decay_lambda, execution_steps))
    except ValueError:
        return len(order)


def summarize_candidate(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "candidate_manifest.json")
    result = _read_json(directory / "eval_results.json")
    seed = int(manifest["seed"])
    policy_seed = int(manifest["policy_inference_seed"])
    if policy_seed != seed:
        raise ValueError(
            f"environment seed {seed} and policy inference seed {policy_seed} differ"
        )
    test_count = int(result["test_count"])
    success_count = int(result["success_count"])
    if test_count <= 0 or not 0 <= success_count <= test_count:
        raise ValueError(f"invalid evaluation result in {directory}")
    if int(manifest.get("episodes", -1)) != test_count:
        raise ValueError(f"manifest episode count changed in {directory}")
    mode = str(manifest.get("mode", ""))
    if not mode:
        raise ValueError(f"manifest evaluation mode is missing in {directory}")
    profile = str(manifest.get("domain_randomization_profile", ""))
    if not profile:
        raise ValueError(
            f"manifest domain-randomization profile is missing in {directory}"
        )
    runtime = policy_runtime_labels(directory)
    expected_runtime_mode = {
        "nominal": "nominal",
        "randomized_validation": "randomized",
        "unseen_dr": "unseen_dr",
    }.get(mode)
    if expected_runtime_mode is None or runtime["evaluation_mode"] != expected_runtime_mode:
        raise ValueError(f"manifest and runtime evaluation modes differ in {directory}")
    if runtime["domain_randomization_profile"] != profile:
        raise ValueError(
            f"manifest and runtime domain-randomization profiles differ in {directory}"
        )
    expected_episode_ids = [f"{seed}:{index}" for index in range(test_count)]
    if manifest.get("episode_ids") != expected_episode_ids:
        raise ValueError(f"manifest episode IDs changed in {directory}")
    return {
        "name": directory.name,
        "directory": str(directory.resolve()),
        "temporal_lambda": str(manifest["temporal_lambda"]),
        "execution_steps": int(manifest["execution_steps"]),
        "mode": mode,
        "runtime_evaluation_mode": runtime["evaluation_mode"],
        "domain_randomization_profile": profile,
        "seed": seed,
        "policy_inference_seed": policy_seed,
        "episode_inference_seeds": policy_episode_seeds(
            directory,
            policy_seed,
            test_count,
        ),
        "episode_ids": expected_episode_ids,
        "test_count": test_count,
        "success_count": success_count,
        "success_rate": float(success_count / test_count),
        "trace": policy_trace_metrics(directory),
    }


def _selection_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    trace = candidate["trace"]
    tracking = trace["tracking_rmse"]
    return (
        -float(candidate["success_rate"]),
        float(trace["target_jerk_rms"]),
        float(trace["target_acceleration_rms"]),
        math.inf if tracking is None else float(tracking),
        float(
            _candidate_preference(
                str(candidate["temporal_lambda"]),
                int(candidate["execution_steps"]),
            )
        ),
    )


def main() -> None:
    args = parse_args()
    if args.candidate_dir is not None:
        if args.sweep_root is not None or args.scripted_dir is not None:
            raise ValueError(
                "--candidate-dir cannot be combined with --sweep-root or --scripted-dir"
            )
        summary = summarize_candidate(args.candidate_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, allow_nan=False))
        return
    if args.sweep_root is None or args.scripted_dir is None:
        raise ValueError(
            "--sweep-root and --scripted-dir are required for temporal selection"
        )
    scripted = scripted_tracking_metrics(args.scripted_dir)
    if not scripted["passed"]:
        raise RuntimeError(
            "scripted controller tracking gate failed; learned-policy results are invalid"
        )
    candidates = [
        summarize_candidate(path)
        for path in sorted(args.sweep_root.iterdir())
        if path.is_dir() and (path / "candidate_manifest.json").is_file()
    ]
    if len(candidates) != 12:
        raise ValueError(f"expected 12 temporal candidates, found {len(candidates)}")
    selected = min(candidates, key=_selection_key)
    report = {
        "schema_version": "team_ramen_groot_n17_temporal_sweep/v1",
        "selection_basis": [
            "success_rate_descending",
            "target_jerk_rms_ascending",
            "target_acceleration_rms_ascending",
            "tracking_rmse_ascending",
            "documented_primary_setting_tie_break",
        ],
        "scripted_controller_tracking": scripted,
        "candidates": candidates,
        "selected": {
            "name": selected["name"],
            "temporal_lambda": selected["temporal_lambda"],
            "execution_steps": selected["execution_steps"],
            "success_rate": selected["success_rate"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["selected"], indent=2))


if __name__ == "__main__":
    main()
