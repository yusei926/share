#!/usr/bin/env python3
"""Identify real and simulated upper-body response from recorded replays.

The fitter is an offline diagnostic.  It never changes a replay action and it
never exposes state, delay, or fitted parameters to a policy.  Its only output
is a reproducible recommendation for simulator actuator parameters.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json
from .replay import DEX1_CLOSE_POS, DEX1_OPEN_POS, LEFT_DEX1_SIM_INDICES, RIGHT_DEX1_SIM_INDICES, UPPER_BODY_SIM_INDICES


CHANNEL_NAMES = (
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    "left_hand_command",
    "right_hand_command",
)

# The calibration overrides in ``patch_g1_global_camera.py`` apply only to
# shoulder, elbow, and wrist actuators.  Keep waist and Dex1 diagnostics in
# the report, but never use them to rank an arm-profile candidate: the waist
# retains its organizer WBC drive and hand measurements are often contact
# limited during an assembly demonstration.
WAIST_CHANNEL_NAMES = CHANNEL_NAMES[:3]
ARM_CHANNEL_NAMES = CHANNEL_NAMES[3:17]
DEX1_CHANNEL_NAMES = CHANNEL_NAMES[17:]


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 19 or len(result) < 3 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [T,19] with T >= 3, got {result.shape}")
    return result


def _first_order_response(command: np.ndarray, alpha: float, delay: int, initial: float) -> np.ndarray:
    response = np.empty_like(command)
    response[0] = initial
    for index in range(1, len(command)):
        command_index = max(0, index - delay)
        response[index] = response[index - 1] + alpha * (command[command_index] - response[index - 1])
    return response


def _fit_channel(command: np.ndarray, observed: np.ndarray, *, max_delay: int) -> dict[str, float | int]:
    """Fit a stable, discrete first-order response by exhaustive small-grid search."""

    best: tuple[float, float, int, np.ndarray] | None = None
    for delay in range(max_delay + 1):
        for alpha in np.arange(0.02, 1.0001, 0.01):
            predicted = _first_order_response(command, float(alpha), delay, float(observed[0]))
            mse = float(np.mean((predicted - observed) ** 2))
            candidate = (mse, float(alpha), delay, predicted)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    assert best is not None
    mse, alpha, delay, predicted = best
    residual = predicted - observed
    baseline = command - observed
    return {
        "alpha_per_sample": alpha,
        "delay_samples": delay,
        "rmse": float(np.sqrt(mse)),
        "p95_abs_error": float(np.quantile(np.abs(residual), 0.95)),
        "raw_command_rmse": float(np.sqrt(np.mean(baseline * baseline))),
        "raw_command_p95_abs_error": float(np.quantile(np.abs(baseline), 0.95)),
    }


def _raw_alignment_delay(command: np.ndarray, observed: np.ndarray, *, max_delay: int) -> dict[str, float | int]:
    """Find a diagnostic command/encoder offset without changing the replay."""

    best: tuple[float, int] | None = None
    for delay in range(max_delay + 1):
        shifted = (
            command
            if delay == 0
            else np.concatenate((np.full(delay, command[0], dtype=np.float64), command[:-delay]))
        )
        mse = float(np.mean((shifted - observed) ** 2))
        candidate = (mse, delay)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {"raw_alignment_rmse": float(math.sqrt(best[0])), "raw_alignment_delay_samples": best[1]}


def _group_summary(
    channels: dict[str, dict[str, float | int]], names: tuple[str, ...]
) -> dict[str, float | int]:
    """Summarize one actuator family without mixing incomparable drives."""

    selected = [channels[name] for name in names]
    if not selected:
        raise ValueError("actuator group must contain at least one channel")
    alpha = np.asarray([entry["alpha_per_sample"] for entry in selected], dtype=np.float64)
    delay = np.asarray([entry["delay_samples"] for entry in selected], dtype=np.float64)
    time_constants = np.asarray([entry["time_constant_s"] for entry in selected], dtype=np.float64)
    raw_alignment_delay = np.asarray(
        [entry["raw_alignment_delay_samples"] for entry in selected], dtype=np.float64
    )
    return {
        "channels": int(len(selected)),
        "median_alpha_per_sample": float(np.median(alpha)),
        "median_delay_samples": float(np.median(delay)),
        "median_time_constant_s": float(np.median(time_constants)),
        "median_channel_rmse": float(np.median([entry["rmse"] for entry in selected])),
        "median_raw_alignment_delay_samples": float(np.median(raw_alignment_delay)),
    }


def fit_response(
    command: np.ndarray,
    observed: np.ndarray,
    *,
    hz: float,
    max_delay_s: float = 0.20,
    raw_alignment_max_delay_s: float = 2.0,
) -> dict[str, Any]:
    command = _finite_matrix(command, "command")
    observed = _finite_matrix(observed, "observed")
    if command.shape != observed.shape or hz <= 0.0:
        raise ValueError("command/observed shapes and sampling rate are invalid")
    if raw_alignment_max_delay_s < max_delay_s:
        raise ValueError("raw alignment delay search must be at least first-order fit delay search")
    max_delay = int(round(max_delay_s * hz))
    raw_alignment_max_delay = int(round(raw_alignment_max_delay_s * hz))
    channels = {
        name: _fit_channel(command[:, index], observed[:, index], max_delay=max_delay)
        for index, name in enumerate(CHANNEL_NAMES)
    }
    for entry in channels.values():
        alpha = float(entry["alpha_per_sample"])
        entry["time_constant_s"] = (
            0.0 if alpha >= 1.0 - 1.0e-12 else float(-1.0 / (hz * math.log1p(-alpha)))
        )
        entry["delay_s"] = float(entry["delay_samples"]) / hz
    for index, entry in enumerate(channels.values()):
        entry.update(_raw_alignment_delay(command[:, index], observed[:, index], max_delay=raw_alignment_max_delay))
        entry["raw_alignment_delay_s"] = float(entry["raw_alignment_delay_samples"]) / hz
    alpha = np.asarray([entry["alpha_per_sample"] for entry in channels.values()], dtype=np.float64)
    delay = np.asarray([entry["delay_samples"] for entry in channels.values()], dtype=np.float64)
    time_constants = np.asarray([entry["time_constant_s"] for entry in channels.values()], dtype=np.float64)
    raw_alignment_delay = np.asarray(
        [entry["raw_alignment_delay_samples"] for entry in channels.values()], dtype=np.float64
    )
    return {
        "samples": int(len(command)),
        "hz": float(hz),
        "max_delay_s": float(max_delay_s),
        "raw_alignment_max_delay_s": float(raw_alignment_max_delay_s),
        "channels": channels,
        "summary": {
            "median_alpha_per_sample": float(np.median(alpha)),
            "median_delay_s": float(np.median(delay) / hz),
            "median_time_constant_s": float(np.median(time_constants)),
            "median_channel_rmse": float(np.median([entry["rmse"] for entry in channels.values()])),
            "median_raw_alignment_delay_s": float(np.median(raw_alignment_delay) / hz),
        },
        "group_summaries": {
            "waist": _group_summary(channels, WAIST_CHANNEL_NAMES),
            "arms": _group_summary(channels, ARM_CHANNEL_NAMES),
            "dex1": _group_summary(channels, DEX1_CHANNEL_NAMES),
        },
    }


def _sim_actual_19d(rows: list[dict[str, Any]]) -> np.ndarray:
    states = np.asarray([row["state_after"] for row in rows], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 33 or not np.isfinite(states).all():
        raise ValueError("trace state_after must be finite [T,33]")
    body = states[:, UPPER_BODY_SIM_INDICES]
    fingers = np.column_stack(
        (states[:, LEFT_DEX1_SIM_INDICES].mean(axis=1), states[:, RIGHT_DEX1_SIM_INDICES].mean(axis=1))
    )
    hands = (fingers - DEX1_CLOSE_POS) / (DEX1_OPEN_POS - DEX1_CLOSE_POS) * 4.5
    return np.column_stack((body, hands))


def identify(
    bundle_path: Path,
    trace_path: Path,
    *,
    max_delay_s: float = 0.20,
    raw_alignment_max_delay_s: float = 2.0,
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    real_command = _finite_matrix(bundle["recorded_upper_body_target_and_hand_cmd"], "real command")
    real_observed = _finite_matrix(bundle["observed_upper_body_state_and_hand_state"], "real observed")
    source_hz = float(bundle["fps"])
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    replay_rows = [row for row in rows if not bool(row.get("replay_warmup", False))]
    if len(replay_rows) < 3:
        raise ValueError("trace has too few post-warmup rows")
    sim_command = _finite_matrix([row["source_action_19d"] for row in replay_rows], "sim command")
    sim_actual = _sim_actual_19d(replay_rows)
    # The policy holds a 30 Hz command at a 50 Hz simulator rate.  Its trace
    # represents physical controller steps, so retain all rows for fitting.
    return {
        "schema_version": "team_ramen_flip_table_actuator_identification/v1",
        "policy_use": "forbidden: offline simulator-actuator calibration only",
        "source_episode_index": int(bundle["source_episode_index"]),
        "real": fit_response(
            real_command,
            real_observed,
            hz=source_hz,
            max_delay_s=max_delay_s,
            raw_alignment_max_delay_s=raw_alignment_max_delay_s,
        ),
        "sim": fit_response(
            sim_command,
            sim_actual,
            hz=50.0,
            max_delay_s=max_delay_s,
            raw_alignment_max_delay_s=raw_alignment_max_delay_s,
        ),
        "interpretation": (
            "Compare real/sim alpha and delay per channel before changing joint stiffness, damping, "
            "armature, friction, or command latency. Rank an arm-profile candidate only with the arms "
            "group; waist and Dex1 groups are diagnostic because their drives differ and hand motion can "
            "be contact-limited. raw_alignment_delay is a broad offline timestamp diagnostic only; it "
            "never alters replay timing. This result is not a physical parameter estimate by itself."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-delay-s",
        type=float,
        default=0.20,
        help="offline target-to-encoder lag search bound; does not alter replay timing",
    )
    parser.add_argument(
        "--raw-alignment-max-delay-s",
        type=float,
        default=2.0,
        help="broad offline command/encoder offset search; does not alter replay timing",
    )
    args = parser.parse_args()
    if not math.isfinite(args.max_delay_s) or not 0.0 <= args.max_delay_s <= 5.0:
        raise ValueError("--max-delay-s must be finite and in [0,5]")
    if (
        not math.isfinite(args.raw_alignment_max_delay_s)
        or not args.max_delay_s <= args.raw_alignment_max_delay_s <= 5.0
    ):
        raise ValueError("--raw-alignment-max-delay-s must be finite, in [max-delay-s,5]")
    report = identify(
        args.episode_bundle.expanduser().resolve(),
        args.trace.expanduser().resolve(),
        max_delay_s=args.max_delay_s,
        raw_alignment_max_delay_s=args.raw_alignment_max_delay_s,
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"real": report["real"]["summary"], "sim": report["sim"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
