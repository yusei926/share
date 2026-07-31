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
ARM_CHANNEL_NAMES = CHANNEL_NAMES[:14]
DEX1_CHANNEL_NAMES = CHANNEL_NAMES[14:]


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 16 or len(result) < 3 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [T,16] with T >= 3, got {result.shape}")
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
    # A diagnostic interval can be shorter than the broad alignment-search
    # horizon. Delays at or beyond its length are indistinguishable from a
    # constant first sample, so exclude them instead of producing a shifted
    # vector with the wrong length.
    for delay in range(min(max_delay, len(command) - 1) + 1):
        if delay == 0:
            shifted = command
        else:
            shifted = np.empty_like(command)
            shifted[:delay] = command[0]
            shifted[delay:] = command[:-delay]
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
            "arms": _group_summary(channels, ARM_CHANNEL_NAMES),
            "dex1": _group_summary(channels, DEX1_CHANNEL_NAMES),
        },
    }


def _sim_actual_16d(rows: list[dict[str, Any]]) -> np.ndarray:
    states = np.asarray([row["state_after"] for row in rows], dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 33 or not np.isfinite(states).all():
        raise ValueError("trace state_after must be finite [T,33]")
    arms = states[:, UPPER_BODY_SIM_INDICES[3:]]
    fingers = np.column_stack(
        (states[:, LEFT_DEX1_SIM_INDICES].mean(axis=1), states[:, RIGHT_DEX1_SIM_INDICES].mean(axis=1))
    )
    hands = (fingers - DEX1_CLOSE_POS) / (DEX1_OPEN_POS - DEX1_CLOSE_POS) * 4.5
    return np.column_stack((arms, hands))


def _interpolate_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate a recorded 16-D encoder stream at bounded source frames."""

    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError("encoder stream must be [T,16]")
    if positions.ndim != 1 or not np.isfinite(positions).all():
        raise ValueError("source positions must be finite [N]")
    if np.any(positions < 0.0) or np.any(positions > len(values) - 1):
        raise ValueError("source positions are outside the encoder stream")
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    fraction = (positions - lower)[:, None]
    return (1.0 - fraction) * values[lower] + fraction * values[upper]


def _observation_match(sim_actual: np.ndarray, source_observed: np.ndarray) -> dict[str, float | int]:
    """Measure actual simulator joints against same-time recorded encoders."""

    if sim_actual.shape != source_observed.shape or sim_actual.shape[1] != 16:
        raise ValueError("simulator and source observation matrices must both be [T,16]")
    arm_error = sim_actual[:, :14] - source_observed[:, :14]
    hand_error = sim_actual[:, 14:] - source_observed[:, 14:]
    return {
        "samples": int(len(sim_actual)),
        "upper_body_rmse_rad": float(np.sqrt(np.mean(arm_error * arm_error))),
        "upper_body_p95_abs_error_rad": float(np.quantile(np.abs(arm_error), 0.95)),
        # Dex1 remains separately reported because table contact can prevent a
        # physically correct finger from reaching the commanded opening.
        "dex1_rmse_command_units": float(np.sqrt(np.mean(hand_error * hand_error))),
        "dex1_p95_abs_error_command_units": float(np.quantile(np.abs(hand_error), 0.95)),
    }


def identify(
    bundle_path: Path,
    trace_path: Path,
    *,
    max_delay_s: float = 0.20,
    raw_alignment_max_delay_s: float = 2.0,
    source_frame_start: int = 0,
    source_frame_end: int | None = None,
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    full_real_command = _finite_matrix(bundle["recorded_arm_hand_target_16d"], "real command")
    observed_state = np.asarray(bundle["observed_upper_body_state_and_hand_state"], dtype=np.float64)
    if observed_state.ndim != 2 or observed_state.shape[1] != 19:
        raise ValueError("real observed state must be [T,19]")
    full_real_observed = _finite_matrix(
        np.concatenate((observed_state[:, 3:17], observed_state[:, 17:19]), axis=1),
        "real observed arm/hand state",
    )
    source_hz = float(bundle["fps"])
    frame_end = len(full_real_command) if source_frame_end is None else int(source_frame_end)
    frame_start = int(source_frame_start)
    if not 0 <= frame_start < frame_end <= len(full_real_command):
        raise ValueError(
            "source frame interval must satisfy "
            f"0 <= start < end <= {len(full_real_command)}, got [{frame_start}, {frame_end})"
        )
    real_command = full_real_command[frame_start:frame_end]
    real_observed = full_real_observed[frame_start:frame_end]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    def observed_frame(row: dict[str, Any]) -> float:
        value = row.get("source_observed_frame", row.get("replay_index", -1))
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("trace has an invalid source_observed_frame/replay_index")
        return float(value)

    replay_rows = [
        row
        for row in rows
        if not bool(row.get("replay_warmup", False))
        and frame_start <= observed_frame(row) < frame_end
    ]
    if len(replay_rows) < 3:
        raise ValueError("trace has too few post-warmup rows")
    sim_command = _finite_matrix([row["source_action_16d"] for row in replay_rows], "sim command")
    sim_actual = _sim_actual_16d(replay_rows)
    source_positions = np.asarray([observed_frame(row) for row in replay_rows], dtype=np.float64)
    source_encoder_at_sim_time = _interpolate_rows(full_real_observed, source_positions)
    # The policy holds a 30 Hz command at a 50 Hz simulator rate.  Its trace
    # represents physical controller steps, so retain all rows for fitting.
    return {
        "schema_version": "team_ramen_flip_table_actuator_identification/v1",
        "policy_use": "forbidden: offline simulator-actuator calibration only",
        "source_episode_index": int(bundle["source_episode_index"]),
        "source_frame_interval": {
            "start_inclusive": frame_start,
            "end_exclusive": frame_end,
            "start_s": frame_start / source_hz,
            "end_s": frame_end / source_hz,
            "selection_rule": (
                "explicit caller-selected interval; use a demonstrably free-space interval "
                "for actuator identification and keep contact-dominated rows out of the fit"
            ),
        },
        "command_delay_steps": sorted(
            {int(row.get("command_delay_steps", 0)) for row in replay_rows}
        ),
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
        "sim_to_real_encoder_match": _observation_match(sim_actual, source_encoder_at_sim_time),
        "interpretation": (
            "Compare real/sim alpha and delay per channel before changing joint stiffness, damping, "
            "armature, friction, or command latency. Rank an arm-profile candidate only with the arms "
            "group; Dex1 is diagnostic because its drive differs and hand motion can "
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
    parser.add_argument(
        "--source-frame-start",
        type=int,
        default=0,
        help="inclusive recorded-source frame for a contact-free identification interval",
    )
    parser.add_argument(
        "--source-frame-end",
        type=int,
        help="exclusive recorded-source frame; default is the complete episode",
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
        source_frame_start=args.source_frame_start,
        source_frame_end=args.source_frame_end,
    )
    atomic_write_json(args.output.expanduser().resolve(), report)
    print(json.dumps({"real": report["real"]["summary"], "sim": report["sim"]["summary"]}, indent=2))


if __name__ == "__main__":
    main()
