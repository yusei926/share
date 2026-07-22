#!/usr/bin/env python3
"""Materialize and evaluate fixed-base 19-D real-demonstration replays."""

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

from data.flip_table_data_augmentation.io_utils import atomic_write_json, atomic_write_text

from .contracts import SOURCE_FPS, finite_matrix
from .extract_visual_evidence import frame_indices as evidence_frame_indices


UPPER_BODY_SIM_INDICES = (2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
LEFT_DEX1_SIM_INDICES = (29, 30)
RIGHT_DEX1_SIM_INDICES = (31, 32)
DEX1_OPEN_POS = 0.0245
DEX1_CLOSE_POS = -0.02
INITIALIZATION_MAX_RMSE_RAD = 0.03
INITIALIZATION_MAX_P95_ABS_ERROR_RAD = 0.08


def camera_frame_map(
    source_frames: int,
    *,
    warmup_steps: int = 120,
    simulator_hz: float = 50.0,
    additional_source_frames: tuple[int, ...] = (),
) -> list[dict[str, int]]:
    """Map source RGB evidence frames to exact fixed-base replay steps.

    Frame zero is exported at the final warmup step so it reflects the settled
    reset state. Later frames use the same 30 Hz to 50 Hz rounding as the
    recorded-target policy. The map is diagnostic only, never a policy input.
    """

    if warmup_steps < 1 or simulator_hz <= 0.0:
        raise ValueError("warmup_steps and simulator_hz are invalid")
    requested = set(evidence_frame_indices(source_frames))
    for source_frame in additional_source_frames:
        if not isinstance(source_frame, int) or not 0 <= source_frame < source_frames:
            raise ValueError(
                f"additional camera source frame must be in [0,{source_frames}), got {source_frame!r}"
            )
        requested.add(source_frame)
    return [
        {
            "source_frame": source_frame,
            "simulator_step": (
                warmup_steps - 1
                if source_frame == 0
                else warmup_steps + int(round(source_frame * simulator_hz / SOURCE_FPS))
            ),
        }
        for source_frame in sorted(requested)
    ]


def materialize(
    bundle_path: Path,
    output_path: Path,
    *,
    additional_camera_source_frames: tuple[int, ...] = (),
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    actions = finite_matrix(
        bundle.get("recorded_upper_body_target_and_hand_cmd"),
        19,
        "recorded_upper_body_target_and_hand_cmd",
    )
    initial_state = finite_matrix(
        bundle.get("observed_upper_body_state_and_hand_state"),
        19,
        "observed_upper_body_state_and_hand_state",
    )
    fps = float(bundle.get("fps", 0.0))
    if fps != SOURCE_FPS:
        raise ValueError(f"replay source must be {SOURCE_FPS} Hz, got {fps}")
    payload = {
        "schema_version": "team_ramen_fixed_base_recorded_replay/v1",
        "source_episode_index": int(bundle["source_episode_index"]),
        "source_episode_name": bundle.get("source_episode_name"),
        "fps": fps,
        "action_layout": bundle["action_layout"],
        "actions": actions.tolist(),
        # This is an observed encoder stream, not another command.  It is
        # carried alongside the replay solely to evaluate whether the simulator
        # produces the state that the real camera actually observed after the
        # same command.  The runtime policy never reads it to choose actions.
        "observed_states_19d": initial_state.tolist(),
        "initial_state_19d": initial_state[0].tolist(),
        "camera_frame_map": camera_frame_map(
            len(actions), additional_source_frames=additional_camera_source_frames
        ),
        "root_replay": "forbidden_per_frame",
        "lower_body_replay": "reference_only",
    }
    atomic_write_json(output_path, payload)
    return payload


def runtime_environment(
    replay: dict[str, Any], output_dir: Path, *, initial_pose_only: bool = False
) -> dict[str, str]:
    """Return the deterministic runtime environment for one recorded replay.

    ``initial_pose_only`` is an offline camera/scene calibration probe.  It
    holds the recorded encoder state through the normal settling interval and
    stops before the first recorded command.  This keeps candidate scoring
    fast without confusing an image-only reset-pose comparison with a replay
    or a policy evaluation.
    """
    first = np.asarray(replay["actions"], dtype=np.float64)[0]
    initial_state = np.asarray(replay.get("initial_state_19d"), dtype=np.float64)
    if first.shape != (19,) or not np.isfinite(first).all():
        raise ValueError("replay must contain finite 19-D actions")
    if initial_state.shape != (19,) or not np.isfinite(initial_state).all():
        raise ValueError("replay must contain a finite initial_state_19d")
    warmup_steps = 120
    simulator_hz = 50.0
    # ``RecordedJointTargetPolicy`` indexes the 30 Hz stream with round(step
    # * source_hz / simulator_hz).  Stop at the first control step after the
    # final source frame, rather than holding the last command for an arbitrary
    # tail.  A tail changes both the physical scene and the tracking metric.
    replay_steps = math.ceil((len(replay["actions"]) - 1) * simulator_hz / float(replay["fps"])) + 1
    time_out_limit = warmup_steps if initial_pose_only else warmup_steps + replay_steps
    frame_map = replay.get("camera_frame_map")
    if not isinstance(frame_map, list) or not frame_map:
        raise ValueError("replay must contain a non-empty camera_frame_map")
    camera_steps = []
    for entry in frame_map:
        if not isinstance(entry, dict) or not isinstance(entry.get("simulator_step"), int):
            raise ValueError("camera_frame_map must contain integer simulator steps")
        camera_steps.append(entry["simulator_step"])
    if camera_steps[0] != warmup_steps - 1 or len(set(camera_steps)) != len(camera_steps):
        raise ValueError("camera_frame_map must start at terminal warmup with unique steps")
    return {
        "FLIP_TABLE_POLICY_NAME": "RecordedJointTargetPolicy",
        "FLIP_TABLE_REPLAY_ACTION_PATH": str((output_dir / "replay_actions.json").resolve()),
        "FLIP_TABLE_REPLAY_HZ": f"{float(replay['fps']):.9g}",
        # Camera tensors can retain pre-reset frames briefly in RTX. Hold the
        # measured source state for a deterministic two-second settling window.
        # The frame saved at the end of that interval therefore corresponds to
        # real source frame zero, before the first command is replayed.
        "FLIP_TABLE_REPLAY_WARMUP_STEPS": str(warmup_steps),
        "FLIP_TABLE_CAMERA_FRAME_INDEX": str(warmup_steps - 1),
        "FLIP_TABLE_CAMERA_FRAME_INDICES": ",".join(str(step) for step in camera_steps),
        # A calibration replay is not useful without the exact RGB evidence
        # needed to compare its reset pose with the source episode.  The
        # policy receives the recorded-camera remap, so persisted diagnostic
        # frames must use that exact remap as well.  Comparing a raw pinhole
        # render to source RGB would incorrectly attribute intrinsic/distortion
        # differences to the camera mount or table reset.
        "FLIP_TABLE_SAVE_CAMERA_FRAMES": "true",
        "FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY": "true",
        "FLIP_TABLE_INITIAL_UPPER_BODY_STATE": ",".join(
            f"{value:.9g}" for value in initial_state
        ),
        "FLIP_TABLE_TEST_NUM": "1",
        "FLIP_TABLE_TIME_OUT_LIMIT": str(time_out_limit),
        "FLIP_TABLE_EVAL_MODE": "nominal",
        "FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE": "false",
        "FLIP_TABLE_JOINT_NOISE_RAD": "0",
        "FLIP_TABLE_DEX1_FINGER_NOISE_M": "0",
        "FLIP_TABLE_SAVE_ACTION_STATE_TRACE": "true",
        # This writes simulator-only table/contact diagnostics alongside the
        # replay trace.  They are never exposed through policy observations.
        "FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE": "true",
        # Camera and contact fitting must begin from a repeatable scene.  The
        # randomized distributions are restored only for held-out validation;
        # otherwise a changed light, texture, mount, or friction sample could
        # be incorrectly attributed to an optimizer parameter.
        "FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION": "false",
        "FLIP_TABLE_TABLE_LONG_RANGE_M": "0",
        "FLIP_TABLE_TABLE_DEPTH_RANGE_M": "0",
        "FLIP_TABLE_TABLE_YAW_RANGE_RAD": "0",
        "FLIP_TABLE_ROBOT_DISTANCE_RANGE_M": "0",
        "FLIP_TABLE_ROBOT_LATERAL_RANGE_M": "0",
        "FLIP_TABLE_ROBOT_YAW_RANGE_RAD": "0",
        "FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS": "false",
        "FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES": "false",
        "FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS": "false",
        "FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY": "false",
        "FLIP_TABLE_EVAL_RANDOMIZE_MASS": "false",
        "FLIP_TABLE_RANDOMIZE_LIGHTING": "false",
        "FLIP_TABLE_RANDOMIZE_ROOM": "false",
        "FLIP_TABLE_RANDOMIZE_ROOM_PROPS": "false",
        "FLIP_TABLE_SIM_OUTPUT_DIR": str(output_dir.resolve()),
    }


def _shell_escape(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_runtime_env(values: dict[str, str], path: Path) -> None:
    lines = ["# Generated from an immutable real-demonstration replay bundle."]
    lines.extend(f"export {key}={_shell_escape(value)}" for key, value in sorted(values.items()))
    atomic_write_text(path, "\n".join(lines) + "\n")


def _read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_actions: list[list[float]] = []
    actual: list[list[float]] = []
    warmup: list[bool] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        action = np.asarray(row.get("source_action_19d"), dtype=np.float64)
        observed = np.asarray(row.get("source_observed_state_19d"), dtype=np.float64)
        state = np.asarray(row.get("state_after"), dtype=np.float64)
        if action.shape != (19,) or observed.shape != (19,) or state.shape != (33,):
            raise ValueError(
                f"trace row {line_number} must contain source_action_19d[19], "
                "source_observed_state_19d[19], and state_after[33]"
            )
        if not np.isfinite(action).all() or not np.isfinite(observed).all() or not np.isfinite(state).all():
            raise ValueError(f"trace row {line_number} contains NaN or Inf")
        if "replay_warmup" not in row:
            raise ValueError(f"trace row {line_number} must contain replay_warmup")
        source_actions.append([*action.tolist(), *observed.tolist()])
        actual.append(state.tolist())
        warmup.append(bool(row["replay_warmup"]))
    return (
        np.asarray(source_actions, dtype=np.float64),
        finite_matrix(actual, 33, "trace state_after"),
        np.asarray(warmup, dtype=bool),
    )


def _tracking_metrics(target: np.ndarray, state: np.ndarray) -> dict[str, float | int]:
    """Compute 19-D tracking metrics in the source dataset conventions."""

    if len(target) != len(state) or len(target) == 0:
        raise ValueError("tracking metrics require equally sized, non-empty target and state arrays")
    actual_body = state[:, UPPER_BODY_SIM_INDICES]
    body_error = actual_body - target[:, :17]
    left_finger = state[:, LEFT_DEX1_SIM_INDICES].mean(axis=1)
    right_finger = state[:, RIGHT_DEX1_SIM_INDICES].mean(axis=1)
    actual_hand_cmd = np.column_stack((left_finger, right_finger))
    actual_hand_cmd = (actual_hand_cmd - DEX1_CLOSE_POS) / (DEX1_OPEN_POS - DEX1_CLOSE_POS) * 4.5
    hand_error = actual_hand_cmd - target[:, 17:]
    return {
        "samples": int(len(target)),
        "upper_body_rmse_rad": float(np.sqrt(np.mean(body_error * body_error))),
        "upper_body_p95_abs_error_rad": float(np.quantile(np.abs(body_error), 0.95)),
        "hand_command_rmse": float(np.sqrt(np.mean(hand_error * hand_error))),
        "hand_command_p95_abs_error": float(np.quantile(np.abs(hand_error), 0.95)),
    }


def _within_initialization_tolerance(metrics: dict[str, float | int]) -> bool:
    """Return whether a rendered reset frame still represents source frame zero."""

    return bool(
        metrics["upper_body_rmse_rad"] <= INITIALIZATION_MAX_RMSE_RAD
        and metrics["upper_body_p95_abs_error_rad"] <= INITIALIZATION_MAX_P95_ABS_ERROR_RAD
    )


def analyze_trace(trace_path: Path, output_path: Path) -> dict[str, Any]:
    source, state, warmup = _read_trace(trace_path)
    if source.ndim != 2 or source.shape[1] != 38 or not np.isfinite(source).all():
        raise ValueError("trace source signals must be finite [T,38]")
    target = source[:, :19]
    observed = source[:, 19:]
    replay = ~warmup
    if not replay.any():
        raise ValueError("trace contains no replay samples after warmup")
    replay_command_metrics = _tracking_metrics(target[replay], state[replay])
    replay_observation_metrics = _tracking_metrics(observed[replay], state[replay])
    warmup_metrics = _tracking_metrics(observed[warmup], state[warmup]) if warmup.any() else None
    # Camera frame export uses the last warmup step.  Checking only an average
    # over warmup would permit a visually stale/reset-mismatched frame to enter
    # a real-vs-sim comparison, so gate the exact terminal step as well.
    initialization_metrics = (
        _tracking_metrics(observed[warmup][-1:], state[warmup][-1:])
        if warmup.any()
        else _tracking_metrics(observed[replay][:1], state[replay][:1])
    )
    initialization_passed = _within_initialization_tolerance(initialization_metrics)
    report = {
        "schema_version": "team_ramen_fixed_base_replay_trace/v3",
        "trace_path": str(trace_path),
        "replay_command_tracking": replay_command_metrics,
        "replay_observation_matching": replay_observation_metrics,
        "warmup_observed_initial_state": warmup_metrics,
        "comparison_initialization": {
            "camera_frame": "last_warmup_step" if warmup.any() else "first_replay_step",
            "metrics": initialization_metrics,
            "passed": initialization_passed,
            "max_rmse_rad": INITIALIZATION_MAX_RMSE_RAD,
            "max_p95_abs_error_rad": INITIALIZATION_MAX_P95_ABS_ERROR_RAD,
        },
        "source_hz": SOURCE_FPS,
        "simulator_hz_expected": 50.0,
        "lower_body_tracking": "not_measured: fixed-base production replay does not command lower body",
        "passed_upper_body": _within_initialization_tolerance(replay_observation_metrics),
        # A real-vs-sim image comparison is only meaningful if the rendered
        # reset image is still the recorded source initial posture.
        "eligible_for_initial_frame_comparison": initialization_passed,
    }
    atomic_write_json(output_path, report)
    return report


def analyze_initialization_trace(trace_path: Path, output_path: Path) -> dict[str, Any]:
    """Gate one initial-pose-only probe at its terminal warmup state.

    This is intentionally narrower than :func:`analyze_trace`: the probe has
    no post-warmup samples, so reporting an action-tracking score would be
    misleading.  It proves only that the saved calibration RGB corresponds to
    the recorded observed initial upper-body state.
    """

    source, state, warmup = _read_trace(trace_path)
    if source.ndim != 2 or source.shape[1] != 38 or not np.isfinite(source).all():
        raise ValueError("trace source signals must be finite [T,38]")
    if not warmup.any():
        raise ValueError("initialization probe requires at least one warmup trace row")
    observed = source[:, 19:]
    metrics = _tracking_metrics(observed[warmup][-1:], state[warmup][-1:])
    report = {
        "schema_version": "team_ramen_fixed_base_initialization_probe/v1",
        "trace_path": str(trace_path),
        "camera_frame": "last_warmup_step",
        "metrics": metrics,
        "passed": _within_initialization_tolerance(metrics),
        "max_rmse_rad": INITIALIZATION_MAX_RMSE_RAD,
        "max_p95_abs_error_rad": INITIALIZATION_MAX_P95_ABS_ERROR_RAD,
        "policy_use": "forbidden: offline calibration evidence only",
    }
    atomic_write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--episode-bundle", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument(
        "--camera-source-frame",
        type=int,
        action="append",
        default=[],
        help="also export this exact source frame for an RGB/CAD comparison",
    )
    materialize_parser.add_argument(
        "--initial-pose-only",
        action="store_true",
        help=(
            "hold the observed initial state for the normal warmup only; "
            "do not replay recorded actions (offline scene/camera calibration)"
        ),
    )
    analyze_parser = subparsers.add_parser("analyze-trace")
    analyze_parser.add_argument("--trace", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    initialization_parser = subparsers.add_parser("analyze-initialization")
    initialization_parser.add_argument("--trace", type=Path, required=True)
    initialization_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        replay = materialize(
            args.episode_bundle.expanduser().resolve(),
            output_dir / "replay_actions.json",
            additional_camera_source_frames=tuple(args.camera_source_frame),
        )
        environment = runtime_environment(
            replay,
            output_dir,
            initial_pose_only=bool(args.initial_pose_only),
        )
        write_runtime_env(environment, output_dir / "replay_runtime.env")
        print(json.dumps({"replay_actions": str(output_dir / "replay_actions.json"), "environment": environment}, indent=2))
    elif args.command == "analyze-trace":
        report = analyze_trace(args.trace.expanduser().resolve(), args.output.expanduser().resolve())
        print(json.dumps(report, indent=2))
    else:
        report = analyze_initialization_trace(
            args.trace.expanduser().resolve(), args.output.expanduser().resolve()
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
