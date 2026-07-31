#!/usr/bin/env python3
"""Materialize auditable real-demonstration controller replays.

The default 16-D replay is the deployable arms-plus-Dex1 path through the
organizer WBC.  A separate 31-D body diagnostic is available solely for
controller identification; it commands recorded joints but never teleports
the floating root and is not a policy/runtime deployment path.
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
    full_body_diagnostic: bool = False,
) -> dict[str, Any]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if full_body_diagnostic:
        raw_actions = bundle.get("recorded_full_body_hand_target_31d")
        observed_key = "observed_full_body_state_and_hand_state"
        action_width = 31
        state_width = 31
        schema_version = "team_ramen_full_body_diagnostic_recorded_replay/v1"
        body_mode = "full_body_diagnostic"
        policy_name = "RecordedFullBodyTargetPolicy"
        initial_key = "initial_state_31d"
    else:
        raw_actions = bundle.get("recorded_arm_hand_target_16d")
        observed_key = "observed_upper_body_state_and_hand_state"
        action_width = 16
        state_width = 19
        schema_version = "team_ramen_balanced_wbc_recorded_replay/v2"
        body_mode = "balanced_wbc"
        policy_name = "RecordedJointTargetPolicy"
        initial_key = "initial_state_19d"
    if raw_actions is None and not full_body_diagnostic:
        legacy = finite_matrix(
            bundle.get("recorded_upper_body_target_and_hand_cmd"),
            19,
            "legacy recorded_upper_body_target_and_hand_cmd",
        )
        actions = np.concatenate((legacy[:, 3:17], legacy[:, 17:19]), axis=1)
    else:
        actions = finite_matrix(raw_actions, action_width, "recorded replay actions")
    initial_state = finite_matrix(
        bundle.get(observed_key),
        state_width,
        observed_key,
    )
    fps = float(bundle.get("fps", 0.0))
    if fps != SOURCE_FPS:
        raise ValueError(f"replay source must be {SOURCE_FPS} Hz, got {fps}")
    payload = {
        "schema_version": schema_version,
        "source_episode_index": int(bundle["source_episode_index"]),
        "source_episode_name": bundle.get("source_episode_name"),
        "fps": fps,
        "action_layout": bundle["action_layout"],
        "actions": actions.tolist(),
        # This is an observed encoder stream, not another command.  It is
        # carried alongside the replay solely to evaluate whether the simulator
        # produces the state that the real camera actually observed after the
        # same command.  The runtime policy never reads it to choose actions.
        f"observed_states_{state_width}d": initial_state.tolist(),
        initial_key: initial_state[0].tolist(),
        "camera_frame_map": camera_frame_map(
            len(actions), additional_source_frames=additional_camera_source_frames
        ),
        "body_mode": body_mode,
        "policy_name": policy_name,
        "root_replay": "forbidden_per_frame",
        "lower_body_owner": (
            "organizer_wbc" if not full_body_diagnostic else "recorded_joint_targets_diagnostic_only"
        ),
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
    full_body_diagnostic = replay.get("body_mode") == "full_body_diagnostic"
    action_width = 31 if full_body_diagnostic else 16
    state_width = 31 if full_body_diagnostic else 19
    state_key = f"initial_state_{state_width}d"
    first = np.asarray(replay["actions"], dtype=np.float64)[0]
    initial_state = np.asarray(replay.get(state_key), dtype=np.float64)
    if first.shape != (action_width,) or not np.isfinite(first).all():
        raise ValueError(f"replay must contain finite {action_width}-D actions")
    if initial_state.shape != (state_width,) or not np.isfinite(initial_state).all():
        raise ValueError(f"replay must contain a finite {state_key}")
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
        "FLIP_TABLE_POLICY_NAME": (
            "RecordedFullBodyTargetPolicy" if full_body_diagnostic else "RecordedJointTargetPolicy"
        ),
        "FLIP_TABLE_REPLAY_ACTION_PATH": str((output_dir / "replay_actions.json").resolve()),
        "FLIP_TABLE_REPLAY_HZ": f"{float(replay['fps']):.9g}",
        # The default preserves the source command timestamps exactly.  A
        # nonzero value is an explicit actuator-identification hypothesis,
        # backed by source command/encoder timing, never a policy feature.
        "FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS": "0",
        # Calibration traces and requested RGB evidence retain the 50 Hz
        # control clock.  The review MP4 is intentionally sparse to prevent
        # video encoding from changing which physics candidates are practical
        # to evaluate.  Its sampling rate is recorded in every trace.
        "FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ": "10",
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
        **(
            {
                "FLIP_TABLE_INITIAL_FULL_BODY_STATE": ",".join(
                    f"{value:.9g}" for value in initial_state
                )
            }
            if full_body_diagnostic
            else {
                "FLIP_TABLE_INITIAL_UPPER_BODY_STATE": ",".join(
                    f"{value:.9g}" for value in initial_state
                )
            }
        ),
        "FLIP_TABLE_TEST_NUM": "1",
        "FLIP_TABLE_TIME_OUT_LIMIT": str(time_out_limit),
        "FLIP_TABLE_EVAL_MODE": "nominal",
        "FLIP_TABLE_SIM_BODY_MODE": "full_body_diagnostic" if full_body_diagnostic else "balanced_wbc",
        "FLIP_TABLE_LOCK_LOWER_BODY": "false",
        "FLIP_TABLE_LOCK_ROBOT_ROOT": "false",
        "FLIP_TABLE_FIX_ROOT_LINK": "false",
        "FLIP_TABLE_REQUIRE_WAIST_LOCK": "false",
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
        action = np.asarray(row.get("source_action_16d"), dtype=np.float64)
        observed = np.asarray(row.get("source_observed_state_19d"), dtype=np.float64)
        state = np.asarray(row.get("state_after"), dtype=np.float64)
        if action.shape != (16,) or observed.shape != (19,) or state.shape != (33,):
            raise ValueError(
                f"trace row {line_number} must contain source_action_16d[16], "
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
    """Compute action16 or observed-state19 tracking metrics."""

    if len(target) != len(state) or len(target) == 0:
        raise ValueError("tracking metrics require equally sized, non-empty target and state arrays")
    actual_body = state[:, UPPER_BODY_SIM_INDICES]
    if target.shape[1] == 16:
        body_error = actual_body[:, 3:17] - target[:, :14]
        hand_target = target[:, 14:16]
    elif target.shape[1] == 19:
        body_error = actual_body - target[:, :17]
        hand_target = target[:, 17:19]
    else:
        raise ValueError(f"tracking target must be [T,16] or [T,19], got {target.shape}")
    left_finger = state[:, LEFT_DEX1_SIM_INDICES].mean(axis=1)
    right_finger = state[:, RIGHT_DEX1_SIM_INDICES].mean(axis=1)
    actual_hand_cmd = np.column_stack((left_finger, right_finger))
    actual_hand_cmd = (actual_hand_cmd - DEX1_CLOSE_POS) / (DEX1_OPEN_POS - DEX1_CLOSE_POS) * 4.5
    hand_error = actual_hand_cmd - hand_target
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


def _interpolate_source_xyz(samples: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Interpolate a source XYZ stream at fractional source-frame positions."""

    if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) < 2:
        raise ValueError("source root positions must be [T,3] with T >= 2")
    clipped = np.clip(np.asarray(positions, dtype=np.float64), 0.0, float(len(samples) - 1))
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, len(samples) - 1)
    fraction = (clipped - lower)[:, None]
    return (1.0 - fraction) * samples[lower] + fraction * samples[upper]


def _root_trajectory_diagnostics(rows: list[dict[str, Any]], bundle_path: Path | None) -> dict[str, Any]:
    """Compare root motion diagnostically without using it to control sim."""

    result: dict[str, Any] = {
        "policy_use": "forbidden: offline actuator/physics diagnostic only",
        "available": False,
    }
    if bundle_path is None:
        result["reason"] = "episode bundle not supplied"
        return result
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    source = np.asarray(bundle.get("observed_root_pose_xyz_wxyz"), dtype=np.float64)
    fps = float(bundle.get("fps", 0.0))
    if source.ndim != 2 or source.shape[1] != 7 or len(source) < 2 or not np.isfinite(source).all():
        raise ValueError("episode bundle must contain finite observed_root_pose_xyz_wxyz [T,7]")
    if fps != SOURCE_FPS:
        raise ValueError(f"root diagnostic source must be {SOURCE_FPS} Hz")
    replay_rows = [row for row in rows if not bool(row.get("replay_warmup", False))]
    sim = np.asarray(
        [row.get("simulator_scene_diagnostics", {}).get("root_pose_world_xyzw", [])[:3] for row in replay_rows],
        dtype=np.float64,
    )
    if sim.shape != (len(replay_rows), 3) or not np.isfinite(sim).all():
        raise ValueError("trace lacks finite simulator root positions")
    source_position = (np.arange(len(replay_rows), dtype=np.float64) + 1.0) * SOURCE_FPS / 50.0
    source_xyz = _interpolate_source_xyz(source[:, :3], source_position)
    source_relative = source_xyz - source_xyz[0]
    sim_relative = sim - sim[0]
    error = sim_relative - source_relative
    source_displacement = np.linalg.norm(source_relative, axis=1)
    sim_displacement = np.linalg.norm(sim_relative, axis=1)
    return {
        "policy_use": "forbidden: offline actuator/physics diagnostic only",
        "available": True,
        "source_episode_index": int(bundle["source_episode_index"]),
        "comparison": "relative_xyz_from_first_post_warmup_sample",
        "source_hz": SOURCE_FPS,
        "simulator_hz": 50.0,
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "p95_abs_error_m": float(np.quantile(np.abs(error), 0.95)),
        "source_displacement_max_m": float(source_displacement.max()),
        "simulator_displacement_max_m": float(sim_displacement.max()),
    }


def analyze_trace(
    trace_path: Path, output_path: Path, *, episode_bundle_path: Path | None = None
) -> dict[str, Any]:
    source, state, warmup = _read_trace(trace_path)
    if source.ndim != 2 or source.shape[1] != 35 or not np.isfinite(source).all():
        raise ValueError("trace source signals must be finite [T,35]")
    target = source[:, :16]
    observed = source[:, 16:]
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
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
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
        "root_trajectory_diagnostics": _root_trajectory_diagnostics(rows, episode_bundle_path),
        "passed_upper_body": _within_initialization_tolerance(replay_observation_metrics),
        # A real-vs-sim image comparison is only meaningful if the rendered
        # reset image is still the recorded source initial posture.
        "eligible_for_initial_frame_comparison": initialization_passed,
    }
    atomic_write_json(output_path, report)
    return report


def _full_body_tracking_metrics(target: np.ndarray, actual: np.ndarray) -> dict[str, float | int]:
    """Summarize a 31-D diagnostic replay without mixing controller families."""

    if target.shape != actual.shape or target.ndim != 2 or target.shape[1] != 31 or len(target) == 0:
        raise ValueError("full-body tracking requires equally sized [T,31] arrays")

    error = actual - target

    def metric(values: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(np.mean(values * values))),
            "p95_abs_error": float(np.quantile(np.abs(values), 0.95)),
        }

    return {
        "samples": int(len(target)),
        "lower_body_joint_rad": metric(error[:, :12]),
        "upper_body_joint_rad": metric(error[:, 12:29]),
        "hand_encoder_units": metric(error[:, 29:31]),
    }


def analyze_full_body_trace(trace_path: Path, output_path: Path) -> dict[str, Any]:
    """Analyze direct body29/Dex1 replay used only for controller diagnosis.

    Unlike :func:`analyze_trace`, this records whether the floating-base model
    remains physically viable.  It does not convert the result into a policy
    score or use the table/root trace to change a command.
    """

    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("full-body trace is empty")
    replay = [row for row in rows if not bool(row.get("replay_warmup", False))]
    if len(replay) < 2:
        raise ValueError("full-body trace has fewer than two post-warmup rows")

    def matrix(key: str) -> np.ndarray:
        value = np.asarray([row.get(key) for row in replay], dtype=np.float64)
        return finite_matrix(value, 31, key)

    command = matrix("source_action_31d")
    observed = matrix("source_observed_state_31d")
    actual = matrix("actual_state_31d")

    root_poses_all = np.asarray(
        [row.get("simulator_scene_diagnostics", {}).get("root_pose_world_xyzw") for row in rows],
        dtype=np.float64,
    )
    table_positions_all = np.asarray(
        [
            row.get("simulator_scene_diagnostics", {})
            .get("white_table", {})
            .get("position_world_m")
            for row in rows
        ],
        dtype=np.float64,
    )
    if root_poses_all.shape != (len(rows), 7) or not np.isfinite(root_poses_all).all():
        raise ValueError("full-body trace lacks finite root_pose_world_xyzw diagnostics")
    if table_positions_all.shape != (len(rows), 3) or not np.isfinite(table_positions_all).all():
        raise ValueError("full-body trace lacks finite white-table position diagnostics")

    warmup_count = len(rows) - len(replay)
    root_poses = root_poses_all[warmup_count:]
    table_positions = table_positions_all[warmup_count:]
    root_displacement = np.linalg.norm(root_poses_all[:, :3] - root_poses_all[0, :3], axis=1)
    table_displacement = np.linalg.norm(table_positions_all - table_positions_all[0], axis=1)
    report = {
        "schema_version": "team_ramen_full_body_diagnostic_trace/v1",
        "trace_path": str(trace_path),
        "policy_use": "forbidden: controller identification only",
        "root_replay": "forbidden_per_frame",
        "command_tracking": _full_body_tracking_metrics(command, actual),
        "observed_state_matching": _full_body_tracking_metrics(observed, actual),
        "floating_base_diagnostics": {
            "trace_initial_root_position_world_m": root_poses_all[0, :3].tolist(),
            "replay_initial_root_position_world_m": root_poses[0, :3].tolist(),
            "final_root_position_world_m": root_poses_all[-1, :3].tolist(),
            "root_displacement_max_m": float(root_displacement.max()),
            "root_height_min_m": float(root_poses_all[:, 2].min()),
            "root_height_final_m": float(root_poses_all[-1, 2]),
            "warmup_root_height_drop_m": float(root_poses_all[0, 2] - root_poses[0, 2]),
            "table_displacement_max_m": float(table_displacement.max()),
        },
        "interpretation": (
            "This diagnostic intentionally leaves the floating root dynamic. "
            "A large root displacement demonstrates that recorded position targets alone do not "
            "substitute for the real robot's balance controller; it must not be hidden by a root lock "
            "or used to justify a deployable whole-body policy."
        ),
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
    if source.ndim != 2 or source.shape[1] != 35 or not np.isfinite(source).all():
        raise ValueError("trace source signals must be finite [T,35]")
    if not warmup.any():
        raise ValueError("initialization probe requires at least one warmup trace row")
    observed = source[:, 16:]
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
    materialize_parser.add_argument(
        "--full-body-diagnostic",
        action="store_true",
        help=(
            "replay recorded body29 plus Dex1 commands through direct joint drives; "
            "offline controller identification only, never a deployable policy path"
        ),
    )
    analyze_parser = subparsers.add_parser("analyze-trace")
    analyze_parser.add_argument("--trace", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument(
        "--episode-bundle",
        type=Path,
        help="optional source bundle for diagnostic-only relative root-trajectory comparison",
    )
    full_body_analyze_parser = subparsers.add_parser("analyze-full-body-trace")
    full_body_analyze_parser.add_argument("--trace", type=Path, required=True)
    full_body_analyze_parser.add_argument("--output", type=Path, required=True)
    initialization_parser = subparsers.add_parser("analyze-initialization")
    initialization_parser.add_argument("--trace", type=Path, required=True)
    initialization_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "materialize":
        if args.initial_pose_only and args.full_body_diagnostic:
            raise ValueError("--initial-pose-only and --full-body-diagnostic cannot be combined")
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        replay = materialize(
            args.episode_bundle.expanduser().resolve(),
            output_dir / "replay_actions.json",
            additional_camera_source_frames=tuple(args.camera_source_frame),
            full_body_diagnostic=bool(args.full_body_diagnostic),
        )
        environment = runtime_environment(
            replay,
            output_dir,
            initial_pose_only=bool(args.initial_pose_only),
        )
        write_runtime_env(environment, output_dir / "replay_runtime.env")
        print(json.dumps({"replay_actions": str(output_dir / "replay_actions.json"), "environment": environment}, indent=2))
    elif args.command == "analyze-trace":
        report = analyze_trace(
            args.trace.expanduser().resolve(),
            args.output.expanduser().resolve(),
            episode_bundle_path=(
                args.episode_bundle.expanduser().resolve() if args.episode_bundle is not None else None
            ),
        )
        print(json.dumps(report, indent=2))
    elif args.command == "analyze-full-body-trace":
        report = analyze_full_body_trace(args.trace.expanduser().resolve(), args.output.expanduser().resolve())
        print(json.dumps(report, indent=2))
    else:
        report = analyze_initialization_trace(
            args.trace.expanduser().resolve(), args.output.expanduser().resolve()
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
