#!/usr/bin/env python3
"""Run GR00T N1.7 pick-table-leg inference on a physical G1.

This is intentionally independent from the ACT and Diffusion runners.  The
model consumes its original 38-D whole-body state and emits its original 38-D
absolute action, but this process discards root/legs/waist and sends only
arms14 + Dex1-1 left/right.  Unitree Regular Mode owns balance and locomotion.

Without ``--actuate`` the command performs live four-camera/model inference and
sends no robot command.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.flip_table_data_augmentation.teleop.config import (
    DEFAULT_TELEOP_CONFIG_PATH,
    TeleopConfig,
    load_teleop_config,
)
from data.flip_table_data_augmentation.teleop.contracts import (
    ArmHandTarget,
    ControlEvent,
    ControlMode,
    TeleopObservation,
)
from inference.desktop.upper_policy.gravity_compensation import (
    OfficialG1ArmGravityCompensator,
)
from inference.desktop.upper_policy.groot_pick_leg_contract import (
    CAMERA_ROLES,
    DEX1_DATASET_OPEN_VALUE,
    MODEL_ACTION_HORIZON,
    MODEL_ARM_SLICE,
    MODEL_HAND_SLICE,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_STATE_DIM,
    TASK_TEXT,
    camera_payloads,
    compose_model_state,
    extract_executable_action,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (
    CommandSequence,
    PolicyActionLimiter,
    PolicyStartPoseHold,
    append_log,
    command_from_action,
    initialize_policy_worker_with_live_camera,
    run_blocking_check_with_pose_hold,
    run_arm_pre_motion,
    start_policy_interval_recording,
    stop_policy_interval_recording,
    return_arms_before_release,
    validate_runtime_backend,
    verify_regular_mode,
    verify_regular_mode_after_release,
    wait_for_policy_start_with_hold,
)
from inference.desktop.upper_policy.pre_motion import (
    ArmPreMotionWaypoint,
    build_arm_pre_motion_waypoints,
)
from inference.desktop.upper_policy.subtask_start_pose import (
    SubtaskStartPose,
    subtask_start_pose_for_model,
)
from inference.desktop.upper_policy.worker_protocol import receive_message, send_message


def build_pick_leg_start_motion(
    start_pose: SubtaskStartPose,
) -> tuple[tuple[ArmPreMotionWaypoint, ...], dict[str, tuple[float, float]]]:
    """Build the collision-clearance path and its explicit Dex1 targets."""

    if start_pose.dex1_opening_fraction is None:
        raise ValueError(
            "physical GR00T evaluation requires a pinned dataset frame-zero "
            "Dex1 opening"
        )
    waypoints = (
        ArmPreMotionWaypoint(
            "hands_full_open_before_clearance",
            (0.0,) * 14,
            tuple(range(14)),
        ),
    ) + build_arm_pre_motion_waypoints(start_pose.arm_position_rad)
    hand_targets = {waypoint.name: (1.0, 1.0) for waypoint in waypoints}
    hand_targets["dataset_frame0_pose"] = tuple(
        start_pose.dex1_opening_fraction
    )
    return waypoints, hand_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--image-server-ip", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=REPO_ROOT / "model/subtask_policy_training/.venv/bin/python",
    )
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=(
            REPO_ROOT
            / "model/subtask_policy_training/deployment/real_groot_n17_worker.py"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--preflight-samples", type=int, default=2)
    parser.add_argument(
        "--action-execution-steps",
        type=int,
        default=MODEL_ACTION_HORIZON,
        help="Decoded absolute targets executed before replanning (official checkpoint horizon: 16).",
    )
    parser.add_argument(
        "--initial-delta-limit-rad",
        type=float,
        default=0.50,
        help=(
            "Maximum raw model-target distance at policy start. The command "
            "is still acceleration/velocity limited before DDS transmission."
        ),
    )
    parser.add_argument("--step-delta-limit-rad", type=float, default=0.20)
    parser.add_argument("--pre-motion-arm-velocity-rad-s", type=float, default=0.5)
    parser.add_argument(
        "--pre-motion-arm-acceleration-rad-s2", type=float, default=1.0
    )
    parser.add_argument(
        "--pre-motion-waypoint-tolerance-rad", type=float, default=0.10
    )
    parser.add_argument("--pre-motion-stage-timeout-s", type=float, default=15.0)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float, default=0.75)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float, default=3.0)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float, default=0.75)
    parser.add_argument(
        "--policy-hand-acceleration-fraction-s2", type=float, default=3.0
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "outputs/real_groot_pick_leg/last_run.jsonl",
    )
    return parser.parse_args()


@dataclass
class PolicyWorker:
    process: subprocess.Popen[bytes]
    input: Any
    output: Any
    ready: dict[str, Any]
    task: str
    request_id: int = 0

    @classmethod
    def start(
        cls,
        python: Path,
        script: Path,
        checkpoint: Path,
        *,
        device: str,
        seed: int,
        model_repo_id: str,
        model_revision: str,
        task: str = TASK_TEXT,
    ) -> "PolicyWorker":
        process = subprocess.Popen(
            [
                str(python),
                str(script),
                "--checkpoint",
                str(checkpoint),
                "--device",
                device,
                "--seed",
                str(seed),
                "--model-repo-id",
                model_repo_id,
                "--model-revision",
                model_revision,
                "--task",
                task,
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        try:
            ready = receive_message(process.stdout)
            if ready.get("type") != "ready":
                raise RuntimeError(f"GR00T worker did not become ready: {ready}")
            contract = ready.get("contract") or {}
            expected = {
                "model_repo_id": model_repo_id,
                "model_revision": model_revision,
                "task": task,
                "state_dim": MODEL_STATE_DIM,
                "decoded_action_dim": 38,
                "executable_action_dim": 16,
                "action_horizon": MODEL_ACTION_HORIZON,
                "lower_body_command_dimensions": 0,
            }
            mismatch = {
                key: (contract.get(key), value)
                for key, value in expected.items()
                if contract.get(key) != value
            }
            if mismatch:
                raise RuntimeError(f"GR00T worker contract mismatch: {mismatch}")
            return cls(process, process.stdin, process.stdout, ready, task)
        except Exception:
            process.terminate()
            process.wait(timeout=5.0)
            raise

    def predict(self, observation: TeleopObservation) -> tuple[np.ndarray, float]:
        self.request_id += 1
        send_message(
            self.input,
            {
                "type": "predict",
                "request_id": self.request_id,
                "state": compose_model_state(
                    observation.body_joint_position_rad,
                    observation.dex1_opening_fraction,
                ).tolist(),
                "cameras": camera_payloads(observation.camera_jpeg),
                "task": self.task,
            },
        )
        response = receive_message(self.output)
        if response.get("type") == "error":
            raise RuntimeError(f"GR00T worker failed: {response.get('error')}")
        if (
            response.get("type") != "prediction"
            or response.get("request_id") != self.request_id
        ):
            raise RuntimeError(f"unexpected GR00T response: {response}")
        decoded = np.asarray(response["actions"], dtype=np.float64)
        return extract_executable_action(decoded), float(response["inference_ms"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                send_message(self.input, {"type": "close"})
                receive_message(self.output)
            except (BrokenPipeError, EOFError):
                pass
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5.0)


def _camera_generation(observation: TeleopObservation) -> tuple[int, ...]:
    try:
        return tuple(
            int(observation.camera_stream_metadata[role]["jpeg_generation"])
            for role in CAMERA_ROLES
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("four-camera generation metadata is unavailable") from exc


def _camera_skew_ms(observation: TeleopObservation) -> float:
    timestamps = np.asarray(
        [observation.camera_capture_monotonic_ns[role] for role in CAMERA_ROLES],
        dtype=np.int64,
    )
    if timestamps.shape != (4,) or np.any(timestamps <= 0):
        raise ValueError("four positive camera timestamps are required")
    return float((timestamps.max() - timestamps.min()) / 1.0e6)


def collect_fresh_observation(
    backend: Any,
    *,
    previous_generation: tuple[int, ...] | None = None,
    timeout_s: float = 5.0,
    maximum_skew_ms: float = 1000.0 / 30.0,
    hold: Any | None = None,
) -> TeleopObservation:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = backend.observe(timeout_s=min(1.0, timeout_s))
        validate_runtime_backend(observation)
        if hold is not None:
            hold(observation)
        if set(observation.camera_jpeg) != set(CAMERA_ROLES):
            raise RuntimeError("GR00T requires head stereo and both wrist cameras")
        generation = _camera_generation(observation)
        advanced = previous_generation is None or all(
            current > previous
            for current, previous in zip(
                generation, previous_generation, strict=True
            )
        )
        if advanced and _camera_skew_ms(observation) <= maximum_skew_ms:
            return observation
        time.sleep(0.005)
    raise TimeoutError("fresh synchronized four-camera observation was not available")


def validate_state_distribution(
    observation: TeleopObservation,
    checkpoint: Path,
    *,
    validate_executable_state: bool = True,
) -> dict[str, Any]:
    state = compose_model_state(
        observation.body_joint_position_rad,
        observation.dex1_opening_fraction,
    ).astype(np.float64)
    stats = json.loads((checkpoint / "statistics.json").read_text(encoding="utf-8"))[
        "new_embodiment"
    ]["state"]
    q01 = np.concatenate(
        (
            np.asarray(stats["robot_q"]["q01"], dtype=np.float64),
            np.asarray(stats["hand"]["q01"], dtype=np.float64),
        )
    )
    q99 = np.concatenate(
        (
            np.asarray(stats["robot_q"]["q99"], dtype=np.float64),
            np.asarray(stats["hand"]["q99"], dtype=np.float64),
        )
    )
    mean = np.concatenate(
        (
            np.asarray(stats["robot_q"]["mean"], dtype=np.float64),
            np.asarray(stats["hand"]["mean"], dtype=np.float64),
        )
    )
    std = np.concatenate(
        (
            np.asarray(stats["robot_q"]["std"], dtype=np.float64),
            np.asarray(stats["hand"]["std"], dtype=np.float64),
        )
    )
    raw_minimum = np.concatenate(
        (
            np.asarray(stats["robot_q"]["min"], dtype=np.float64),
            np.asarray(stats["hand"]["min"], dtype=np.float64),
        )
    )
    raw_maximum = np.concatenate(
        (
            np.asarray(stats["robot_q"]["max"], dtype=np.float64),
            np.asarray(stats["hand"]["max"], dtype=np.float64),
        )
    )
    if any(
        value.shape != (MODEL_STATE_DIM,)
        for value in (q01, q99, mean, std, raw_minimum, raw_maximum)
    ):
        raise ValueError("GR00T state statistics must flatten to 38 dimensions")
    robust_margin = np.maximum(0.05, 0.15 * (q99 - q01))
    # A 0.05-rad absolute allowance covers small Regular-Mode balance offsets
    # and encoder/pose differences while remaining far below a meaningful G1
    # leg/waist configuration change. The z>8 guard remains independent.
    raw_margin = np.maximum(0.05, 0.05 * (raw_maximum - raw_minimum))
    z = np.abs((state - mean) / np.maximum(std, 1.0e-6))
    robust_outside = (
        (state < q01 - robust_margin)
        | (state > q99 + robust_margin)
        | (z > 8.0)
    )
    raw_outside = (
        (state < raw_minimum - raw_margin)
        | (state > raw_maximum + raw_margin)
        | (z > 8.0)
    )
    # Global root x/y/yaw are unavailable on hardware and intentionally replaced
    # with a session-local identity proxy. They are observation-only and their
    # predicted action counterparts are discarded.
    robust_outside[:7] = False
    raw_outside[:7] = False

    # Training statistics are model diagnostics, not robot safety limits.
    # Root/legs/waist are inputs only and Unitree Regular Mode remains their
    # sole controller. Arm/Dex1 excursions are reported too, but must not
    # silently alter or reject a physically valid model action. Physical
    # limits are enforced independently by validate_action_chunk and the
    # actuator-side limiter.
    context_indices = np.arange(7, MODEL_ARM_SLICE.start)
    executable_indices = np.concatenate(
        (
            np.arange(MODEL_ARM_SLICE.start, MODEL_ARM_SLICE.stop),
            np.arange(MODEL_HAND_SLICE.start, MODEL_HAND_SLICE.stop),
        )
    )
    diagnostic_outside = np.zeros(MODEL_STATE_DIM, dtype=bool)
    diagnostic_outside[context_indices] = raw_outside[context_indices]
    if validate_executable_state:
        diagnostic_outside[executable_indices] = robust_outside[
            executable_indices
        ]
    context_tail = context_indices[robust_outside[context_indices]]
    outside_indices = np.flatnonzero(diagnostic_outside)
    return {
        "model_state_38d": state.tolist(),
        "state_max_abs_z_non_root": float(z[7:].max()),
        "state_context_tail_indices": context_tail.astype(int).tolist(),
        "state_context_tail_count": int(context_tail.size),
        "state_context_max_abs_z": float(z[context_indices].max()),
        "state_executable_validation_enabled": validate_executable_state,
        "state_distribution_warning_indices": outside_indices.astype(int).tolist(),
        "state_distribution_warning_count": int(outside_indices.size),
        "training_distribution_action_modified": False,
        "root_pose_proxy_xyz_wxyz": state[:7].tolist(),
        "camera_payload_skew_ms": _camera_skew_ms(observation),
    }


def validate_action_chunk(
    executable: np.ndarray,
    *,
    measured_arm: np.ndarray,
    config: TeleopConfig,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    enforce_initial_delta: bool = True,
) -> dict[str, float]:
    values = np.asarray(executable, dtype=np.float64)
    measured = np.asarray(measured_arm, dtype=np.float64)
    if values.shape != (MODEL_ACTION_HORIZON, 16) or not np.isfinite(values).all():
        raise ValueError(f"executable GR00T chunk must be finite [16,16], got {values.shape}")
    if measured.shape != (14,) or not np.isfinite(measured).all():
        raise ValueError("measured arm must be finite [14]")
    arms = values[:, :14]
    hand = values[:, 14:]
    lower = np.asarray(config.safety.arm_position_lower_rad, dtype=np.float64)
    upper = np.asarray(config.safety.arm_position_upper_rad, dtype=np.float64)
    invalid_arm = (arms < lower) | (arms > upper)
    if np.any(invalid_arm):
        step, joint = np.argwhere(invalid_arm)[0]
        raise ValueError(
            "GR00T arm target exceeds configured hardware margin "
            f"(step={step}, joint={joint}, value={arms[step, joint]:.4f})"
        )
    if np.any((hand < 0.0) | (hand > DEX1_DATASET_OPEN_VALUE)):
        raise ValueError("GR00T Dex1 target is outside dataset range [0,4.5]")
    initial_delta = float(np.max(np.abs(arms[0] - measured)))
    step_delta = float(np.max(np.abs(np.diff(arms, axis=0))))
    if enforce_initial_delta and initial_delta > initial_delta_limit_rad:
        raise ValueError(
            f"first GR00T arm target is {initial_delta:.4f} rad from measured pose; "
            f"limit is {initial_delta_limit_rad:.4f}"
        )
    if step_delta > step_delta_limit_rad:
        raise ValueError(
            f"GR00T chunk contains a {step_delta:.4f} rad arm step; "
            f"limit is {step_delta_limit_rad:.4f}"
        )
    return {
        "initial_arm_delta_max_rad": initial_delta,
        "chunk_step_delta_max_rad": step_delta,
        "dex1_min_fraction": float(hand.min() / DEX1_DATASET_OPEN_VALUE),
        "dex1_max_fraction": float(hand.max() / DEX1_DATASET_OPEN_VALUE),
    }


def evaluate_policy_preflight(
    worker: PolicyWorker,
    observation: TeleopObservation,
    *,
    checkpoint: Path,
    config: TeleopConfig,
    samples: int,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    validate_executable_state: bool = True,
    enforce_initial_delta: bool = True,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, float]]:
    """Evaluate the full state/action contract at the current staged pose."""

    state_diagnostics = validate_state_distribution(
        observation,
        checkpoint,
        validate_executable_state=validate_executable_state,
    )
    predictions: list[np.ndarray] = []
    inference_times: list[float] = []
    prediction_diagnostics: list[dict[str, float]] = []
    for _ in range(samples):
        actions, inference_ms = worker.predict(observation)
        predictions.append(actions)
        inference_times.append(inference_ms)
        prediction_diagnostics.append(
            validate_action_chunk(
                actions,
                measured_arm=np.asarray(observation.arm_joint_position_rad),
                config=config,
                initial_delta_limit_rad=initial_delta_limit_rad,
                step_delta_limit_rad=step_delta_limit_rad,
                enforce_initial_delta=enforce_initial_delta,
            )
        )
    diagnostics = {
        "initial_arm_delta_max_rad": max(
            value["initial_arm_delta_max_rad"]
            for value in prediction_diagnostics
        ),
        "chunk_step_delta_max_rad": max(
            value["chunk_step_delta_max_rad"]
            for value in prediction_diagnostics
        ),
        "arm_prediction_std_max_rad": float(
            np.stack(predictions)[:, :, :14].std(axis=0).max()
        ),
        "inference_ms_mean": float(np.mean(inference_times)),
        "inference_ms_max": float(np.max(inference_times)),
    }
    return predictions[-1], inference_times[-1], state_diagnostics, diagnostics


def _validate_args(args: argparse.Namespace) -> None:
    if args.preflight_samples < 1:
        raise ValueError("--preflight-samples must be positive")
    if not 1 <= args.action_execution_steps <= MODEL_ACTION_HORIZON:
        raise ValueError("--action-execution-steps must be in [1,16]")
    for value, label in (
        (args.max_seconds, "--max-seconds"),
        (args.initial_delta_limit_rad, "--initial-delta-limit-rad"),
        (args.step_delta_limit_rad, "--step-delta-limit-rad"),
        (args.pre_motion_arm_velocity_rad_s, "--pre-motion-arm-velocity-rad-s"),
        (
            args.pre_motion_arm_acceleration_rad_s2,
            "--pre-motion-arm-acceleration-rad-s2",
        ),
        (
            args.pre_motion_waypoint_tolerance_rad,
            "--pre-motion-waypoint-tolerance-rad",
        ),
        (args.pre_motion_stage_timeout_s, "--pre-motion-stage-timeout-s"),
        (args.policy_arm_velocity_rad_s, "--policy-arm-velocity-rad-s"),
        (args.policy_arm_acceleration_rad_s2, "--policy-arm-acceleration-rad-s2"),
        (args.policy_hand_velocity_fraction_s, "--policy-hand-velocity-fraction-s"),
        (
            args.policy_hand_acceleration_fraction_s2,
            "--policy-hand-acceleration-fraction-s2",
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")


def main() -> int:
    from data.flip_table_data_augmentation.teleop.desktop_preview import (
        enable_camera_preview_for_policy_runner,
    )

    enable_camera_preview_for_policy_runner()
    args = parse_args()
    _validate_args(args)
    config = load_teleop_config(args.config)
    from data.flip_table_data_augmentation.teleop.real.backend import RealDdsBackend

    worker: PolicyWorker | None = None
    backend: RealDdsBackend | None = None
    command_sequence = CommandSequence()
    actuation_started = False
    pre_motion_complete = False
    initial_arm_position: np.ndarray | None = None
    start_pose: SubtaskStartPose | None = None
    gravity: OfficialG1ArmGravityCompensator | None = None
    try:
        backend = RealDdsBackend(args.interface, args.image_server_ip, config)
        worker = initialize_policy_worker_with_live_camera(
            backend,
            lambda: PolicyWorker.start(
                args.worker_python,
                args.worker_script,
                args.checkpoint,
                device=args.device,
                seed=args.seed,
                model_repo_id=args.model_repo_id,
                model_revision=args.model_revision,
                task=args.task,
            ),
        )
        print(
            "[model] GR00T N1.7 ready "
            f"revision={args.model_revision[:12]} device={worker.ready['device']}",
            flush=True,
        )
        observation = collect_fresh_observation(backend)
        if not args.actuate:
            (
                _actions,
                _inference_ms,
                state_diagnostics,
                diagnostics,
            ) = evaluate_policy_preflight(
                worker,
                observation,
                checkpoint=args.checkpoint,
                config=config,
                samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                # The non-actuating diagnostic observes the robot before the
                # collision-aware start pose is applied. Validate lower-body
                # context and action safety without pretending that this is
                # already the executable start pose.
                validate_executable_state=False,
                # This pose is intentionally replaced by pre-motion before
                # actuation. Preserve the raw delta as a diagnostic, but do
                # not reject a read-only run for being far from that future
                # start pose.
                enforce_initial_delta=False,
            )
            append_log(
                args.log,
                {
                    "event": "preflight_prediction",
                    "policy": "groot_n1.7_pick_table_leg",
                    "actuate_requested": False,
                    "model_repo_id": args.model_repo_id,
                    "model_revision": args.model_revision,
                    "task": args.task,
                    "camera_mapping": {
                        "cam_0": "head_left",
                        "cam_1": "head_right",
                        "cam_2": "left_wrist",
                        "cam_3": "right_wrist",
                    },
                    "model_state_dim": 38,
                    "model_action_dim": 38,
                    "executable_action_dim": 16,
                    "discarded_action_dimensions": "root7+legs12+waist3",
                    "lower_body_command_dimensions": 0,
                    **state_diagnostics,
                    **diagnostics,
                },
            )
            print(
                "[preflight] live 4-camera GR00T inference OK; NO command sent "
                f"samples={args.preflight_samples} "
                f"inference_ms_mean={diagnostics['inference_ms_mean']:.1f} "
                f"initial_delta={diagnostics['initial_arm_delta_max_rad']:.4f}rad "
                f"step_delta={diagnostics['chunk_step_delta_max_rad']:.4f}rad "
                f"camera_skew={state_diagnostics['camera_payload_skew_ms']:.2f}ms",
                flush=True,
            )
            print(
                "[preflight-only] root/legs/waist predictions were discarded; "
                f"Regular Mode remains owner; log={args.log}",
                flush=True,
            )
            return 0

        # Check the observation-only root/lower-body context before moving the
        # arms.  The initial arm pose is intentionally excluded here because
        # collision-aware staging moves it into the policy start distribution.
        context_diagnostics = validate_state_distribution(
            observation,
            args.checkpoint,
            validate_executable_state=False,
        )
        append_log(
            args.log,
            {
                "event": "pre_actuation_context_preflight",
                "lower_body_command_dimensions": 0,
                **context_diagnostics,
            },
        )
        if context_diagnostics["state_context_tail_count"]:
            print(
                "[preflight] observation-only lower-body state is in the "
                "recorded training range but outside its central q01-q99 "
                "range; continuing with actual measured state "
                f"(dimensions={context_diagnostics['state_context_tail_indices']})",
                flush=True,
            )

        confirmation = input(
            "Harness / E-stop / table clearance confirmed. Arms will move "
            "shoulders-back -> lateral-high -> forward-outside -> ready -> "
            "dataset-frame0. "
            "Press Enter to start the arm-only pre-motion, or Ctrl+C to cancel: "
        )
        if confirmation != "":
            print(
                "[cancelled] Enter must be pressed without typing text; "
                "NO command sent",
                flush=True,
            )
            return 2
        verify_regular_mode(args.interface)
        start_pose = subtask_start_pose_for_model(args.model_repo_id)
        observation = backend.observe(timeout_s=1.0)
        validate_runtime_backend(observation)
        initial_arm_position = np.asarray(
            observation.arm_joint_position_rad, dtype=np.float64
        ).copy()
        # Open both hands before moving either arm so a previously closed hand
        # cannot catch on the table during the collision-clearance path.  Keep
        # them fully open through all clearance waypoints, then transition to
        # the exact dataset frame-zero median only at the final arm pose.
        start_waypoints, startup_hand_targets = build_pick_leg_start_motion(
            start_pose
        )
        append_log(
            args.log,
            {
                "event": "dataset_frame0_pose_selected",
                "model_repo_id": args.model_repo_id,
                "dataset_repo_id": start_pose.dataset_repo_id,
                "dataset_revision": start_pose.dataset_revision,
                "training_episode_count": start_pose.training_episode_count,
                "statistic": start_pose.statistic,
                "exact_training_revision": start_pose.exact_training_revision,
                "pose_sha256": start_pose.sha256,
                "arm_position_rad": list(start_pose.arm_position_rad),
                "dex1_opening_fraction": list(
                    start_pose.dex1_opening_fraction
                ),
                "lower_body_command_dimensions": 0,
            },
        )
        print(
            "[pre-motion] final startup target is the training-dataset "
            f"frame-0 median ({start_pose.dataset_repo_id}, "
            f"sha256={start_pose.sha256[:12]})",
            flush=True,
        )
        gravity = OfficialG1ArmGravityCompensator()
        # Set before the first deterministic staging command so all exception
        # paths perform the controlled arm_sdk release.
        actuation_started = True
        observation = run_arm_pre_motion(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=command_sequence,
            gravity_compensator=gravity,
            arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.pre_motion_arm_acceleration_rad_s2,
            waypoint_tolerance_rad=args.pre_motion_waypoint_tolerance_rad,
            stage_timeout_s=args.pre_motion_stage_timeout_s,
            waypoints=start_waypoints,
            hand_targets_by_waypoint=startup_hand_targets,
        )
        pre_motion_complete = True
        observation = wait_for_policy_start_with_hold(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=command_sequence,
            gravity_compensator=gravity,
            latest=observation,
        )
        pose_hold = PolicyStartPoseHold(
            backend,
            command_sequence=command_sequence,
            gravity_compensator=gravity,
            latest=observation,
        )
        _, observation = run_blocking_check_with_pose_hold(
            lambda: verify_regular_mode(args.interface, arm_sdk_active=True),
            backend=backend,
            config=config,
            pose_hold=pose_hold,
            latest=observation,
        )
        start_policy_interval_recording(backend, args.log)
        previous_generation = _camera_generation(observation)
        observation = collect_fresh_observation(
            backend,
            previous_generation=previous_generation,
            hold=pose_hold.refresh,
        )
        (
            (
                actions,
                inference_ms,
                state_diagnostics,
                action_diagnostics,
            ),
            latest,
        ) = run_blocking_check_with_pose_hold(
            lambda: evaluate_policy_preflight(
                worker,
                observation,
                checkpoint=args.checkpoint,
                config=config,
                samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
            ),
            backend=backend,
            config=config,
            pose_hold=pose_hold,
            latest=observation,
        )
        limiter = PolicyActionLimiter(
            np.asarray(latest.arm_joint_position_rad),
            np.asarray(latest.dex1_opening_fraction),
            command_hz=config.rates.command_hz,
            arm_velocity_rad_s=args.policy_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.policy_arm_acceleration_rad_s2,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=args.policy_hand_acceleration_fraction_s2,
            arm_position_lower_rad=config.safety.arm_position_lower_rad,
            arm_position_upper_rad=config.safety.arm_position_upper_rad,
        )
        preview_limiter = PolicyActionLimiter(
            np.asarray(latest.arm_joint_position_rad),
            np.asarray(latest.dex1_opening_fraction),
            command_hz=config.rates.command_hz,
            arm_velocity_rad_s=args.policy_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.policy_arm_acceleration_rad_s2,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=args.policy_hand_acceleration_fraction_s2,
            arm_position_lower_rad=config.safety.arm_position_lower_rad,
            arm_position_upper_rad=config.safety.arm_position_upper_rad,
        )
        first_limited = preview_limiter.apply(actions[0])
        action_diagnostics["first_transmitted_arm_delta_max_rad"] = float(
            np.max(
                np.abs(
                    first_limited[:14]
                    - np.asarray(latest.arm_joint_position_rad, dtype=np.float64)
                )
            )
        )
        append_log(
            args.log,
            {
                "event": "armed_prediction",
                "inference_ms": inference_ms,
                **state_diagnostics,
                **action_diagnostics,
            },
        )
        print(
            "[armed] fresh Regular/state/4-camera prediction verified; "
            "sending arms14+Dex1 only "
            f"(raw_target_delta="
            f"{action_diagnostics['initial_arm_delta_max_rad']:.4f}rad, "
            f"first_transmitted_delta="
            f"{action_diagnostics['first_transmitted_arm_delta_max_rad']:.4f}rad)",
            flush=True,
        )
        deadline = time.monotonic() + args.max_seconds
        command_period_s = 1.0 / config.rates.command_hz
        while time.monotonic() < deadline:
            for desired in actions[: args.action_execution_steps]:
                if time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                live = backend.observe(
                    timeout_s=min(0.05, config.safety.command_hold_timeout_s)
                )
                validate_runtime_backend(live)
                limited = limiter.apply(desired)
                torque = gravity.torque_nm(limited[:14])
                sequence = command_sequence.next()
                backend.apply(
                    command_from_action(
                        sequence,
                        limited,
                        arm_feedforward_torque_nm=torque,
                    )
                )
                append_log(
                    args.log,
                    {
                        "event": "command",
                        "sequence": sequence,
                        "desired_arm_target_rad": desired[:14].tolist(),
                        "arm_target_rad": limited[:14].tolist(),
                        "dex1_target_fraction": (
                            limited[14:] / DEX1_DATASET_OPEN_VALUE
                        ).tolist(),
                        "arm_feedforward_torque_nm": torque.tolist(),
                        "lower_body_command_dimensions": 0,
                    },
                )
                time.sleep(max(0.0, command_period_s - (time.monotonic() - tick)))
            if time.monotonic() >= deadline:
                break
            previous_generation = _camera_generation(observation)
            observation = collect_fresh_observation(
                backend,
                previous_generation=previous_generation,
                timeout_s=config.safety.command_hold_timeout_s * 0.75,
            )
            state_diagnostics = validate_state_distribution(observation, args.checkpoint)
            actions, inference_ms = worker.predict(observation)
            action_diagnostics = validate_action_chunk(
                actions,
                measured_arm=np.asarray(observation.arm_joint_position_rad),
                config=config,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
            )
            append_log(
                args.log,
                {
                    "event": "chunk",
                    "inference_ms": inference_ms,
                    **state_diagnostics,
                    **action_diagnostics,
                },
            )
        print(f"[done] reached --max-seconds={args.max_seconds:.2f}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[interrupt] Ctrl+C received; controlled arm_sdk release", flush=True)
        return 130
    finally:
        if backend is not None:
            stop_policy_interval_recording(backend, args.log)
            if actuation_started:
                if (
                    initial_arm_position is not None
                    and start_pose is not None
                    and gravity is not None
                    and pre_motion_complete
                ):
                    return_arms_before_release(
                        backend,
                        config=config,
                        log_path=args.log,
                        command_sequence=command_sequence,
                        gravity_compensator=gravity,
                        initial_arm_position_rad=initial_arm_position,
                        dataset_frame0_arm_rad=start_pose.arm_position_rad,
                        arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
                        arm_acceleration_rad_s2=(
                            args.pre_motion_arm_acceleration_rad_s2
                        ),
                        waypoint_tolerance_rad=(
                            args.pre_motion_waypoint_tolerance_rad
                        ),
                        stage_timeout_s=args.pre_motion_stage_timeout_s,
                        dex1_return_opening_fraction=(1.0, 1.0),
                    )
                try:
                    latest = backend.observe(timeout_s=1.0)
                    backend.apply(
                        ArmHandTarget(
                            sequence=command_sequence.next(),
                            monotonic_ns=time.monotonic_ns(),
                            mode=ControlMode.IDLE,
                            event=ControlEvent.QUIT,
                            arm_position_rad=latest.arm_joint_position_rad,
                            dex1_opening_fraction=latest.dex1_opening_fraction,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[shutdown] IDLE request failed: {exc}", file=sys.stderr)
            try:
                backend.close()
            finally:
                if actuation_started:
                    verify_regular_mode_after_release(args.interface)
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
