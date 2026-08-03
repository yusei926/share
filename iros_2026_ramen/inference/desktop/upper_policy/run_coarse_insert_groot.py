#!/usr/bin/env python3
"""Run the pinned coarse-insert GR00T policy on a physical G1.

Without ``--actuate`` this performs live camera/model inference but never calls
the backend command API. Only arms14 and Dex1 two are executable; decoded EEF,
waist, base-height and navigation values are discarded before safety checks.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.flip_table_data_augmentation.teleop.config import (  # noqa: E402
    DEFAULT_TELEOP_CONFIG_PATH,
    load_teleop_config,
)
from data.flip_table_data_augmentation.teleop.contracts import (  # noqa: E402
    ArmHandTarget,
    ControlEvent,
    ControlMode,
    TeleopObservation,
)
from data.flip_table_data_augmentation.teleop.numeric import (  # noqa: E402
    G1EefForwardKinematics,
)
from inference.desktop.upper_policy.coarse_insert_groot_contract import (  # noqa: E402
    CAMERA_ROLES,
    DEX1_DATASET_OPEN_VALUE,
    MODEL_ACTION_HORIZON,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    TASK_TEXT,
    camera_payloads,
    compose_model_state,
    extract_executable_action,
)
from inference.desktop.upper_policy.gravity_compensation import (  # noqa: E402
    OfficialG1ArmGravityCompensator,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (  # noqa: E402
    CommandSequence,
    PolicyActionLimiter,
    PolicyStartPoseHold,
    append_log,
    command_from_action,
    current_camera_skew_ms,
    initialize_policy_worker_with_live_camera,
    run_blocking_check_with_pose_hold,
    run_arm_pre_motion,
    start_policy_interval_recording,
    stop_policy_interval_recording,
    return_arms_before_release,
    validate_policy_chunk,
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
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)


DEFAULT_URDF = (
    REPO_ROOT
    / "inference/orin/ros2_ws/src/g1_description/urdf/unitree_g1"
    / "g1_29dof_mode_15_with_dex1_1.urdf"
)

# The exact checkpoint training split contains 577,849 adjacent arm-target
# transitions. Its p99.99 per-transition maximum is 0.23714 rad; 127 genuine
# samples exceed the old generic 0.20-rad Diffusion threshold. GR00T returns a
# raw 16-step chunk and the runner executes only its first 8 steps through the
# independent velocity/acceleration limiter. Keep 0.30 rad as a fail-closed
# raw-model plausibility bound without rejecting supported training motion.
COARSE_INSERT_RAW_STEP_LIMIT_RAD = 0.30


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
            / "model/subtask_policy_training/deployment"
            / "real_coarse_insert_groot_n17_worker.py"
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    parser.add_argument("--expected-checkpoint-sha256", default=MODEL_SHA256)
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--preflight-samples", type=int, default=4)
    parser.add_argument("--action-execution-steps", type=int, default=8)
    parser.add_argument("--initial-delta-limit-rad", type=float, default=0.20)
    parser.add_argument(
        "--step-delta-limit-rad",
        type=float,
        default=COARSE_INSERT_RAW_STEP_LIMIT_RAD,
        help=(
            "raw executed-prefix plausibility bound; transmitted commands "
            "remain governed by the stricter velocity/acceleration limiter"
        ),
    )
    parser.add_argument("--pre-motion-arm-velocity-rad-s", type=float, default=0.5)
    parser.add_argument(
        "--pre-motion-arm-acceleration-rad-s2", type=float, default=1.0
    )
    parser.add_argument(
        "--pre-motion-waypoint-tolerance-rad", type=float, default=0.10
    )
    parser.add_argument("--pre-motion-stage-timeout-s", type=float, default=15.0)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float, default=0.5)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float, default=2.0)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float, default=0.5)
    parser.add_argument(
        "--policy-hand-acceleration-fraction-s2", type=float, default=2.0
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "outputs/real_coarse_insert_groot/last_run.jsonl",
    )
    return parser.parse_args()


class PolicyWorker:
    def __init__(
        self,
        python: Path,
        script: Path,
        checkpoint: Path,
        *,
        device: str,
        seed: int,
        model_repo_id: str = MODEL_REPO_ID,
        model_revision: str = MODEL_REVISION,
        task: str = TASK_TEXT,
        expected_checkpoint_sha256: str = MODEL_SHA256,
    ) -> None:
        self._task = task
        self._request_id = 0
        self._process = subprocess.Popen(
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
                "--expected-model-sha256",
                expected_checkpoint_sha256,
            ],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._input = self._process.stdin
        self._output = self._process.stdout
        ready = receive_message(self._output)
        contract = ready.get("contract") or {}
        expected = {
            "model_repo_id": model_repo_id,
            "model_revision": model_revision,
            "task": task,
            "weights_sha256": expected_checkpoint_sha256,
            "state_dim": 49,
            "decoded_action_dim": 53,
            "executable_action_dim": 16,
            "action_horizon": 16,
            "lower_body_command_dimensions": 0,
        }
        mismatch = {
            key: (contract.get(key), value)
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if ready.get("type") != "ready" or mismatch:
            self._process.terminate()
            self._process.wait(timeout=5.0)
            raise RuntimeError(f"coarse-insert worker contract mismatch: {mismatch}")
        self.ready = ready

    def predict(
        self, observation: TeleopObservation, state: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._request_id += 1
        send_message(
            self._input,
            {
                "type": "predict",
                "request_id": self._request_id,
                "state": np.asarray(state, dtype=np.float32).tolist(),
                "cameras": camera_payloads(observation.camera_jpeg),
                "task": self._task,
            },
        )
        response = receive_message(self._output)
        if response.get("type") == "error":
            raise RuntimeError(f"coarse-insert worker failed: {response.get('error')}")
        if (
            response.get("type") != "prediction"
            or response.get("request_id") != self._request_id
        ):
            raise RuntimeError(f"unexpected coarse-insert response: {response}")
        native = np.asarray(response["actions"], dtype=np.float64)
        return extract_executable_action(native), float(response["inference_ms"])

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            send_message(self._input, {"type": "close"})
            receive_message(self._output)
        except (BrokenPipeError, EOFError):
            pass
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5.0)


def _generation(observation: TeleopObservation) -> tuple[int, ...]:
    return tuple(
        int(observation.camera_stream_metadata[role]["jpeg_generation"])
        for role in CAMERA_ROLES
    )


def collect_observation(
    backend: Any,
    *,
    previous_generation: tuple[int, ...] | None = None,
    timeout_s: float = 5.0,
    hold: Any | None = None,
) -> TeleopObservation:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = backend.observe(timeout_s=min(0.2, timeout_s))
        validate_runtime_backend(observation)
        if hold is not None:
            hold(observation)
        if not set(CAMERA_ROLES) <= set(observation.camera_jpeg):
            raise RuntimeError("coarse-insert requires head-left and both wrist cameras")
        generation = _generation(observation)
        advanced = previous_generation is None or all(
            current > previous
            for current, previous in zip(
                generation, previous_generation, strict=True
            )
        )
        if advanced and current_camera_skew_ms(observation) <= 1000.0 / 30.0:
            return observation
        time.sleep(0.005)
    raise TimeoutError("fresh synchronized coarse-insert cameras unavailable")


def _state(observation: TeleopObservation, fk: G1EefForwardKinematics) -> np.ndarray:
    return compose_model_state(
        observation.body_joint_position_rad,
        observation.dex1_opening_fraction,
        fk(observation.body_joint_position_rad),
    )


def evaluate_policy_preflight(
    worker: PolicyWorker,
    observation: TeleopObservation,
    *,
    fk: G1EefForwardKinematics,
    config: Any,
    samples: int,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    execution_steps: int = MODEL_ACTION_HORIZON,
    enforce_initial_delta: bool = True,
    enforce_step_delta: bool = True,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Run the model and optionally enforce readiness from the measured pose."""

    model_state = _state(observation, fk)
    diagnostics: list[dict[str, float]] = []
    inference_ms: list[float] = []
    actions = np.zeros((MODEL_ACTION_HORIZON, 16))
    for _ in range(samples):
        actions, latency = worker.predict(observation, model_state)
        inference_ms.append(latency)
        diagnostics.append(
            validate_policy_chunk(
                actions,
                measured_arm=np.asarray(observation.arm_joint_position_rad),
                config=config,
                initial_delta_limit_rad=initial_delta_limit_rad,
                step_delta_limit_rad=step_delta_limit_rad,
                expected_horizon=MODEL_ACTION_HORIZON,
                execution_steps=execution_steps,
                enforce_initial_delta=enforce_initial_delta,
                enforce_step_delta=enforce_step_delta,
            )
        )
    return (
        actions,
        inference_ms[-1],
        {
            "inference_ms_mean": float(np.mean(inference_ms)),
            "inference_ms_max": float(np.max(inference_ms)),
            "initial_arm_delta_max_rad": max(
                item["initial_arm_delta_max_rad"] for item in diagnostics
            ),
            "chunk_step_delta_max_rad": max(
                item["chunk_step_delta_max_rad"] for item in diagnostics
            ),
            "full_chunk_step_delta_max_rad": max(
                item["full_chunk_step_delta_max_rad"] for item in diagnostics
            ),
            "validated_execution_steps": execution_steps,
            "camera_skew_ms": current_camera_skew_ms(observation),
        },
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.preflight_samples < 1 or args.max_seconds <= 0:
        raise ValueError("--preflight-samples and --max-seconds must be positive")
    if not 1 <= args.action_execution_steps <= MODEL_ACTION_HORIZON:
        raise ValueError("--action-execution-steps must be in [1,16]")
    for value in (
        args.initial_delta_limit_rad,
        args.step_delta_limit_rad,
        args.pre_motion_arm_velocity_rad_s,
        args.pre_motion_arm_acceleration_rad_s2,
        args.pre_motion_waypoint_tolerance_rad,
        args.pre_motion_stage_timeout_s,
        args.policy_arm_velocity_rad_s,
        args.policy_arm_acceleration_rad_s2,
        args.policy_hand_velocity_fraction_s,
        args.policy_hand_acceleration_fraction_s2,
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("all safety limits must be finite and positive")


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
    sequence = CommandSequence()
    actuation_started = False
    pre_motion_complete = False
    initial_arm_position: np.ndarray | None = None
    start_pose: SubtaskStartPose | None = None
    gravity: OfficialG1ArmGravityCompensator | None = None
    try:
        backend = RealDdsBackend(args.interface, args.image_server_ip, config)
        worker = initialize_policy_worker_with_live_camera(
            backend,
            lambda: PolicyWorker(
                args.worker_python,
                args.worker_script,
                args.checkpoint,
                device=args.device,
                seed=args.seed,
                model_repo_id=args.model_repo_id,
                model_revision=args.model_revision,
                task=args.task,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            ),
        )
        fk = G1EefForwardKinematics(args.urdf)
        observation = collect_observation(backend)
        if not args.actuate:
            actions, latency, diagnostics = evaluate_policy_preflight(
                worker,
                observation,
                fk=fk,
                config=config,
                samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                execution_steps=args.action_execution_steps,
                # This sample precedes deterministic pre-motion. Report the
                # delta now, then enforce it on a fresh dataset-frame0 sample
                # immediately before any policy command is allowed.
                enforce_initial_delta=False,
                enforce_step_delta=False,
            )
            append_log(
                args.log,
                {
                    "event": "preflight_prediction",
                    "policy": "coarse_insert_groot_n17_v2",
                    "actuate_requested": False,
                    "model_repo_id": args.model_repo_id,
                    "model_revision": args.model_revision,
                    "task": args.task,
                    "model_state_dim": 49,
                    "model_action_dim": 53,
                    "executable_action_dim": 16,
                    "discarded_action_dimensions": (
                        "eef18+waist3+base_height1+navigation3"
                    ),
                    "lower_body_command_dimensions": 0,
                    **diagnostics,
                },
            )
            print(
                "[preflight] live 3-camera coarse-insert GR00T inference OK; "
                "NO command sent; EEF/waist/base/nav discarded",
                flush=True,
            )
            return 0
        confirmation = input(
            "Harness / E-stop / table clearance confirmed. On the first "
            "Enter, both Dex1 hands open fully and the arms move "
            "shoulders-back -> lateral-high -> forward-outside -> ready -> "
            "dataset-frame0. Press Enter to start arm-only pre-motion, or "
            "Ctrl+C to cancel: "
        )
        if confirmation != "":
            print(
                "[cancelled] Enter must be pressed without text; NO command sent",
                flush=True,
            )
            return 2
        verify_regular_mode(args.interface)
        start_pose = subtask_start_pose_for_model(args.model_repo_id)
        if start_pose.dex1_opening_fraction is None:
            raise RuntimeError(
                "coarse-insert deployment requires a verified dataset "
                "frame-zero Dex1 grasp width"
            )
        observation = backend.observe(timeout_s=1.0)
        validate_runtime_backend(observation)
        initial_arm_position = np.asarray(
            observation.arm_joint_position_rad, dtype=np.float64
        ).copy()
        start_waypoints = (
            ArmPreMotionWaypoint(
                "hands_full_open_before_clearance",
                (0.0,) * 14,
                tuple(range(14)),
            ),
        ) + build_arm_pre_motion_waypoints(start_pose.arm_position_rad)
        startup_hand_targets = {
            waypoint.name: (1.0, 1.0) for waypoint in start_waypoints
        }
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
                "startup_dex1_opening_fraction": [1.0, 1.0],
                "operator_gate_sequence": [
                    "enter_1_open_hands_and_move_arms_to_dataset_frame0",
                    "enter_2_close_hands_to_dataset_frame0_grasp",
                    "enter_3_start_policy_inference",
                ],
                "lower_body_command_dimensions": 0,
            },
        )
        gravity = OfficialG1ArmGravityCompensator()
        actuation_started = True
        observation = run_arm_pre_motion(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=sequence,
            gravity_compensator=gravity,
            arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.pre_motion_arm_acceleration_rad_s2,
            waypoint_tolerance_rad=args.pre_motion_waypoint_tolerance_rad,
            stage_timeout_s=args.pre_motion_stage_timeout_s,
            waypoints=start_waypoints,
            hand_targets_by_waypoint=startup_hand_targets,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=(
                args.policy_hand_acceleration_fraction_s2
            ),
        )
        pre_motion_complete = True

        # Enter 2: keep the measured dataset-frame0 arm pose fixed and close
        # only Dex1 to the frame-zero grasp width recovered from the exact
        # checkpoint training split.  The generic gate continues refreshing
        # the arm_sdk watchdog while the operator positions the workpiece.
        observation = wait_for_policy_start_with_hold(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=sequence,
            gravity_compensator=gravity,
            latest=observation,
            prompt=(
                "Arms are held at dataset frame-zero with both hands fully "
                "open. Confirm the insert target is inside both Dex1 jaws, "
                "then press Enter to close to the dataset initial grasp "
                "width, or Ctrl+C to stop: "
            ),
            invalid_prompt=(
                "Press Enter to close both Dex1 hands to the dataset initial "
                "grasp width, or Ctrl+C to stop: "
            ),
            waiting_event="dataset_grasp_gate_waiting",
            confirmed_event="dataset_grasp_gate_confirmed",
        )
        grasp_waypoint = ArmPreMotionWaypoint(
            "dataset_frame0_hand_grasp",
            (0.0,) * 14,
            tuple(range(14)),
        )
        observation = run_arm_pre_motion(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=sequence,
            gravity_compensator=gravity,
            arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.pre_motion_arm_acceleration_rad_s2,
            waypoint_tolerance_rad=args.pre_motion_waypoint_tolerance_rad,
            stage_timeout_s=args.pre_motion_stage_timeout_s,
            waypoints=(grasp_waypoint,),
            hand_targets_by_waypoint={
                grasp_waypoint.name: start_pose.dex1_opening_fraction
            },
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=(
                args.policy_hand_acceleration_fraction_s2
            ),
        )
        append_log(
            args.log,
            {
                "event": "dataset_frame0_grasp_reached",
                "target_dex1_fraction": list(
                    start_pose.dex1_opening_fraction
                ),
                "measured_dex1_fraction": list(
                    observation.dex1_opening_fraction
                ),
                "lower_body_command_dimensions": 0,
            },
        )

        # Enter 3: continue to hold both the arm pose and the measured grasp.
        # Only after this gate do we repeat the Regular/camera/model checks and
        # allow policy-generated arm/Dex1 targets to be transmitted.
        observation = wait_for_policy_start_with_hold(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=sequence,
            gravity_compensator=gravity,
            latest=observation,
        )
        pose_hold = PolicyStartPoseHold(
            backend,
            command_sequence=sequence,
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
        observation = collect_observation(
            backend,
            previous_generation=_generation(observation),
            hold=pose_hold.refresh,
        )
        (
            (actions, latency, diagnostics),
            latest,
        ) = run_blocking_check_with_pose_hold(
            lambda: evaluate_policy_preflight(
                worker,
                observation,
                fk=fk,
                config=config,
                samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                execution_steps=args.action_execution_steps,
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
        append_log(
            args.log,
            {"event": "armed_prediction", "inference_ms": latency, **diagnostics},
        )
        deadline = time.monotonic() + args.max_seconds
        period = 1.0 / config.rates.command_hz
        while time.monotonic() < deadline:
            for desired in actions[: args.action_execution_steps]:
                if time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                observation = backend.observe(
                    timeout_s=config.safety.command_hold_timeout_s
                )
                validate_runtime_backend(observation)
                safe = limiter.apply(desired)
                backend.apply(
                    command_from_action(
                        sequence.next(),
                        safe,
                        arm_feedforward_torque_nm=gravity.torque_nm(safe[:14]),
                    )
                )
                append_log(
                    args.log,
                    {
                        "event": "command",
                        "arm_target_rad": safe[:14].tolist(),
                        "dex1_target_fraction": (
                            safe[14:] / DEX1_DATASET_OPEN_VALUE
                        ).tolist(),
                        "lower_body_command_dimensions": 0,
                    },
                )
                time.sleep(max(0.0, period - (time.monotonic() - tick)))
            if time.monotonic() >= deadline:
                break
            observation = collect_observation(
                backend, previous_generation=_generation(observation)
            )
            model_state = _state(observation, fk)
            actions, _ = worker.predict(observation, model_state)
            validate_policy_chunk(
                actions,
                measured_arm=np.asarray(observation.arm_joint_position_rad),
                config=config,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                expected_horizon=MODEL_ACTION_HORIZON,
                execution_steps=args.action_execution_steps,
            )
        return 0
    except KeyboardInterrupt:
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
                        command_sequence=sequence,
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
                    )
                try:
                    latest = backend.observe(timeout_s=1.0)
                    backend.apply(
                        ArmHandTarget(
                            sequence=sequence.next(),
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
