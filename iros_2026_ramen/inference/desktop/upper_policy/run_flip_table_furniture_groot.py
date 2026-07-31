#!/usr/bin/env python3
"""Run the finalized Furniture-GR00T flip-table policy on a physical G1."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
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
from inference.desktop.upper_policy.furniture_groot_contract import (  # noqa: E402
    CAMERA_KEYS,
    DATASET_FPS,
    DEX1_DATASET_OPEN_VALUE,
    MODEL_ACTION_HORIZON,
    TASK_TEXT,
    VIDEO_DELTA_INDICES,
    camera_payload_history,
    compose_model_state,
)
from inference.desktop.upper_policy.gravity_compensation import (  # noqa: E402
    OfficialG1ArmGravityCompensator,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (  # noqa: E402
    CommandSequence,
    PolicyActionLimiter,
    PolicyStartPoseHold,
    append_log,
    camera_generation,
    command_from_action,
    is_fresh_policy_observation,
    run_blocking_check_with_pose_hold,
    run_arm_pre_motion,
    return_arms_before_release,
    validate_policy_chunk,
    validate_runtime_backend,
    verify_regular_mode,
    verify_regular_mode_after_release,
    wait_for_policy_start_with_hold,
)
from inference.desktop.upper_policy.pre_motion import build_arm_pre_motion_waypoints
from inference.desktop.upper_policy.subtask_start_pose import (
    FLIP_TABLE_V2_FRAME0,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)
from model.subtask_policy_training.gr00t.temporal_ensemble import (  # noqa: E402
    PhysicalTargetTemporalEnsembler,
)


DEFAULT_URDF = (
    REPO_ROOT
    / "inference"
    / "orin"
    / "ros2_ws"
    / "src"
    / "g1_description"
    / "urdf"
    / "unitree_g1"
    / "g1_29dof_mode_15_with_dex1_1.urdf"
)
COMMAND_HZ = DATASET_FPS
HISTORY_TARGET_SECONDS = abs(VIDEO_DELTA_INDICES[0]) / DATASET_FPS
HISTORY_TOLERANCE_SECONDS = 1.5 / DATASET_FPS
VALID_EXECUTION_STEPS = frozenset({5, 10, 20})
VALID_TEMPORAL_LAMBDA_LABELS = frozenset({"none", "-0.25", "-0.1", "0"})
MAX_INFERENCE_AGE_SECONDS = MODEL_ACTION_HORIZON / COMMAND_HZ


@dataclass(frozen=True)
class InferenceRequestContext:
    """State snapshot needed to validate and align an asynchronous H40 chunk."""

    origin_step: int
    measured_arm_rad: np.ndarray
    submitted_monotonic_s: float

    def __post_init__(self) -> None:
        measured = np.asarray(self.measured_arm_rad, dtype=np.float64).copy()
        if (
            self.origin_step < 0
            or measured.shape != (14,)
            or not np.isfinite(measured).all()
            or not math.isfinite(self.submitted_monotonic_s)
        ):
            raise ValueError("invalid asynchronous inference request context")
        measured.setflags(write=False)
        object.__setattr__(self, "measured_arm_rad", measured)

    def has_remaining_target(self, current_step: int) -> bool:
        return int(current_step) < self.origin_step + MODEL_ACTION_HORIZON

    def age_seconds(self, now_monotonic_s: float) -> float:
        return float(now_monotonic_s) - self.submitted_monotonic_s


def release_execution_schedule(
    contract: dict[str, Any],
) -> tuple[int, float | None, str]:
    """Read the only execution schedule that passed the release evaluation."""

    execution_steps = int(contract.get("execution_steps", -1))
    label = str(contract.get("temporal_lambda_label"))
    decay = contract.get("temporal_lambda")
    if execution_steps not in VALID_EXECUTION_STEPS:
        raise ValueError(f"unvalidated execution interval: {execution_steps}")
    if label not in VALID_TEMPORAL_LAMBDA_LABELS:
        raise ValueError(f"unvalidated temporal lambda: {label!r}")
    expected_decay = None if label == "none" else float(label)
    if decay is None:
        actual_decay = None
    else:
        actual_decay = float(decay)
        if not math.isfinite(actual_decay):
            raise ValueError("temporal lambda must be finite or None")
    if actual_decay != expected_decay:
        raise ValueError(
            "temporal lambda numeric value differs from its release label"
        )
    return execution_steps, actual_decay, label


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
            / "model"
            / "subtask_policy_training"
            / "deployment"
            / "real_furniture_groot_n17_worker.py"
        ),
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--pre-motion-only", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--preflight-samples", type=int, default=4)
    parser.add_argument("--initial-delta-limit-rad", type=float, default=0.20)
    parser.add_argument("--step-delta-limit-rad", type=float, default=0.20)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float, default=1.0)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float, default=4.0)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float, default=1.0)
    parser.add_argument("--policy-hand-acceleration-fraction-s2", type=float, default=4.0)
    parser.add_argument("--pre-motion-arm-velocity-rad-s", type=float, default=0.5)
    parser.add_argument("--pre-motion-arm-acceleration-rad-s2", type=float, default=1.0)
    parser.add_argument("--pre-motion-waypoint-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--pre-motion-stage-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "outputs/real_furniture_groot/last_run.jsonl",
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
    ) -> None:
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
        try:
            ready = receive_message(self._output)
            if ready.get("type") != "ready":
                raise RuntimeError(f"policy worker did not become ready: {ready}")
            contract = ready.get("contract") or {}
            if (
                contract.get("task") != TASK_TEXT
                or contract.get("state_dim") != 49
                or contract.get("logical_action_dim") != 53
                or contract.get("executable_action_dim") != 16
                or contract.get("action_horizon") != 40
                or contract.get("camera_roles")
                != ["head_left", "left_wrist", "right_wrist"]
                or contract.get("video_delta_indices") != list(VIDEO_DELTA_INDICES)
                or contract.get("lower_body_command_dimensions") != 0
            ):
                raise RuntimeError(f"unexpected policy worker contract: {contract}")
            release_execution_schedule(contract)
            self.ready = ready
        except Exception:
            self._process.terminate()
            self._process.wait(timeout=3.0)
            raise

    def predict(
        self,
        history: list[TeleopObservation],
        state: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        self._request_id += 1
        send_message(
            self._input,
            {
                "type": "predict",
                "request_id": self._request_id,
                "state": np.asarray(state, dtype=np.float32).tolist(),
                "camera_history": camera_payload_history(history),
                "task": TASK_TEXT,
            },
        )
        response = receive_message(self._output)
        if response.get("type") == "error":
            raise RuntimeError(f"policy worker failed: {response.get('error')}")
        if (
            response.get("type") != "prediction"
            or response.get("request_id") != self._request_id
        ):
            raise RuntimeError(f"unexpected policy response: {response}")
        return np.asarray(response["actions"], dtype=np.float64), float(
            response["inference_ms"]
        )

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                send_message(self._input, {"type": "close"})
                receive_message(self._output)
            except (BrokenPipeError, EOFError):
                pass
            try:
                self._process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=3.0)

    def terminate(self) -> None:
        """Stop an in-flight worker without waiting for an inference response."""
        if self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3.0)


class TemporalObservationBuffer:
    """Select the two real camera bundles nearest the official [-20,0] times."""

    def __init__(self, *, capacity: int = 96) -> None:
        self._samples: deque[TeleopObservation] = deque(maxlen=capacity)
        self._previous_generation: tuple[int, ...] | None = None

    def add(self, observation: TeleopObservation) -> bool:
        if not is_fresh_policy_observation(
            observation,
            self._previous_generation,
            maximum_skew_ms=1000.0 / DATASET_FPS,
        ):
            return False
        self._samples.append(observation)
        self._previous_generation = camera_generation(observation)
        return True

    @staticmethod
    def _timestamp_s(observation: TeleopObservation) -> float:
        values = np.asarray(
            [
                observation.camera_capture_monotonic_ns[role]
                for role in ("head_left", "left_wrist", "right_wrist")
            ],
            dtype=np.float64,
        )
        return float(np.mean(values) * 1.0e-9)

    def pair(self) -> list[TeleopObservation]:
        if len(self._samples) < 2:
            raise RuntimeError("camera history has fewer than two fresh bundles")
        latest = self._samples[-1]
        latest_time = self._timestamp_s(latest)
        target_time = latest_time - HISTORY_TARGET_SECONDS
        candidates = list(self._samples)[:-1]
        earlier = min(
            candidates,
            key=lambda observation: abs(self._timestamp_s(observation) - target_time),
        )
        error = abs(self._timestamp_s(earlier) - target_time)
        if error > HISTORY_TOLERANCE_SECONDS:
            raise RuntimeError(
                "camera history cannot satisfy [-20,0] at 30 Hz: "
                f"timing error={error * 1000.0:.1f}ms"
            )
        return [earlier, latest]


def state_for_observation(
    observation: TeleopObservation,
    *,
    fk: G1EefForwardKinematics,
) -> np.ndarray:
    return compose_model_state(
        observation.body_joint_position_rad,
        observation.dex1_opening_fraction,
        fk(observation.body_joint_position_rad),
    )


def collect_temporal_history(
    backend: Any,
    buffer: TemporalObservationBuffer,
    *,
    timeout_s: float,
    hold: Any | None = None,
) -> list[TeleopObservation]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = backend.observe(timeout_s=min(0.05, timeout_s))
        validate_runtime_backend(observation)
        buffer.add(observation)
        if hold is not None:
            hold(observation)
        try:
            return buffer.pair()
        except RuntimeError:
            time.sleep(0.002)
    raise TimeoutError("official [-20,0] policy camera history was not available")


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_seconds <= 0 or args.preflight_samples < 1:
        raise ValueError("--max-seconds and --preflight-samples must be positive")
    if args.pre_motion_only and not args.actuate:
        raise ValueError("--pre-motion-only requires --actuate")
    for value, label in (
        (args.initial_delta_limit_rad, "--initial-delta-limit-rad"),
        (args.step_delta_limit_rad, "--step-delta-limit-rad"),
        (args.policy_arm_velocity_rad_s, "--policy-arm-velocity-rad-s"),
        (args.policy_arm_acceleration_rad_s2, "--policy-arm-acceleration-rad-s2"),
        (args.policy_hand_velocity_fraction_s, "--policy-hand-velocity-fraction-s"),
        (
            args.policy_hand_acceleration_fraction_s2,
            "--policy-hand-acceleration-fraction-s2",
        ),
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
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be finite and positive")
def main() -> int:
    args = parse_args()
    _validate_args(args)
    config = load_teleop_config(args.config)
    from data.flip_table_data_augmentation.teleop.real.backend import RealDdsBackend

    worker: PolicyWorker | None = None
    backend: RealDdsBackend | None = None
    executor: ThreadPoolExecutor | None = None
    pending: Future[tuple[np.ndarray, float]] | None = None
    command_sequence = CommandSequence()
    actuation_started = False
    pre_motion_complete = False
    initial_arm_position: np.ndarray | None = None
    gravity: OfficialG1ArmGravityCompensator | None = None
    try:
        worker = PolicyWorker(
            args.worker_python,
            args.worker_script,
            args.checkpoint,
            device=args.device,
            seed=args.seed,
        )
        execution_steps, temporal_lambda, temporal_lambda_label = (
            release_execution_schedule(worker.ready["contract"])
        )
        fk = G1EefForwardKinematics(args.urdf)
        backend = RealDdsBackend(args.interface, args.image_server_ip, config)
        buffer = TemporalObservationBuffer()
        history = collect_temporal_history(backend, buffer, timeout_s=3.0)
        state = state_for_observation(history[-1], fk=fk)
        measured = np.asarray(history[-1].arm_joint_position_rad)
        preflight_actions: list[np.ndarray] = []
        preflight_latency: list[float] = []
        preflight_diagnostics: list[dict[str, float]] = []
        for _ in range(args.preflight_samples):
            actions, inference_ms = worker.predict(history, state)
            preflight_actions.append(actions)
            preflight_latency.append(inference_ms)
            preflight_diagnostics.append(
                validate_policy_chunk(
                    actions,
                    measured_arm=measured,
                    config=config,
                    initial_delta_limit_rad=args.initial_delta_limit_rad,
                    step_delta_limit_rad=args.step_delta_limit_rad,
                    expected_horizon=40,
                )
            )
        stacked = np.stack(preflight_actions)
        diagnostics = {
            "inference_ms_mean": float(np.mean(preflight_latency)),
            "inference_ms_max": float(np.max(preflight_latency)),
            "initial_arm_delta_max_rad": max(
                value["initial_arm_delta_max_rad"]
                for value in preflight_diagnostics
            ),
            "chunk_step_delta_max_rad": max(
                value["chunk_step_delta_max_rad"]
                for value in preflight_diagnostics
            ),
            "arm_prediction_std_max_rad": float(
                stacked[:, :, :14].std(axis=0).max()
            ),
            "dex1_prediction_std_max": float(
                stacked[:, :, 14:].std(axis=0).max()
            ),
        }
        append_log(
            args.log,
            {
                "event": "preflight_prediction",
                "actuate_requested": args.actuate,
                "contract": worker.ready["contract"],
                "state_49d": state.tolist(),
                "history_delta_indices": list(VIDEO_DELTA_INDICES),
                "command_hz": COMMAND_HZ,
                "temporal_lambda": temporal_lambda_label,
                "execution_steps": execution_steps,
                **diagnostics,
            },
        )
        print(
            "[preflight] 49D / 3-camera[-20,0] / H40 inference OK; "
            f"NO command sent, mean={diagnostics['inference_ms_mean']:.1f}ms",
            flush=True,
        )
        if not args.actuate:
            return 0

        confirmation = input(
            "Harness / E-stop / table clearance confirmed. Arms will move "
            "shoulders-back -> lateral-high -> forward-outside -> ready -> "
            "dataset-frame0. "
            "Press Enter to start arm-only pre-motion, or Ctrl+C to cancel: "
        )
        if confirmation != "":
            print(
                "[cancelled] Enter must be pressed without text; NO command sent",
                flush=True,
            )
            return 2
        verify_regular_mode(args.interface)
        latest = backend.observe(timeout_s=1.0)
        validate_runtime_backend(latest)
        initial_arm_position = np.asarray(
            latest.arm_joint_position_rad, dtype=np.float64
        ).copy()
        start_waypoints = build_arm_pre_motion_waypoints(
            FLIP_TABLE_V2_FRAME0.arm_position_rad
        )
        append_log(
            args.log,
            {
                "event": "dataset_frame0_pose_selected",
                "dataset_repo_id": FLIP_TABLE_V2_FRAME0.dataset_repo_id,
                "dataset_revision": FLIP_TABLE_V2_FRAME0.dataset_revision,
                "training_episode_count": FLIP_TABLE_V2_FRAME0.training_episode_count,
                "statistic": FLIP_TABLE_V2_FRAME0.statistic,
                "exact_training_revision": FLIP_TABLE_V2_FRAME0.exact_training_revision,
                "pose_sha256": FLIP_TABLE_V2_FRAME0.sha256,
                "arm_position_rad": list(FLIP_TABLE_V2_FRAME0.arm_position_rad),
                "lower_body_command_dimensions": 0,
            },
        )
        gravity = OfficialG1ArmGravityCompensator()
        actuation_started = True
        latest = run_arm_pre_motion(
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
        )
        pre_motion_complete = True
        if args.pre_motion_only:
            return 0
        latest = wait_for_policy_start_with_hold(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=command_sequence,
            gravity_compensator=gravity,
            latest=latest,
        )
        pose_hold = PolicyStartPoseHold(
            backend,
            command_sequence=command_sequence,
            gravity_compensator=gravity,
            latest=latest,
        )
        _, latest = run_blocking_check_with_pose_hold(
            lambda: verify_regular_mode(args.interface, arm_sdk_active=True),
            backend=backend,
            config=config,
            pose_hold=pose_hold,
            latest=latest,
        )

        buffer = TemporalObservationBuffer()
        history = collect_temporal_history(
            backend,
            buffer,
            timeout_s=3.0,
            hold=pose_hold.refresh,
        )
        state = state_for_observation(history[-1], fk=fk)
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="groot-inference")
        initial_context = InferenceRequestContext(
            origin_step=0,
            measured_arm_rad=np.asarray(
                history[-1].arm_joint_position_rad,
                dtype=np.float64,
            ),
            submitted_monotonic_s=time.monotonic(),
        )
        pending = executor.submit(worker.predict, history, state)
        while not pending.done():
            observation = backend.observe(
                timeout_s=min(0.05, config.safety.command_hold_timeout_s)
            )
            buffer.add(observation)
            pose_hold.refresh(observation)
            if (
                initial_context.age_seconds(time.monotonic())
                > MAX_INFERENCE_AGE_SECONDS
            ):
                raise TimeoutError(
                    "initial Furniture-GR00T inference exceeded the H40 "
                    "validity window"
                )
        actions, inference_ms = pending.result()
        pending = None
        validate_policy_chunk(
            actions,
            measured_arm=initial_context.measured_arm_rad,
            config=config,
            initial_delta_limit_rad=args.initial_delta_limit_rad,
            step_delta_limit_rad=args.step_delta_limit_rad,
            expected_horizon=40,
        )

        ensemble = PhysicalTargetTemporalEnsembler(
            decay_lambda=temporal_lambda
        )
        ensemble.add_chunk(origin_step=0, absolute_targets=actions)
        limiter = PolicyActionLimiter(
            np.asarray(history[-1].arm_joint_position_rad),
            np.asarray(history[-1].dex1_opening_fraction),
            command_hz=COMMAND_HZ,
            arm_velocity_rad_s=args.policy_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.policy_arm_acceleration_rad_s2,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=args.policy_hand_acceleration_fraction_s2,
        )
        append_log(
            args.log,
            {
                "event": "armed_prediction",
                "origin_step": 0,
                "inference_ms": inference_ms,
            },
        )

        pending_context: InferenceRequestContext | None = None
        next_replan = execution_steps
        last_limited = np.concatenate(
            (
                np.asarray(history[-1].arm_joint_position_rad, dtype=np.float64),
                DEX1_DATASET_OPEN_VALUE
                * np.asarray(
                    history[-1].dex1_opening_fraction,
                    dtype=np.float64,
                ),
            )
        )
        step = 0
        deadline = time.monotonic() + args.max_seconds
        period = 1.0 / COMMAND_HZ
        while time.monotonic() < deadline:
            tick = time.monotonic()
            observation = backend.observe(
                timeout_s=min(0.05, config.safety.command_hold_timeout_s)
            )
            validate_runtime_backend(observation)
            buffer.add(observation)

            if pending is not None and pending.done():
                new_actions, latency_ms = pending.result()
                assert pending_context is not None
                validate_policy_chunk(
                    new_actions,
                    measured_arm=pending_context.measured_arm_rad,
                    config=config,
                    initial_delta_limit_rad=args.initial_delta_limit_rad,
                    step_delta_limit_rad=args.step_delta_limit_rad,
                    expected_horizon=40,
                )
                usable = pending_context.has_remaining_target(step)
                if usable:
                    ensemble.add_chunk(
                        origin_step=pending_context.origin_step,
                        absolute_targets=new_actions,
                    )
                append_log(
                    args.log,
                    {
                        "event": "chunk" if usable else "stale_chunk_discarded",
                        "origin_step": pending_context.origin_step,
                        "received_step": step,
                        "inference_ms": latency_ms,
                    },
                )
                pending = None
                pending_context = None

            if pending_context is not None and (
                pending_context.age_seconds(time.monotonic())
                > MAX_INFERENCE_AGE_SECONDS
            ):
                raise TimeoutError(
                    "Furniture-GR00T inference exceeded the H40 validity window; "
                    "stale commands were not executed"
                )

            if pending is None and step >= next_replan:
                replan_history = buffer.pair()
                replan_state = state_for_observation(replan_history[-1], fk=fk)
                pending_context = InferenceRequestContext(
                    origin_step=step,
                    measured_arm_rad=np.asarray(
                        replan_history[-1].arm_joint_position_rad,
                        dtype=np.float64,
                    ),
                    submitted_monotonic_s=time.monotonic(),
                )
                pending = executor.submit(
                    worker.predict,
                    replan_history,
                    replan_state,
                )
                next_replan = step + execution_steps

            candidate_count = ensemble.candidate_count(step)
            if candidate_count:
                target = ensemble.target(step)
                holding_for_inference = False
            else:
                target = last_limited.copy()
                holding_for_inference = True
            limited = limiter.apply(target)
            last_limited = limited.copy()
            gravity_torque = gravity.torque_nm(limited[:14])
            backend.apply(
                command_from_action(
                    command_sequence.next(),
                    limited,
                    arm_feedforward_torque_nm=gravity_torque,
                )
            )
            append_log(
                args.log,
                {
                    "event": "command",
                    "step": step,
                    "temporal_candidate_count": candidate_count,
                    "holding_for_inference": holding_for_inference,
                    "desired_arm_target_rad": target[:14].tolist(),
                    "arm_target_rad": limited[:14].tolist(),
                    "dex1_target_fraction": (
                        limited[14:] / DEX1_DATASET_OPEN_VALUE
                    ).tolist(),
                    "lower_body_command_dimensions": 0,
                },
            )
            step += 1
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
        print(f"[done] executed {step} upper-body steps at {COMMAND_HZ:.0f} Hz", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[interrupt] controlled arm_sdk release", flush=True)
        return 130
    finally:
        if backend is not None:
            if actuation_started:
                if (
                    initial_arm_position is not None
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
                        dataset_frame0_arm_rad=(
                            FLIP_TABLE_V2_FRAME0.arm_position_rad
                        ),
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
        if worker is not None and pending is not None and not pending.done():
            worker.terminate()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if worker is not None:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
