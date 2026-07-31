#!/usr/bin/env python3
"""Run the 16-D flip-table Diffusion policy on a physical G1.

The policy owns only the fourteen arm joints and two Dex1 opening targets.
Unitree Regular Mode retains the floating base, waist, and legs.  Without
``--actuate`` this command performs a complete live-camera/model preflight but
sends no robot command.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

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
from data.flip_table_data_augmentation.teleop.shared.policy_contract import (
    state_19d,
)
from inference.desktop.upper_policy.worker_protocol import (
    receive_message,
    send_message,
)
from inference.desktop.upper_policy.pre_motion import (
    ARM_PRE_MOTION_WAYPOINTS,
    ArmPreMotionWaypoint,
    build_arm_return_waypoints,
    build_arm_pre_motion_waypoints,
    validate_arm_pre_motion_waypoints,
)
from inference.desktop.upper_policy.subtask_start_pose import (
    SubtaskStartPose,
    subtask_start_pose_for_model,
)
from inference.desktop.upper_policy.gravity_compensation import (
    OfficialG1ArmGravityCompensator,
)


CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
CAMERA_ROLES = ("head_left", "left_wrist", "right_wrist")
# The model uses the source-dataset scalar convention 0=closed, 4.5=open.
# This is not the physical Dex1 motor range. RealDdsBackend converts the
# resulting 0..1 opening fraction to the official physical 0..5.4 rad range.
MODEL_DEX1_OPEN_VALUE = 4.5
# A z-score Diffusion head with clip_sample=false is intentionally unbounded.
# Permit only a small, explicitly diagnosed extrapolation here; the executable
# limiter below clamps it to the physical [0, 1] opening contract. Gross model
# excursions still fail closed before any physical command is enabled.
DEX1_PREFLIGHT_EXTRAPOLATION_FRACTION = 0.05
EXPECTED_CHECKPOINT_SHA256 = (
    "1a5786d38b9aad995aaf030b6c38ca8e20d2b15471c644e61f7d1c3a3258fd67"
)
DEFAULT_MODEL_REPO_ID = (
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_1"
)
DEFAULT_MODEL_REVISION = "3291d3743a25ec8a69570fd7f57599b71fe69a63"
DEFAULT_TASK = "flip table"


@dataclass
class CommandSequence:
    """Mutable monotonic sequence shared by normal and exceptional paths."""

    value: int = 0

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("command sequence must be non-negative")

    def next(self) -> int:
        self.value += 1
        return self.value


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
            / "model/subtask_policy_training/deployment/real_diffusion_worker.py"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=EXPECTED_CHECKPOINT_SHA256,
        help="Pinned model.safetensors SHA-256 from the model registry.",
    )
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument(
        "--pre-motion-only",
        action="store_true",
        help="Run the verified arm staging sequence, then release without policy.",
    )
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--preflight-samples", type=int, default=8)
    parser.add_argument("--initial-delta-limit-rad", type=float, default=0.20)
    parser.add_argument("--step-delta-limit-rad", type=float, default=0.20)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float, default=1.0)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float, default=4.0)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float, default=1.0)
    parser.add_argument(
        "--policy-hand-acceleration-fraction-s2", type=float, default=4.0
    )
    parser.add_argument(
        "--pre-motion-arm-velocity-rad-s",
        type=float,
        default=0.5,
        help="Arm-only staging velocity before policy inference.",
    )
    parser.add_argument(
        "--pre-motion-arm-acceleration-rad-s2",
        type=float,
        default=1.0,
        help="Arm-only staging acceleration before policy inference.",
    )
    parser.add_argument(
        "--pre-motion-waypoint-tolerance-rad",
        type=float,
        default=0.10,
        help="Maximum measured 14-joint error required before the next stage.",
    )
    parser.add_argument(
        "--pre-motion-stage-timeout-s",
        type=float,
        default=10.0,
        help="Fail-closed timeout for each arm-only staging waypoint.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "outputs/real_diffusion/last_run.jsonl",
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
        expected_checkpoint_sha256: str = EXPECTED_CHECKPOINT_SHA256,
        model_repo_id: str = DEFAULT_MODEL_REPO_ID,
        model_revision: str = DEFAULT_MODEL_REVISION,
        task: str = DEFAULT_TASK,
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
                "--expected-model-sha256",
                expected_checkpoint_sha256,
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
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._input = self._process.stdin
        self._output = self._process.stdout
        try:
            ready = receive_message(self._output)
            if ready.get("type") != "ready":
                raise RuntimeError(f"policy worker did not become ready: {ready}")
            contract = ready.get("contract", {})
            if contract.get("weights_sha256") != expected_checkpoint_sha256:
                raise RuntimeError(f"unexpected policy worker contract: {contract}")
            expected_identity = {
                "model_repo_id": model_repo_id,
                "model_revision": model_revision,
                "task": task,
            }
            if any(contract.get(key) != value for key, value in expected_identity.items()):
                raise RuntimeError(f"policy worker identity mismatch: {contract}")
            self.ready = ready
        except Exception:
            self._process.terminate()
            self._process.wait(timeout=3.0)
            raise

    def predict(self, history: list[TeleopObservation]) -> tuple[np.ndarray, float]:
        if len(history) != 2:
            raise ValueError("policy requires exactly two observations")
        self._request_id += 1
        send_message(
            self._input,
            {
                "type": "predict",
                "request_id": self._request_id,
                "state_history": [
                    state_19d(
                        observation.body_joint_position_rad,
                        observation.dex1_opening_fraction,
                    ).tolist()
                    for observation in history
                ],
                "camera_history": {
                    key: [
                        observation.camera_jpeg[role]
                        for observation in history
                    ]
                    for key, role in zip(CAMERA_KEYS, CAMERA_ROLES, strict=True)
                },
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


def camera_generation(observation: TeleopObservation) -> tuple[int, ...]:
    return tuple(
        int(observation.camera_stream_metadata[role]["jpeg_generation"])
        for role in CAMERA_ROLES
    )


def current_camera_skew_ms(observation: TeleopObservation) -> float:
    """Return the skew of the three JPEGs actually sent to the policy."""

    timestamps = np.asarray(
        [
            int(observation.camera_capture_monotonic_ns[role])
            for role in CAMERA_ROLES
        ],
        dtype=np.int64,
    )
    if timestamps.shape != (3,) or np.any(timestamps <= 0):
        raise ValueError("policy camera timestamps must contain three positive values")
    return float((timestamps.max() - timestamps.min()) / 1.0e6)


def is_fresh_policy_observation(
    observation: TeleopObservation,
    previous_generation: tuple[int, ...] | None,
    *,
    maximum_skew_ms: float,
) -> bool:
    """Validate the exact latest samples, not a historical backend bundle."""

    if observation.stale_roles:
        return False
    generation = camera_generation(observation)
    if previous_generation is not None and not all(
        current > previous
        for current, previous in zip(generation, previous_generation, strict=True)
    ):
        return False
    return current_camera_skew_ms(observation) <= maximum_skew_ms


def collect_policy_history(
    backend: Any,
    *,
    timeout_s: float = 5.0,
    hold: Callable[[TeleopObservation], None] | None = None,
) -> list[TeleopObservation]:
    history: deque[TeleopObservation] = deque(maxlen=2)
    previous: tuple[int, ...] | None = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        observation = backend.observe(timeout_s=min(1.0, timeout_s))
        validate_runtime_backend(observation)
        if hold is not None:
            hold(observation)
        if not is_fresh_policy_observation(
            observation,
            previous,
            maximum_skew_ms=1000.0 / 30.0,
        ):
            time.sleep(0.005)
            continue
        generation = camera_generation(observation)
        history.append(observation)
        previous = generation
        if len(history) == 2:
            return list(history)
        time.sleep(0.005)
    raise TimeoutError("two fresh policy camera observations were not available")


def validate_policy_chunk(
    actions: np.ndarray,
    *,
    measured_arm: np.ndarray,
    config: TeleopConfig,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    expected_horizon: int = 16,
    execution_steps: int | None = None,
    enforce_initial_delta: bool = True,
    enforce_step_delta: bool = True,
) -> dict[str, float]:
    values = np.asarray(actions, dtype=np.float64)
    measured = np.asarray(measured_arm, dtype=np.float64)
    if expected_horizon < 1:
        raise ValueError("expected_horizon must be positive")
    if execution_steps is None:
        execution_steps = expected_horizon
    if not 1 <= execution_steps <= expected_horizon:
        raise ValueError("execution_steps must be in [1, expected_horizon]")
    if values.shape != (expected_horizon, 16) or not np.isfinite(values).all():
        raise ValueError(
            f"policy action chunk must be finite [{expected_horizon},16], "
            f"got {values.shape}"
        )
    if measured.shape != (14,) or not np.isfinite(measured).all():
        raise ValueError("measured arm must be finite [14]")
    arms = values[:, :14]
    grippers_rad = values[:, 14:]
    lower = np.asarray(config.safety.arm_position_lower_rad)
    upper = np.asarray(config.safety.arm_position_upper_rad)
    if np.any(arms < lower) or np.any(arms > upper):
        joint, step = np.argwhere((arms < lower) | (arms > upper))[0][::-1]
        raise ValueError(
            "policy arm target exceeds configured hardware margin "
            f"(step={step}, joint={joint}, value={arms[step, joint]:.4f})"
        )
    dex1_margin = (
        DEX1_PREFLIGHT_EXTRAPOLATION_FRACTION * MODEL_DEX1_OPEN_VALUE
    )
    if np.any(grippers_rad < -dex1_margin) or np.any(
        grippers_rad > MODEL_DEX1_OPEN_VALUE + dex1_margin
    ):
        raise ValueError(
            "policy Dex1 target exceeds the supported clamp margin around "
            "the dataset scalar range [0,4.5] "
            f"(min={float(grippers_rad.min()):.6f}, "
            f"max={float(grippers_rad.max()):.6f}, "
            f"margin={dex1_margin:.6f})"
        )
    clamped_grippers_rad = np.clip(
        grippers_rad, 0.0, MODEL_DEX1_OPEN_VALUE
    )
    initial_delta = float(np.max(np.abs(arms[0] - measured)))
    full_step_deltas = np.abs(np.diff(arms, axis=0))
    full_step_delta = (
        float(np.max(full_step_deltas)) if len(full_step_deltas) else 0.0
    )
    executed_step_deltas = np.abs(np.diff(arms[:execution_steps], axis=0))
    if len(executed_step_deltas):
        flat_index = int(np.argmax(executed_step_deltas))
        step_index, joint_index = np.unravel_index(
            flat_index, executed_step_deltas.shape
        )
        step_delta = float(executed_step_deltas[step_index, joint_index])
    else:
        step_index = -1
        joint_index = -1
        step_delta = 0.0
    if enforce_initial_delta and initial_delta > initial_delta_limit_rad:
        raise ValueError(
            f"first policy arm target is {initial_delta:.4f} rad from measured pose; "
            f"limit is {initial_delta_limit_rad:.4f}"
        )
    if enforce_step_delta and step_delta > step_delta_limit_rad:
        raise ValueError(
            f"policy chunk contains a {step_delta:.4f} rad step; "
            f"limit is {step_delta_limit_rad:.4f} "
            f"(executed transition={step_index}->{step_index + 1}, "
            f"arm joint={joint_index}, execution_steps={execution_steps})"
        )
    return {
        "initial_arm_delta_max_rad": initial_delta,
        "chunk_step_delta_max_rad": step_delta,
        "full_chunk_step_delta_max_rad": full_step_delta,
        "validated_execution_steps": float(execution_steps),
        "arm_min_rad": float(arms.min()),
        "arm_max_rad": float(arms.max()),
        "dex1_min_fraction": float(
            (clamped_grippers_rad / MODEL_DEX1_OPEN_VALUE).min()
        ),
        "dex1_max_fraction": float(
            (clamped_grippers_rad / MODEL_DEX1_OPEN_VALUE).max()
        ),
        "dex1_raw_min_scalar": float(grippers_rad.min()),
        "dex1_raw_max_scalar": float(grippers_rad.max()),
        "dex1_clamp_max_scalar": float(
            np.max(np.abs(grippers_rad - clamped_grippers_rad))
        ),
    }


def validate_state_distribution(
    history: list[TeleopObservation],
    checkpoint: Path,
) -> dict[str, Any]:
    """Fail closed when the live 19-D input is outside training support."""

    states = np.stack(
        [
            state_19d(
                observation.body_joint_position_rad,
                observation.dex1_opening_fraction,
            )
            for observation in history
        ]
    )
    if states.shape != (2, 19) or not np.isfinite(states).all():
        raise ValueError(f"policy state history must be finite [2,19], got {states.shape}")
    stats = json.loads(
        (checkpoint / "normalization.json").read_text(encoding="utf-8")
    )["observation.state"]
    minimum = np.asarray(stats["min"], dtype=np.float64)
    maximum = np.asarray(stats["max"], dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if any(value.shape != (19,) for value in (minimum, maximum, mean, std)):
        raise ValueError("checkpoint observation.state statistics must be 19-D")
    # Raw train minima/maxima are not physical limits. Regular Mode naturally
    # moves the waist by a few milliradians even while standing, so retain an
    # explicit small support margin while still rejecting material OOD input.
    support_margin = np.maximum(0.05, 0.10 * (maximum - minimum))
    outside_training_range = (states < minimum) | (states > maximum)
    outside_supported_range = (states < minimum - support_margin) | (
        states > maximum + support_margin
    )
    z_score = np.abs((states - mean) / np.maximum(std, 1.0e-6))
    outside_supported_range |= z_score > 6.0
    if np.any(outside_supported_range):
        sample, dimension = np.argwhere(outside_supported_range)[0]
        raise ValueError(
            "live state is materially outside checkpoint support "
            f"(sample={sample}, dimension={dimension}, value={states[sample, dimension]:.5f}, "
            f"train_range=[{minimum[dimension]:.5f},{maximum[dimension]:.5f}], "
            f"margin={support_margin[dimension]:.5f}, z={z_score[sample, dimension]:.2f})"
        )
    lower_excursion = np.maximum(minimum - states, 0.0)
    upper_excursion = np.maximum(states - maximum, 0.0)
    return {
        "state_19d_latest": states[-1].tolist(),
        "state_max_abs_z": float(z_score.max()),
        "state_outside_training_value_count": int(outside_training_range.sum()),
        "state_training_range_excursion_max": float(
            np.maximum(lower_excursion, upper_excursion).max()
        ),
        "camera_payload_skew_ms": current_camera_skew_ms(history[-1]),
    }


def validate_runtime_backend(observation: TeleopObservation) -> None:
    """Abort autonomous TRACK before it can re-arm after an interlock."""

    diagnostics = observation.diagnostics
    if diagnostics.get("lower_body_policy_command_dimensions") != 0:
        raise RuntimeError("backend no longer reports zero lower-body command dimensions")
    if diagnostics.get("regular_mode_owns_lower_body") is not True:
        raise RuntimeError("backend no longer reports Regular Mode lower-body ownership")
    if observation.stale_roles:
        raise RuntimeError(f"policy camera stream became stale: {observation.stale_roles}")
    interlock_reason = diagnostics.get("arm_interlock_reason")
    if interlock_reason:
        raise RuntimeError(f"arm_sdk interlock latched: {interlock_reason}")
    dex1_error = diagnostics.get("dex1_thread_error")
    if dex1_error:
        raise RuntimeError(f"Dex1 publisher failed: {dex1_error}")
    failures = np.asarray(
        diagnostics.get("dds_write_failure_count_arm_left_right", ()),
        dtype=np.int64,
    )
    if failures.shape != (3,) or np.any(failures < 0):
        raise RuntimeError("backend DDS failure counters are malformed")
    if np.any(failures):
        raise RuntimeError(
            "DDS write failure detected "
            f"(arm,left_dex1,right_dex1={failures.tolist()})"
        )


class PolicyActionLimiter:
    """Stateful velocity/acceleration limiter for learned physical commands."""

    def __init__(
        self,
        arm_position_rad: np.ndarray,
        dex1_opening_fraction: np.ndarray,
        *,
        command_hz: float,
        arm_velocity_rad_s: float,
        arm_acceleration_rad_s2: float,
        hand_velocity_fraction_s: float,
        hand_acceleration_fraction_s2: float,
    ) -> None:
        if not math.isfinite(command_hz) or command_hz <= 0.0:
            raise ValueError("command_hz must be finite and positive")
        self.dt = 1.0 / command_hz
        self.arm = np.asarray(arm_position_rad, dtype=np.float64).copy()
        self.hand = np.asarray(dex1_opening_fraction, dtype=np.float64).copy()
        if (
            self.arm.shape != (14,)
            or self.hand.shape != (2,)
            or not np.isfinite(self.arm).all()
            or not np.isfinite(self.hand).all()
        ):
            raise ValueError("limiter initial state must be finite arms[14]+hands[2]")
        self.arm_velocity = np.zeros(14, dtype=np.float64)
        self.hand_velocity = np.zeros(2, dtype=np.float64)
        self.arm_velocity_limit = float(arm_velocity_rad_s)
        self.arm_acceleration_limit = float(arm_acceleration_rad_s2)
        self.hand_velocity_limit = float(hand_velocity_fraction_s)
        self.hand_acceleration_limit = float(hand_acceleration_fraction_s2)
        for value, label in (
            (self.arm_velocity_limit, "arm velocity"),
            (self.arm_acceleration_limit, "arm acceleration"),
            (self.hand_velocity_limit, "hand velocity"),
            (self.hand_acceleration_limit, "hand acceleration"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{label} limit must be finite and positive")

    @staticmethod
    def _rate_limit(
        desired: np.ndarray,
        previous: np.ndarray,
        previous_velocity: np.ndarray,
        *,
        velocity_limit: float,
        acceleration_limit: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        error = desired - previous
        acceleration_step = acceleration_limit * dt
        stopping_speed = np.maximum(
            0.0,
            -acceleration_step
            + np.sqrt(
                np.maximum(
                    0.0,
                    acceleration_step**2
                    + 2.0 * acceleration_limit * np.abs(error),
                )
            ),
        )
        desired_velocity = np.sign(error) * np.minimum(
            velocity_limit, stopping_speed
        )
        velocity = np.clip(
            previous_velocity
            + np.clip(
                desired_velocity - previous_velocity,
                -acceleration_step,
                acceleration_step,
            ),
            -velocity_limit,
            velocity_limit,
        )
        candidate = previous + velocity * dt
        crossed = np.sign(error) != np.sign(desired - candidate)
        landing_velocity = error / dt
        feasible = (np.abs(landing_velocity) <= velocity_limit + 1.0e-12) & (
            np.abs(landing_velocity - previous_velocity)
            <= acceleration_step + 1.0e-12
        )
        land = crossed & feasible
        return (
            np.where(land, desired, candidate),
            np.where(land, landing_velocity, velocity),
        )

    def apply(self, desired_action: np.ndarray) -> np.ndarray:
        desired = np.asarray(desired_action, dtype=np.float64)
        if desired.shape != (16,) or not np.isfinite(desired).all():
            raise ValueError("desired policy action must be finite [16]")
        arm, arm_velocity = self._rate_limit(
            desired[:14],
            self.arm,
            self.arm_velocity,
            velocity_limit=self.arm_velocity_limit,
            acceleration_limit=self.arm_acceleration_limit,
            dt=self.dt,
        )
        desired_hand = np.clip(
            desired[14:] / MODEL_DEX1_OPEN_VALUE, 0.0, 1.0
        )
        hand, hand_velocity = self._rate_limit(
            desired_hand,
            self.hand,
            self.hand_velocity,
            velocity_limit=self.hand_velocity_limit,
            acceleration_limit=self.hand_acceleration_limit,
            dt=self.dt,
        )
        self.arm = arm
        self.arm_velocity = arm_velocity
        self.hand = hand
        self.hand_velocity = hand_velocity
        return np.concatenate((arm, MODEL_DEX1_OPEN_VALUE * hand))


def command_from_action(
    sequence: int,
    action: np.ndarray,
    *,
    arm_feedforward_torque_nm: np.ndarray | None = None,
) -> ArmHandTarget:
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("executable policy action must be finite [16]")
    torque = np.zeros(14, dtype=np.float64)
    if arm_feedforward_torque_nm is not None:
        torque = np.asarray(arm_feedforward_torque_nm, dtype=np.float64)
    if torque.shape != (14,) or not np.isfinite(torque).all():
        raise ValueError("arm feed-forward torque must be finite [14]")
    return ArmHandTarget(
        sequence=sequence,
        monotonic_ns=time.monotonic_ns(),
        mode=ControlMode.TRACK,
        event=ControlEvent.NONE,
        arm_position_rad=tuple(values[:14]),
        dex1_opening_fraction=tuple(values[14:] / MODEL_DEX1_OPEN_VALUE),
        arm_feedforward_torque_nm=tuple(torque),
    )


class PolicyStartPoseHold:
    """Keep the verified arm-only ready pose alive across blocking checks."""

    def __init__(
        self,
        backend: Any,
        *,
        command_sequence: CommandSequence,
        gravity_compensator: OfficialG1ArmGravityCompensator,
        latest: TeleopObservation,
    ) -> None:
        self.backend = backend
        self.command_sequence = command_sequence
        self.gravity_compensator = gravity_compensator
        self.action = np.concatenate(
            (
                np.asarray(latest.arm_joint_position_rad, dtype=np.float64),
                MODEL_DEX1_OPEN_VALUE
                * np.asarray(latest.dex1_opening_fraction, dtype=np.float64),
            )
        )
        if self.action.shape != (16,) or not np.isfinite(self.action).all():
            raise ValueError("policy start hold target must be finite 16-D")

    def refresh(self, observation: TeleopObservation) -> None:
        validate_runtime_backend(observation)
        self.backend.apply(
            command_from_action(
                self.command_sequence.next(),
                self.action,
                arm_feedforward_torque_nm=self.gravity_compensator.torque_nm(
                    self.action[:14]
                ),
            )
        )


def run_blocking_check_with_pose_hold(
    operation: Callable[[], Any],
    *,
    backend: Any,
    config: TeleopConfig,
    pose_hold: PolicyStartPoseHold,
    latest: TeleopObservation,
) -> tuple[Any, TeleopObservation]:
    """Run a non-backend blocking check while refreshing the arm watchdog."""

    period = 1.0 / config.rates.command_hz
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-start-check") as pool:
        pending = pool.submit(operation)
        while not pending.done():
            tick = time.monotonic()
            latest = backend.observe(
                timeout_s=min(0.05, config.safety.command_hold_timeout_s)
            )
            pose_hold.refresh(latest)
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
        return pending.result(), latest


def run_arm_pre_motion(
    backend: Any,
    *,
    config: TeleopConfig,
    log_path: Path,
    command_sequence: CommandSequence,
    gravity_compensator: OfficialG1ArmGravityCompensator,
    arm_velocity_rad_s: float,
    arm_acceleration_rad_s2: float,
    waypoint_tolerance_rad: float,
    stage_timeout_s: float,
    stable_samples_required: int = 5,
    waypoints: tuple[ArmPreMotionWaypoint, ...] = ARM_PRE_MOTION_WAYPOINTS,
    phase: str = "pre_motion",
    hand_targets_by_waypoint: Mapping[str, Sequence[float]] | None = None,
    hand_tolerance_fraction: float = 0.05,
    hand_velocity_fraction_s: float = 1.0,
    hand_acceleration_fraction_s2: float = 4.0,
) -> TeleopObservation:
    """Run the arm-only clearance sequence and verify measured convergence.

    Dex1 is held at its measured opening unless a caller supplies an explicit
    per-waypoint opening fraction. The command schema has no waist or leg
    fields, so Unitree Regular Mode remains the sole lower-body controller. A
    waypoint is accepted only after measured arm and requested hand targets
    converge for consecutive observations.
    """

    if not isinstance(command_sequence, CommandSequence):
        raise TypeError("command_sequence must be a CommandSequence")
    if not hasattr(gravity_compensator, "torque_nm"):
        raise TypeError("gravity_compensator must provide torque_nm(arms[14])")
    for value, label in (
        (arm_velocity_rad_s, "pre-motion arm velocity"),
        (arm_acceleration_rad_s2, "pre-motion arm acceleration"),
        (waypoint_tolerance_rad, "pre-motion waypoint tolerance"),
        (hand_tolerance_fraction, "pre-motion hand tolerance"),
        (hand_velocity_fraction_s, "pre-motion hand velocity"),
        (hand_acceleration_fraction_s2, "pre-motion hand acceleration"),
        (stage_timeout_s, "pre-motion stage timeout"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    if stable_samples_required < 1:
        raise ValueError("stable_samples_required must be positive")
    if phase not in {"pre_motion", "return_motion"}:
        raise ValueError("unsupported arm waypoint phase")
    validate_arm_pre_motion_waypoints(
        config.safety.arm_position_lower_rad,
        config.safety.arm_position_upper_rad,
        waypoints,
    )
    targets = dict(hand_targets_by_waypoint or {})
    unknown_targets = set(targets) - {waypoint.name for waypoint in waypoints}
    if unknown_targets:
        raise ValueError(
            f"hand targets reference unknown waypoints: {sorted(unknown_targets)}"
        )
    for name, values in targets.items():
        target = np.asarray(values, dtype=np.float64)
        if (
            target.shape != (2,)
            or not np.isfinite(target).all()
            or np.any((target < 0.0) | (target > 1.0))
        ):
            raise ValueError(
                f"Dex1 target for waypoint {name!r} must be finite [2] in [0,1]"
            )

    latest = backend.observe(
        timeout_s=min(1.0, config.safety.command_hold_timeout_s)
    )
    validate_runtime_backend(latest)
    initial_arm = np.asarray(latest.arm_joint_position_rad, dtype=np.float64)
    held_hand = np.asarray(latest.dex1_opening_fraction, dtype=np.float64)
    limiter = PolicyActionLimiter(
        initial_arm,
        held_hand,
        command_hz=config.rates.command_hz,
        arm_velocity_rad_s=arm_velocity_rad_s,
        arm_acceleration_rad_s2=arm_acceleration_rad_s2,
        hand_velocity_fraction_s=hand_velocity_fraction_s,
        hand_acceleration_fraction_s2=hand_acceleration_fraction_s2,
    )
    command_period_s = 1.0 / config.rates.command_hz
    display_phase = "pre-motion" if phase == "pre_motion" else "return"
    for waypoint_index, waypoint in enumerate(waypoints, start=1):
        target_arm = waypoint.resolve(initial_arm)
        lower = np.asarray(config.safety.arm_position_lower_rad, dtype=np.float64)
        upper = np.asarray(config.safety.arm_position_upper_rad, dtype=np.float64)
        if np.any((target_arm < lower) | (target_arm > upper)):
            index = int(np.flatnonzero((target_arm < lower) | (target_arm > upper))[0])
            raise ValueError(
                f"resolved pre-motion waypoint {waypoint.name!r} joint {index} "
                "violates the configured hardware margin"
            )
        target_hand = np.asarray(
            targets.get(waypoint.name, held_hand), dtype=np.float64
        )
        desired = np.concatenate(
            (target_arm, MODEL_DEX1_OPEN_VALUE * target_hand)
        )
        stable_samples = 0
        stage_started = time.monotonic()
        print(
            f"[{display_phase} {waypoint_index}/{len(waypoints)}] "
            f"{waypoint.name} started",
            flush=True,
        )
        append_log(
            log_path,
            {
                "event": f"{phase}_stage_started",
                "stage": waypoint.name,
                "stage_index": waypoint_index,
                "target_arm_rad": target_arm.tolist(),
                "target_dex1_fraction": target_hand.tolist(),
                "preserved_initial_joint_indices": list(
                    waypoint.preserve_initial_joint_indices
                ),
                "lower_body_command_dimensions": 0,
            },
        )
        while True:
            tick = time.monotonic()
            if tick - stage_started >= stage_timeout_s:
                measured = np.asarray(
                    latest.arm_joint_position_rad, dtype=np.float64
                )
                error = float(np.max(np.abs(target_arm - measured)))
                measured_hand = np.asarray(
                    latest.dex1_opening_fraction, dtype=np.float64
                )
                hand_error = float(np.max(np.abs(target_hand - measured_hand)))
                raise TimeoutError(
                    f"pre-motion stage {waypoint.name!r} did not converge "
                    f"within {stage_timeout_s:.2f}s "
                    f"(max_arm_error={error:.4f}rad, "
                    f"max_dex1_error={hand_error:.4f})"
                )
            latest = backend.observe(
                timeout_s=min(0.05, config.safety.command_hold_timeout_s)
            )
            validate_runtime_backend(latest)
            measured = np.asarray(latest.arm_joint_position_rad, dtype=np.float64)
            error = float(np.max(np.abs(target_arm - measured)))
            measured_hand = np.asarray(
                latest.dex1_opening_fraction, dtype=np.float64
            )
            hand_error = float(np.max(np.abs(target_hand - measured_hand)))
            stable_samples = (
                stable_samples + 1
                if error <= waypoint_tolerance_rad
                and hand_error <= hand_tolerance_fraction
                else 0
            )
            sequence = command_sequence.next()
            limited = limiter.apply(desired)
            gravity_torque = gravity_compensator.torque_nm(limited[:14])
            backend.apply(
                command_from_action(
                    sequence,
                    limited,
                    arm_feedforward_torque_nm=gravity_torque,
                )
            )
            append_log(
                log_path,
                {
                    "event": f"{phase}_command",
                    "stage": waypoint.name,
                    "sequence": sequence,
                    "measured_arm_rad": measured.tolist(),
                    "arm_target_rad": limited[:14].tolist(),
                    "final_target_arm_rad": target_arm.tolist(),
                    "max_measured_error_rad": error,
                    "stable_sample_count": stable_samples,
                    "target_dex1_fraction": target_hand.tolist(),
                    "measured_dex1_fraction": measured_hand.tolist(),
                    "max_measured_dex1_error_fraction": hand_error,
                    "arm_feedforward_torque_nm": gravity_torque.tolist(),
                    "lower_body_command_dimensions": 0,
                },
            )
            if stable_samples >= stable_samples_required:
                elapsed_s = time.monotonic() - stage_started
                append_log(
                    log_path,
                    {
                        "event": f"{phase}_stage_complete",
                        "stage": waypoint.name,
                        "stage_index": waypoint_index,
                        "elapsed_s": elapsed_s,
                        "max_measured_error_rad": error,
                        "max_measured_dex1_error_fraction": hand_error,
                        "sequence": sequence,
                    },
                )
                print(
                    f"[{display_phase} {waypoint_index}/"
                    f"{len(waypoints)}] "
                    f"{waypoint.name} reached "
                    f"(error={error:.4f}rad, elapsed={elapsed_s:.2f}s)",
                    flush=True,
                )
                break
            time.sleep(max(0.0, command_period_s - (time.monotonic() - tick)))
    return latest


def return_arms_before_release(
    backend: Any,
    *,
    config: TeleopConfig,
    log_path: Path,
    command_sequence: CommandSequence,
    gravity_compensator: OfficialG1ArmGravityCompensator,
    initial_arm_position_rad: tuple[float, ...] | np.ndarray,
    dataset_frame0_arm_rad: tuple[float, ...] | np.ndarray,
    arm_velocity_rad_s: float,
    arm_acceleration_rad_s2: float,
    waypoint_tolerance_rad: float,
    stage_timeout_s: float,
    dex1_return_opening_fraction: Sequence[float] | None = None,
) -> bool:
    """Return through the reverse collision path before arm_sdk blend-out.

    A failure is reported to the caller but does not suppress the subsequent
    controlled weight-to-zero release; leaving a dead publisher owning the
    arms would be more dangerous than a clearly reported incomplete return.
    """

    waypoints = build_arm_return_waypoints(
        initial_arm_position_rad,
        dataset_frame0_arm_rad,
    )
    if dex1_return_opening_fraction is not None:
        # Open the hands completely while preserving the current policy arm
        # pose, before any reverse arm sweep begins.
        waypoints = (
            ArmPreMotionWaypoint(
                "return_hands_full_open_before_motion",
                (0.0,) * 14,
                tuple(range(14)),
            ),
        ) + waypoints
    hand_targets = (
        None
        if dex1_return_opening_fraction is None
        else {
            waypoint.name: tuple(dex1_return_opening_fraction)
            for waypoint in waypoints
        }
    )
    append_log(
        log_path,
        {
            "event": "return_motion_started",
            "stage_count": len(waypoints),
            "lower_body_command_dimensions": 0,
        },
    )
    try:
        run_arm_pre_motion(
            backend,
            config=config,
            log_path=log_path,
            command_sequence=command_sequence,
            gravity_compensator=gravity_compensator,
            arm_velocity_rad_s=arm_velocity_rad_s,
            arm_acceleration_rad_s2=arm_acceleration_rad_s2,
            waypoint_tolerance_rad=waypoint_tolerance_rad,
            stage_timeout_s=stage_timeout_s,
            waypoints=waypoints,
            phase="return_motion",
            hand_targets_by_waypoint=hand_targets,
        )
    except Exception as exc:  # noqa: BLE001
        append_log(
            log_path,
            {"event": "return_motion_failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        print(
            f"[shutdown] reverse arm return failed; proceeding to controlled "
            f"arm_sdk release: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return False
    append_log(log_path, {"event": "return_motion_complete"})
    print(
        "[shutdown] reverse arm path complete; handing arms back to Regular Mode",
        flush=True,
    )
    return True


def wait_for_policy_start_with_hold(
    backend: Any,
    *,
    config: TeleopConfig,
    log_path: Path,
    command_sequence: CommandSequence,
    gravity_compensator: OfficialG1ArmGravityCompensator,
    latest: TeleopObservation,
    enter_poll: Callable[[], str | None] | None = None,
    prompt: str = (
        "Arms are held in the dataset start pose. "
        "Press Enter to start policy inference, or Ctrl+C to stop: "
    ),
    invalid_prompt: str = (
        "Press Enter to start policy inference, or Ctrl+C to stop: "
    ),
    waiting_event: str = "policy_start_gate_waiting",
    confirmed_event: str = "policy_start_gate_confirmed",
) -> TeleopObservation:
    """Hold the staged pose and watchdog while waiting for an empty Enter."""

    pose_hold = PolicyStartPoseHold(
        backend,
        command_sequence=command_sequence,
        gravity_compensator=gravity_compensator,
        latest=latest,
    )

    def poll_stdin(stream: TextIO = sys.stdin) -> str | None:
        ready, _, _ = select.select((stream,), (), (), 0.0)
        if not ready:
            return None
        line = stream.readline()
        if line == "":
            raise EOFError("stdin closed while waiting to start policy")
        return line.rstrip("\r\n")

    poll = enter_poll or poll_stdin
    print(prompt, end="", flush=True)
    append_log(
        log_path,
        {
            "event": waiting_event,
            "sequence": command_sequence.value,
            "held_arm_target_rad": pose_hold.action[:14].tolist(),
            "held_dex1_fraction": (
                pose_hold.action[14:] / MODEL_DEX1_OPEN_VALUE
            ).tolist(),
            "lower_body_command_dimensions": 0,
        },
    )
    period = 1.0 / config.rates.command_hz
    while True:
        response = poll()
        if response == "":
            print("", flush=True)
            append_log(
                log_path,
                {
                    "event": confirmed_event,
                    "sequence": command_sequence.value,
                    "lower_body_command_dimensions": 0,
                },
            )
            return latest
        if response is not None:
            print(
                "\nAction not started: press Enter without typing any text.",
                flush=True,
            )
            print(invalid_prompt, end="", flush=True)
        tick = time.monotonic()
        latest = backend.observe(
            timeout_s=min(0.05, config.safety.command_hold_timeout_s)
        )
        pose_hold.refresh(latest)
        time.sleep(max(0.0, period - (time.monotonic() - tick)))


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"monotonic_ns": time.monotonic_ns(), **payload},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )


def verify_regular_mode(
    interface: str, *, arm_sdk_active: bool = False
) -> None:
    """Recheck the high-level FSM in an isolated read-only process.

    Startup remains strict ``(501, 0)``.  Once this process has acquired the
    official motion-mode ``rt/arm_sdk`` overlay, the same physical firmware
    reports ``(501, 1)``.  That post-acquisition check accepts only modes 0/1;
    the backend still independently verifies its anchored ``mode_machine``,
    zero lower-body policy dimensions, DDS health, and Regular ownership.
    """

    from inference.desktop.lower_policy.actuators.g1_control_lock import (
        G1_CONTROL_LOCK_PATH,
        current_g1_control_lock_fd,
    )

    lock_fd = current_g1_control_lock_fd()
    environment = os.environ.copy()
    environment["IROS_G1_CONTROL_LOCK_PATH"] = str(G1_CONTROL_LOCK_PATH)
    environment["IROS_G1_CONTROL_LOCK_FD"] = str(lock_fd)
    command = [
        sys.executable,
        str(REPO_ROOT / "inference/desktop/xr/check_g1_regular_mode.py"),
        "--interface",
        interface,
    ]
    if arm_sdk_active:
        command.extend(
            [
                "--allowed-fsm-mode",
                "0",
                "--allowed-fsm-mode",
                "1",
            ]
        )
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        pass_fds=(lock_fd,),
        check=True,
    )


def verify_regular_mode_after_release(
    interface: str,
    *,
    attempts: int = 10,
    retry_interval_s: float = 0.2,
) -> None:
    """Require the firmware to return to strict Regular mode after release."""

    if attempts < 1 or retry_interval_s < 0.0:
        raise ValueError("invalid post-release Regular verification settings")
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        try:
            verify_regular_mode(interface, arm_sdk_active=False)
            print("[shutdown] Regular Mode handoff verified (501,0)", flush=True)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_interval_s)
    raise RuntimeError(
        "arm_sdk reached weight=0 but G1 did not return to strict Regular "
        f"Mode after {attempts} checks"
    ) from last_error


def main() -> int:
    args = parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("--max-seconds must be positive")
    if args.preflight_samples < 1:
        raise ValueError("--preflight-samples must be positive")
    if args.pre_motion_only and not args.actuate:
        raise ValueError("--pre-motion-only requires --actuate")
    for value, label in (
        (args.initial_delta_limit_rad, "--initial-delta-limit-rad"),
        (args.step_delta_limit_rad, "--step-delta-limit-rad"),
        (args.policy_arm_velocity_rad_s, "--policy-arm-velocity-rad-s"),
        (args.policy_arm_acceleration_rad_s2, "--policy-arm-acceleration-rad-s2"),
        (
            args.policy_hand_velocity_fraction_s,
            "--policy-hand-velocity-fraction-s",
        ),
        (
            args.policy_hand_acceleration_fraction_s2,
            "--policy-hand-acceleration-fraction-s2",
        ),
        (
            args.pre_motion_arm_velocity_rad_s,
            "--pre-motion-arm-velocity-rad-s",
        ),
        (
            args.pre_motion_arm_acceleration_rad_s2,
            "--pre-motion-arm-acceleration-rad-s2",
        ),
        (
            args.pre_motion_waypoint_tolerance_rad,
            "--pre-motion-waypoint-tolerance-rad",
        ),
        (
            args.pre_motion_stage_timeout_s,
            "--pre-motion-stage-timeout-s",
        ),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and positive")
    config = load_teleop_config(args.config)
    from data.flip_table_data_augmentation.teleop.real.backend import RealDdsBackend

    worker: PolicyWorker | None = None
    backend: RealDdsBackend | None = None
    command_sequence = CommandSequence()
    actuation_started = False
    pre_motion_complete = False
    initial_arm_position: np.ndarray | None = None
    start_pose: SubtaskStartPose | None = None
    gravity_compensator: OfficialG1ArmGravityCompensator | None = None
    try:
        worker = PolicyWorker(
            args.worker_python,
            args.worker_script,
            args.checkpoint,
            device=args.device,
            seed=args.seed,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            model_repo_id=args.model_repo_id,
            model_revision=args.model_revision,
            task=args.task,
        )
        print(
            "[model] ready "
            f"sha256={worker.ready['contract']['weights_sha256']} "
            f"device={worker.ready['device']}",
            flush=True,
        )
        backend = RealDdsBackend(args.interface, args.image_server_ip, config)
        history = collect_policy_history(backend)
        state_diagnostics = validate_state_distribution(history, args.checkpoint)
        measured = np.asarray(history[-1].arm_joint_position_rad)
        preflight_actions: list[np.ndarray] = []
        preflight_inference_ms: list[float] = []
        preflight_diagnostics: list[dict[str, float]] = []
        for _ in range(args.preflight_samples):
            actions, inference_ms = worker.predict(history)
            preflight_actions.append(actions)
            preflight_inference_ms.append(inference_ms)
            preflight_diagnostics.append(
                validate_policy_chunk(
                    actions,
                    measured_arm=measured,
                    config=config,
                    initial_delta_limit_rad=args.initial_delta_limit_rad,
                    step_delta_limit_rad=args.step_delta_limit_rad,
                )
            )
        stacked_actions = np.stack(preflight_actions)
        diagnostics = {
            "initial_arm_delta_max_rad": max(
                item["initial_arm_delta_max_rad"] for item in preflight_diagnostics
            ),
            "chunk_step_delta_max_rad": max(
                item["chunk_step_delta_max_rad"] for item in preflight_diagnostics
            ),
            "arm_prediction_std_max_rad": float(
                stacked_actions[:, :, :14].std(axis=0).max()
            ),
            "dex1_prediction_std_max_fraction": float(
                (
                    stacked_actions[:, :, 14:] / MODEL_DEX1_OPEN_VALUE
                ).std(axis=0).max()
            ),
            "inference_ms_mean": float(np.mean(preflight_inference_ms)),
            "inference_ms_max": float(np.max(preflight_inference_ms)),
        }
        append_log(
            args.log,
            {
                "event": "preflight_prediction",
                "actuate_requested": args.actuate,
                "checkpoint_sha256": args.expected_checkpoint_sha256,
                "model_repo_id": args.model_repo_id,
                "model_revision": args.model_revision,
                "task": args.task,
                "preflight_samples": args.preflight_samples,
                **state_diagnostics,
                **diagnostics,
            },
        )
        print(
            "[preflight] live 3-camera inference OK; NO command sent "
            f"samples={args.preflight_samples} "
            f"inference_ms_mean={diagnostics['inference_ms_mean']:.2f} "
            f"initial_delta={diagnostics['initial_arm_delta_max_rad']:.4f}rad "
            f"step_delta={diagnostics['chunk_step_delta_max_rad']:.4f}rad "
            f"arm_prediction_std={diagnostics['arm_prediction_std_max_rad']:.4f}rad "
            f"dex1_prediction_std="
            f"{diagnostics['dex1_prediction_std_max_fraction']:.4f} "
            f"state_max_z={state_diagnostics['state_max_abs_z']:.2f} "
            f"camera_skew={state_diagnostics['camera_payload_skew_ms']:.2f}ms",
            flush=True,
        )
        if not args.actuate:
            print(
                f"[preflight-only] lower body remains owned by Regular Mode; log={args.log}",
                flush=True,
            )
            return 0
        confirmation = input(
            "Harness / E-stop / table clearance confirmed. Arms will move "
            "shoulders-back -> lateral-high -> forward-outside -> ready -> "
            "dataset-frame0"
            f"{', then release' if args.pre_motion_only else ', then policy'}. "
            "Press Enter to start arm-only pre-motion, or Ctrl+C to cancel: "
        )
        if confirmation != "":
            print(
                "[cancelled] Enter must be pressed without text; NO command sent",
                flush=True,
            )
            return 2

        # The operator may spend an arbitrary amount of time at the prompt.
        # Never actuate using the old FSM check, camera frames, robot state, or
        # stochastic policy sample.
        verify_regular_mode(args.interface)
        start_pose = subtask_start_pose_for_model(args.model_repo_id)
        latest = backend.observe(timeout_s=1.0)
        validate_runtime_backend(latest)
        initial_arm_position = np.asarray(
            latest.arm_joint_position_rad, dtype=np.float64
        ).copy()
        start_waypoints = build_arm_pre_motion_waypoints(
            start_pose.arm_position_rad
        )
        startup_hand_targets: dict[str, tuple[float, float]] | None = None
        if start_pose.dex1_opening_fraction is not None:
            # Open before moving the arms so the clearance path cannot drag a
            # closed Dex1 into the table. Keep both hands open until the final
            # dataset-frame0 stage, then adopt this model's pinned hand pose.
            start_waypoints = (
                ArmPreMotionWaypoint(
                    "hands_full_open_before_clearance",
                    (0.0,) * 14,
                    tuple(range(14)),
                ),
            ) + start_waypoints
            startup_hand_targets = {
                waypoint.name: (1.0, 1.0) for waypoint in start_waypoints
            }
            startup_hand_targets["dataset_frame0_pose"] = (
                start_pose.dex1_opening_fraction
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
                "dex1_opening_fraction": (
                    None
                    if start_pose.dex1_opening_fraction is None
                    else list(start_pose.dex1_opening_fraction)
                ),
                "lower_body_command_dimensions": 0,
            },
        )
        gravity_compensator = OfficialG1ArmGravityCompensator()
        print(
            "[pre-motion] official G1 RNEA gravity compensation ready "
            f"cache_sha256={gravity_compensator.cache_sha256}",
            flush=True,
        )
        # Set before the first staging command so every exception path requests
        # a controlled arm_sdk release, including a timeout mid-waypoint.
        actuation_started = True
        latest = run_arm_pre_motion(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=command_sequence,
            gravity_compensator=gravity_compensator,
            arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.pre_motion_arm_acceleration_rad_s2,
            waypoint_tolerance_rad=args.pre_motion_waypoint_tolerance_rad,
            stage_timeout_s=args.pre_motion_stage_timeout_s,
            waypoints=start_waypoints,
            hand_targets_by_waypoint=startup_hand_targets,
        )
        pre_motion_complete = True
        print(
            "[pre-motion] dataset frame-zero arm pose reached; "
            + (
                "releasing without policy"
                if args.pre_motion_only
                else "holding until Enter"
            ),
            flush=True,
        )
        if args.pre_motion_only:
            append_log(
                args.log,
                {
                    "event": "pre_motion_only_complete",
                    "sequence": command_sequence.value,
                    "lower_body_command_dimensions": 0,
                },
            )
            return 0
        latest = wait_for_policy_start_with_hold(
            backend,
            config=config,
            log_path=args.log,
            command_sequence=command_sequence,
            gravity_compensator=gravity_compensator,
            latest=latest,
        )
        pose_hold = PolicyStartPoseHold(
            backend,
            command_sequence=command_sequence,
            gravity_compensator=gravity_compensator,
            latest=latest,
        )
        # Recheck the high-level FSM after an arbitrarily long operator wait,
        # without allowing the 0.75 s arm watchdog to release in the meantime.
        _, latest = run_blocking_check_with_pose_hold(
            lambda: verify_regular_mode(args.interface, arm_sdk_active=True),
            backend=backend,
            config=config,
            pose_hold=pose_hold,
            latest=latest,
        )
        history = collect_policy_history(backend, hold=pose_hold.refresh)
        state_diagnostics = validate_state_distribution(history, args.checkpoint)
        (actions, inference_ms), latest = run_blocking_check_with_pose_hold(
            lambda: worker.predict(history),
            backend=backend,
            config=config,
            pose_hold=pose_hold,
            latest=history[-1],
        )
        diagnostics = validate_policy_chunk(
            actions,
            measured_arm=np.asarray(history[-1].arm_joint_position_rad),
            config=config,
            initial_delta_limit_rad=args.initial_delta_limit_rad,
            step_delta_limit_rad=args.step_delta_limit_rad,
        )
        append_log(
            args.log,
            {
                "event": "armed_prediction",
                "inference_ms": inference_ms,
                **state_diagnostics,
                **diagnostics,
            },
        )
        print(
            "[armed] fresh Regular/state/cameras/prediction verified "
            f"initial_delta={diagnostics['initial_arm_delta_max_rad']:.4f}rad "
            f"step_delta={diagnostics['chunk_step_delta_max_rad']:.4f}rad",
            flush=True,
        )

        started = time.monotonic()
        deadline = started + args.max_seconds
        command_period_s = 1.0 / config.rates.command_hz
        limiter = PolicyActionLimiter(
            np.asarray(history[-1].arm_joint_position_rad),
            np.asarray(history[-1].dex1_opening_fraction),
            command_hz=config.rates.command_hz,
            arm_velocity_rad_s=args.policy_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.policy_arm_acceleration_rad_s2,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=(
                args.policy_hand_acceleration_fraction_s2
            ),
        )
        while time.monotonic() < deadline:
            for action in actions[:8]:
                if time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                runtime_observation = backend.observe(
                    timeout_s=min(0.05, config.safety.command_hold_timeout_s)
                )
                validate_runtime_backend(runtime_observation)
                sequence = command_sequence.next()
                limited_action = limiter.apply(action)
                gravity_torque = gravity_compensator.torque_nm(
                    limited_action[:14]
                )
                backend.apply(
                    command_from_action(
                        sequence,
                        limited_action,
                        arm_feedforward_torque_nm=gravity_torque,
                    )
                )
                append_log(
                    args.log,
                    {
                        "event": "command",
                        "sequence": sequence,
                        "desired_arm_target_rad": action[:14].tolist(),
                        "arm_target_rad": limited_action[:14].tolist(),
                        "dex1_target_fraction": (
                            limited_action[14:] / MODEL_DEX1_OPEN_VALUE
                        ).tolist(),
                        "arm_velocity_rad_s": limiter.arm_velocity.tolist(),
                        "dex1_velocity_fraction_s": limiter.hand_velocity.tolist(),
                        "arm_feedforward_torque_nm": gravity_torque.tolist(),
                    },
                )
                time.sleep(max(0.0, command_period_s - (time.monotonic() - tick)))
            if time.monotonic() >= deadline:
                break
            # Never wait through the backend watchdog and then resume
            # automatically. A camera bundle that cannot refresh inside the
            # HOLD threshold aborts and triggers controlled arm_sdk release.
            history = collect_policy_history(
                backend,
                timeout_s=config.safety.command_hold_timeout_s * 0.75,
            )
            state_diagnostics = validate_state_distribution(history, args.checkpoint)
            actions, inference_ms = worker.predict(history)
            diagnostics = validate_policy_chunk(
                actions,
                measured_arm=np.asarray(history[-1].arm_joint_position_rad),
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
                    **diagnostics,
                },
            )
        print(f"[done] reached --max-seconds={args.max_seconds:.2f}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[interrupt] Ctrl+C received; controlled arm_sdk release", flush=True)
        return 130
    finally:
        if backend is not None:
            if actuation_started:
                if (
                    initial_arm_position is not None
                    and start_pose is not None
                    and gravity_compensator is not None
                    and pre_motion_complete
                ):
                    return_arms_before_release(
                        backend,
                        config=config,
                        log_path=args.log,
                        command_sequence=command_sequence,
                        gravity_compensator=gravity_compensator,
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
                        dex1_return_opening_fraction=(
                            (1.0, 1.0)
                            if start_pose.dex1_opening_fraction is not None
                            else None
                        ),
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
