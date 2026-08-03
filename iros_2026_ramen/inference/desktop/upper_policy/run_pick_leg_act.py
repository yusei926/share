#!/usr/bin/env python3
"""Evaluate the pinned joint16 ACT pick-leg policy on a physical G1.

The default path is read-only: it runs live four-camera inference but never
calls ``backend.apply``.  ``--actuate`` is an explicit second gate and still
requires the collision-aware arm-only pre-motion plus operator confirmation.
Waist and legs are never command dimensions; Unitree Regular Mode owns them.
"""

from __future__ import annotations

import argparse
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
    TeleopConfig,
    load_teleop_config,
)
from data.flip_table_data_augmentation.teleop.contracts import (  # noqa: E402
    ArmHandTarget,
    ControlEvent,
    ControlMode,
    TeleopObservation,
)
from inference.desktop.upper_policy.act_pick_leg_contract import (  # noqa: E402
    ACTION_MAX,
    ACTION_MIN,
    CAMERA_ROLES,
    DEX1_DATASET_OPEN_VALUE,
    MODEL_ACTION_HORIZON,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_STATE_DIM,
    TASK_TEXT,
    camera_payloads,
    compose_model_state,
)
from inference.desktop.upper_policy.gravity_compensation import (  # noqa: E402
    OfficialG1ArmGravityCompensator,
)
from inference.desktop.upper_policy.pre_motion import (  # noqa: E402
    ArmPreMotionWaypoint,
    build_arm_pre_motion_waypoints,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (  # noqa: E402
    CommandSequence,
    PolicyActionLimiter,
    PolicyStartPoseHold,
    append_log,
    command_from_action,
    initialize_policy_worker_with_live_camera,
    run_arm_pre_motion,
    run_blocking_check_with_pose_hold,
    start_policy_interval_recording,
    stop_policy_interval_recording,
    return_arms_before_release,
    validate_runtime_backend,
    verify_regular_mode,
    verify_regular_mode_after_release,
    wait_for_policy_start_with_hold,
)
from inference.desktop.upper_policy.run_pick_leg_groot import (  # noqa: E402
    _camera_generation,
    _camera_skew_ms,
    collect_fresh_observation,
)
from inference.desktop.upper_policy.subtask_start_pose import (  # noqa: E402
    SubtaskStartPose,
    subtask_start_pose_for_model,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)


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
            / "model/subtask_policy_training/deployment/real_act_joint16_worker.py"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_TELEOP_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=5.0)
    parser.add_argument("--preflight-samples", type=int, default=3)
    parser.add_argument(
        "--action-execution-steps",
        type=int,
        default=MODEL_ACTION_HORIZON,
        help=(
            "Number of decoded ACT steps to execute before replanning. The "
            "checkpoint was trained with n_action_steps=30, which is the "
            "fail-safe default; the sealed launcher passes its pinned value."
        ),
    )
    parser.add_argument("--initial-delta-limit-rad", type=float, default=0.50)
    parser.add_argument("--step-delta-limit-rad", type=float, default=0.25)
    parser.add_argument("--pre-motion-arm-velocity-rad-s", type=float, default=0.5)
    parser.add_argument("--pre-motion-arm-acceleration-rad-s2", type=float, default=1.0)
    parser.add_argument("--pre-motion-waypoint-tolerance-rad", type=float, default=0.10)
    parser.add_argument("--pre-motion-stage-timeout-s", type=float, default=15.0)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float, default=0.75)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float, default=3.0)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float, default=0.75)
    parser.add_argument("--policy-hand-acceleration-fraction-s2", type=float, default=3.0)
    parser.add_argument(
        "--log",
        type=Path,
        default=REPO_ROOT / "outputs/real_act_pick_leg/last_run.jsonl",
    )
    return parser.parse_args()


@dataclass
class PolicyWorker:
    process: subprocess.Popen[bytes]
    input: Any
    output: Any
    ready: dict[str, Any]
    task: str
    action_min: np.ndarray
    action_max: np.ndarray
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
        expected_hash: str | None,
        model_repo_id: str,
        model_revision: str,
        task: str,
    ) -> "PolicyWorker":
        argv = [
            str(python), str(script), "--checkpoint", str(checkpoint),
            "--device", device, "--seed", str(seed),
            "--model-repo-id", model_repo_id,
            "--model-revision", model_revision, "--task", task,
        ]
        if expected_hash:
            argv += ["--expected-model-sha256", expected_hash]
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            ready = receive_message(process.stdout)
            if ready.get("type") != "ready":
                raise RuntimeError(f"ACT worker did not become ready: {ready}")
            contract = ready.get("contract") or {}
            expected = {
                "model_repo_id": model_repo_id,
                "model_revision": model_revision,
                "task": task,
                "state_dim": MODEL_STATE_DIM,
                "decoded_action_dim": 16,
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
                raise RuntimeError(f"ACT worker contract mismatch: {mismatch}")
            action_min = np.asarray(contract.get("action_min"), dtype=np.float64)
            action_max = np.asarray(contract.get("action_max"), dtype=np.float64)
            if (
                action_min.shape != (16,)
                or action_max.shape != (16,)
                or not np.isfinite(action_min).all()
                or not np.isfinite(action_max).all()
                or np.any(action_min > action_max)
            ):
                raise RuntimeError("ACT worker returned invalid serialized support")
            return cls(
                process,
                process.stdin,
                process.stdout,
                ready,
                task,
                action_min,
                action_max,
            )
        except Exception:
            process.terminate()
            process.wait(timeout=10)
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
            raise RuntimeError(f"ACT worker failed: {response.get('error')}")
        if response.get("type") != "prediction" or response.get("request_id") != self.request_id:
            raise RuntimeError(f"unexpected ACT response: {response}")
        values = np.asarray(response["actions"], dtype=np.float64)
        expected = (MODEL_ACTION_HORIZON, 16)
        if values.shape != expected or not np.isfinite(values).all():
            raise RuntimeError(f"ACT worker returned invalid action {values.shape}")
        return np.clip(values, self.action_min, self.action_max), float(
            response["inference_ms"]
        )

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                send_message(self.input, {"type": "close"})
                receive_message(self.output)
            except (BrokenPipeError, EOFError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)


def validate_action_chunk(
    actions: np.ndarray,
    *,
    measured_arm: np.ndarray,
    config: TeleopConfig,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    enforce_initial_delta: bool,
    action_min: np.ndarray = ACTION_MIN,
    action_max: np.ndarray = ACTION_MAX,
) -> dict[str, float]:
    values = np.asarray(actions, dtype=np.float64)
    expected = (MODEL_ACTION_HORIZON, 16)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"ACT action must be finite {expected}, got {values.shape}")
    lower = np.asarray(config.safety.arm_position_lower_rad, dtype=np.float64)
    upper = np.asarray(config.safety.arm_position_upper_rad, dtype=np.float64)
    invalid = (values[:, :14] < lower) | (values[:, :14] > upper)
    if np.any(invalid):
        step, joint = np.argwhere(invalid)[0]
        raise ValueError(
            "ACT target exceeds configured arm limit "
            f"(step={step}, joint={joint}, value={values[step, joint]:.4f})"
        )
    initial = float(np.max(np.abs(values[0, :14] - measured_arm)))
    step = float(np.max(np.abs(np.diff(values[:, :14], axis=0))))
    if enforce_initial_delta and initial > initial_delta_limit_rad:
        raise ValueError(
            f"first ACT target is {initial:.4f} rad from measured pose; "
            f"limit is {initial_delta_limit_rad:.4f}"
        )
    if step > step_delta_limit_rad:
        raise ValueError(
            f"ACT chunk contains {step:.4f} rad step; limit is {step_delta_limit_rad:.4f}"
        )
    return {
        "initial_arm_delta_max_rad": initial,
        "chunk_step_delta_max_rad": step,
        "action_clamped_count": int(
            np.count_nonzero(
                (values <= np.asarray(action_min) + 1e-7)
                | (values >= np.asarray(action_max) - 1e-7)
            )
        ),
    }


def evaluate(
    worker: PolicyWorker,
    observation: TeleopObservation,
    *,
    config: TeleopConfig,
    samples: int,
    initial_delta_limit_rad: float,
    step_delta_limit_rad: float,
    enforce_initial_delta: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    predictions: list[np.ndarray] = []
    times: list[float] = []
    checks: list[dict[str, float]] = []
    for _ in range(samples):
        action, elapsed = worker.predict(observation)
        predictions.append(action)
        times.append(elapsed)
        checks.append(
            validate_action_chunk(
                action,
                measured_arm=np.asarray(observation.arm_joint_position_rad),
                config=config,
                initial_delta_limit_rad=initial_delta_limit_rad,
                step_delta_limit_rad=step_delta_limit_rad,
                enforce_initial_delta=enforce_initial_delta,
                action_min=worker.action_min,
                action_max=worker.action_max,
            )
        )
    return predictions[-1], {
        "inference_ms_mean": float(np.mean(times)),
        "inference_ms_max": float(np.max(times)),
        "initial_arm_delta_max_rad": max(v["initial_arm_delta_max_rad"] for v in checks),
        "chunk_step_delta_max_rad": max(v["chunk_step_delta_max_rad"] for v in checks),
        "arm_prediction_std_max_rad": float(np.stack(predictions)[:, :, :14].std(axis=0).max()),
        "action_clamped_count": max(v["action_clamped_count"] for v in checks),
    }


def build_act_start_motion(
    start_pose: SubtaskStartPose,
) -> tuple[tuple[ArmPreMotionWaypoint, ...], dict[str, tuple[float, float]]]:
    """Build collision-clearance motion ending at this model's frame-zero pose."""

    if start_pose.dex1_opening_fraction is None:
        raise ValueError(
            "ACT start pose is missing its dataset frame-zero Dex1 opening"
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


def log_prediction_chunk(
    path: Path,
    *,
    actions: np.ndarray,
    observation: TeleopObservation,
    diagnostics: dict[str, float],
    execution_steps: int,
) -> None:
    """Persist native physical ACT output before limiting or unit conversion."""

    values = np.asarray(actions, dtype=np.float64)
    append_log(
        path,
        {
            "event": "prediction_chunk",
            "action_units": "arms14_rad+dex1_physical_0_to_4.5",
            "raw_action_chunk": values.tolist(),
            "raw_action_first": values[0].tolist(),
            "raw_action_last": values[-1].tolist(),
            "raw_action_min": values.min(axis=0).tolist(),
            "raw_action_max": values.max(axis=0).tolist(),
            "model_state16": compose_model_state(
                observation.body_joint_position_rad,
                observation.dex1_opening_fraction,
            ).tolist(),
            "execution_steps": int(execution_steps),
            **diagnostics,
        },
    )


def limit_native_act_action(
    limiter: PolicyActionLimiter, desired: np.ndarray
) -> np.ndarray:
    """Limit one native ACT target without applying a second Dex1 scaling."""

    values = np.asarray(desired, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("native ACT target must be finite [16]")
    return limiter.apply(values)


def _validate_args(args: argparse.Namespace) -> None:
    if args.preflight_samples < 1:
        raise ValueError("--preflight-samples must be positive")
    if not 1 <= args.action_execution_steps <= MODEL_ACTION_HORIZON:
        raise ValueError("--action-execution-steps must be in [1,30]")
    for name in (
        "max_seconds", "initial_delta_limit_rad", "step_delta_limit_rad",
        "pre_motion_arm_velocity_rad_s", "pre_motion_arm_acceleration_rad_s2",
        "pre_motion_waypoint_tolerance_rad", "pre_motion_stage_timeout_s",
        "policy_arm_velocity_rad_s", "policy_arm_acceleration_rad_s2",
        "policy_hand_velocity_fraction_s", "policy_hand_acceleration_fraction_s2",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")


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
    initial_arm: np.ndarray | None = None
    start_pose: SubtaskStartPose | None = None
    gravity: OfficialG1ArmGravityCompensator | None = None
    try:
        backend = RealDdsBackend(args.interface, args.image_server_ip, config)
        worker = initialize_policy_worker_with_live_camera(
            backend,
            lambda: PolicyWorker.start(
                args.worker_python, args.worker_script, args.checkpoint,
                device=args.device, seed=args.seed,
                expected_hash=args.expected_checkpoint_sha256,
                model_repo_id=args.model_repo_id,
                model_revision=args.model_revision, task=args.task,
            ),
        )
        print(
            f"[model] ACT joint16 ready revision={args.model_revision[:12]} "
            f"device={worker.ready['device']}",
            flush=True,
        )
        observation = collect_fresh_observation(backend)
        if not args.actuate:
            _actions, diagnostics = evaluate(
                worker, observation, config=config, samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                enforce_initial_delta=False,
            )
            append_log(
                args.log,
                {
                    "event": "preflight_prediction",
                    "policy": "act_joint16_pick_table_leg",
                    "actuate_requested": False,
                    "model_repo_id": args.model_repo_id,
                    "model_revision": args.model_revision,
                    "camera_mapping": dict(zip(("cam_0", "cam_1", "cam_2", "cam_3"), CAMERA_ROLES, strict=True)),
                    "model_state_dim": 16,
                    "model_action_dim": 16,
                    "lower_body_command_dimensions": 0,
                    "camera_payload_skew_ms": _camera_skew_ms(observation),
                    **diagnostics,
                },
            )
            print(
                "[preflight] live 4-camera ACT inference OK; NO command sent "
                f"samples={args.preflight_samples} "
                f"inference_ms_mean={diagnostics['inference_ms_mean']:.1f} "
                f"raw_initial_delta={diagnostics['initial_arm_delta_max_rad']:.4f}rad "
                f"step_delta={diagnostics['chunk_step_delta_max_rad']:.4f}rad "
                f"camera_skew={_camera_skew_ms(observation):.2f}ms",
                flush=True,
            )
            print(
                f"[preflight-only] arms/hand/lower body untouched; log={args.log}",
                flush=True,
            )
            return 0

        confirmation = input(
            "Harness / E-stop / table clearance confirmed. Arms will move "
            "shoulders-back -> lateral-high -> forward-outside -> ready -> "
            "dataset-frame0. Press Enter to start arm-only pre-motion, or Ctrl+C: "
        )
        if confirmation != "":
            print("[cancelled] Enter must be pressed without text; NO command sent")
            return 2
        verify_regular_mode(args.interface)
        start_pose = subtask_start_pose_for_model(args.model_repo_id)
        current = backend.observe(timeout_s=1.0)
        validate_runtime_backend(current)
        initial_arm = np.asarray(current.arm_joint_position_rad).copy()
        gravity = OfficialG1ArmGravityCompensator()
        actuation_started = True
        start_waypoints, startup_hand_targets = build_act_start_motion(start_pose)
        current = run_arm_pre_motion(
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
        )
        pre_motion_complete = True
        current = wait_for_policy_start_with_hold(
            backend, config=config, log_path=args.log,
            command_sequence=sequence, gravity_compensator=gravity, latest=current,
        )
        hold = PolicyStartPoseHold(
            backend, command_sequence=sequence,
            gravity_compensator=gravity, latest=current,
        )
        _, current = run_blocking_check_with_pose_hold(
            lambda: verify_regular_mode(args.interface, arm_sdk_active=True),
            backend=backend, config=config, pose_hold=hold, latest=current,
        )
        start_policy_interval_recording(backend, args.log)
        previous = _camera_generation(current)
        observation = collect_fresh_observation(
            backend, previous_generation=previous, hold=hold.refresh
        )
        (actions, diagnostics), current = run_blocking_check_with_pose_hold(
            lambda: evaluate(
                worker, observation, config=config, samples=args.preflight_samples,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                enforce_initial_delta=True,
            ),
            backend=backend, config=config, pose_hold=hold, latest=observation,
        )
        log_prediction_chunk(
            args.log,
            actions=actions,
            observation=observation,
            diagnostics=diagnostics,
            execution_steps=args.action_execution_steps,
        )
        limiter = PolicyActionLimiter(
            np.asarray(current.arm_joint_position_rad),
            np.asarray(current.dex1_opening_fraction),
            command_hz=config.rates.command_hz,
            arm_velocity_rad_s=args.policy_arm_velocity_rad_s,
            arm_acceleration_rad_s2=args.policy_arm_acceleration_rad_s2,
            hand_velocity_fraction_s=args.policy_hand_velocity_fraction_s,
            hand_acceleration_fraction_s2=args.policy_hand_acceleration_fraction_s2,
            arm_position_lower_rad=config.safety.arm_position_lower_rad,
            arm_position_upper_rad=config.safety.arm_position_upper_rad,
        )
        print(
            "[armed] fresh Regular/state/4-camera ACT prediction verified; "
            "sending arms14+Dex1 only "
            f"(raw_target_delta={diagnostics['initial_arm_delta_max_rad']:.4f}rad)",
            flush=True,
        )
        deadline = time.monotonic() + args.max_seconds
        period = 1.0 / config.rates.command_hz
        while time.monotonic() < deadline:
            for desired in actions[: args.action_execution_steps]:
                if time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                live = backend.observe(timeout_s=min(0.05, config.safety.command_hold_timeout_s))
                validate_runtime_backend(live)
                # Worker output and limiter input are both in the native
                # physical contract: arms14 radians + Dex1 [0,4.5].  The
                # limiter converts Dex1 to fraction internally exactly once;
                # command_from_action performs the one hardware-boundary
                # conversion on its physical result.
                limited = limit_native_act_action(limiter, desired)
                torque = gravity.torque_nm(limited[:14])
                command_id = sequence.next()
                backend.apply(
                    command_from_action(
                        command_id, limited, arm_feedforward_torque_nm=torque
                    )
                )
                append_log(
                    args.log,
                    {
                        "event": "command",
                        "sequence": command_id,
                        "arm_target_rad": limited[:14].tolist(),
                        "dex1_target_physical": limited[14:].tolist(),
                        "dex1_target_fraction": (
                            limited[14:] / DEX1_DATASET_OPEN_VALUE
                        ).tolist(),
                        "lower_body_command_dimensions": 0,
                    },
                )
                time.sleep(max(0.0, period - (time.monotonic() - tick)))
            if time.monotonic() >= deadline:
                break
            previous = _camera_generation(observation)
            observation = collect_fresh_observation(
                backend,
                previous_generation=previous,
                timeout_s=config.safety.command_hold_timeout_s * 0.75,
            )
            actions, diagnostics = evaluate(
                worker, observation, config=config, samples=1,
                initial_delta_limit_rad=args.initial_delta_limit_rad,
                step_delta_limit_rad=args.step_delta_limit_rad,
                enforce_initial_delta=True,
            )
            log_prediction_chunk(
                args.log,
                actions=actions,
                observation=observation,
                diagnostics=diagnostics,
                execution_steps=args.action_execution_steps,
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
                if initial_arm is not None and start_pose is not None and gravity is not None and pre_motion_complete:
                    return_arms_before_release(
                        backend,
                        config=config,
                        log_path=args.log,
                        command_sequence=sequence,
                        gravity_compensator=gravity,
                        initial_arm_position_rad=initial_arm,
                        dataset_frame0_arm_rad=start_pose.arm_position_rad,
                        arm_velocity_rad_s=args.pre_motion_arm_velocity_rad_s,
                        arm_acceleration_rad_s2=args.pre_motion_arm_acceleration_rad_s2,
                        waypoint_tolerance_rad=args.pre_motion_waypoint_tolerance_rad,
                        stage_timeout_s=args.pre_motion_stage_timeout_s,
                        dex1_return_opening_fraction=(1.0, 1.0),
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
