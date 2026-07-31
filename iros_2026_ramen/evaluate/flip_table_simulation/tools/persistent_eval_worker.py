#!/usr/bin/env python3
"""Keep one Isaac Sim process ready and execute explicit evaluation jobs.

The normal RoboFinals evaluator intentionally tears down Isaac after every
invocation. That is correct for isolated CI jobs, but wastes the multi-minute
Isaac cold-start while iterating on fixed-scene replay or CV diagnostics. This
worker owns one environment and processes file-backed jobs one at a time. A
job always starts with ``env.reset``; physics state is never shared between
jobs.

Only job fields which the reset/task and the selected diagnostic policies
already consume are accepted. The worker does not expose simulator state to a
policy and it refuses path traversal, unknown policy names, or arbitrary
environment-variable injection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import yaml

# The worker is executed by absolute path inside the pinned container. Keep its
# sibling runner import independent of the caller's current working directory.
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from run_in_process_eval import (
    _enforce_realtime_render_settings,
    _env_config,
    _install_realtime_render_config,
    _remove_avp_camera_observation_terms,
    _verify_runtime_rates,
    _verify_unique_upper_body_actuators,
)


SCHEMA_VERSION = "team_ramen_persistent_evaluation_job/v1"
READY_SCHEMA_VERSION = "team_ramen_persistent_evaluation_ready/v1"
SUPPORTED_POLICIES = {
    "RecordedJointTargetPolicy",
    "RecordedFullBodyTargetPolicy",
    "CvRuleBasedPolicy",
    "ScriptedJointPolicy",
    "AvpTeleopPolicy",
    "TeleopPerformanceBenchmarkPolicy",
}
# Dex1ForceCalibrationPolicy is intentionally absent: its static blockers are
# authored into the USD before Isaac starts, whereas this worker keeps one
# already-instantiated standard scene. Run that fixture-only diagnostic through
# the normal isolated runner so it cannot alter replay/contact evidence.
FOUNDATION_ENVIRONMENT_KEYS = (
    "FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE",
    "FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE",
    "FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE",
    "FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM",
)

# Values below are read during policy construction or the task's next reset.
# Keep this a positive allowlist: queue files are an IPC boundary, not shell
# scripts. Foundation settings such as the USD/image/config remain fixed for
# the life of one worker and therefore cannot silently drift between jobs.
ALLOWED_ENVIRONMENT_KEYS = {
    "FLIP_TABLE_EVAL_SEED",
    "FLIP_TABLE_REPLAY_ACTION_PATH",
    "FLIP_TABLE_REPLAY_HZ",
    "FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS",
    "FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ",
    "FLIP_TABLE_REPLAY_WARMUP_STEPS",
    "FLIP_TABLE_REPLAY_HOLD_INDEX",
    "FLIP_TABLE_INITIAL_UPPER_BODY_STATE",
    "FLIP_TABLE_INITIAL_FULL_BODY_STATE",
    # Offline calibration candidates are applied once by env.reset.  They are
    # parsed and range-checked by the task, never exposed to a policy, and are
    # cleared before the next queued job.
    "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON",
    "FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M",
    "FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION",
    "FLIP_TABLE_CAMERA_FRAME_INDEX",
    "FLIP_TABLE_CAMERA_FRAME_INDICES",
    "FLIP_TABLE_SAVE_CAMERA_FRAMES",
    "FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY",
    "FLIP_TABLE_SAVE_CAMERA_NAMES",
    "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT",
    "FLIP_TABLE_SAVE_CAMERA_ROLE_FILENAMES",
    "FLIP_TABLE_SAVE_ACTION_STATE_TRACE",
    "FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE",
    "FLIP_TABLE_TIME_OUT_LIMIT",
    "FLIP_TABLE_EVAL_MODE",
    "FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION",
    "FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE",
    "FLIP_TABLE_JOINT_NOISE_RAD",
    "FLIP_TABLE_DEX1_FINGER_NOISE_M",
    "FLIP_TABLE_TABLE_LONG_RANGE_M",
    "FLIP_TABLE_TABLE_DEPTH_RANGE_M",
    "FLIP_TABLE_TABLE_YAW_RANGE_RAD",
    "FLIP_TABLE_ROBOT_DISTANCE_RANGE_M",
    "FLIP_TABLE_ROBOT_LATERAL_RANGE_M",
    "FLIP_TABLE_ROBOT_YAW_RANGE_RAD",
    "FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS",
    "FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES",
    # Fixed equal endpoints are permitted for actuator identification. The
    # task applies these only during env.reset, so a queued job cannot mutate
    # another job's live drive parameters.
    "FLIP_TABLE_ARM_STIFFNESS_SCALE_RANGE",
    "FLIP_TABLE_ARM_DAMPING_SCALE_RANGE",
    "FLIP_TABLE_ARM_ARMATURE_SCALE_RANGE",
    "FLIP_TABLE_ARM_FRICTION_SCALE_RANGE",
    "FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS",
    "FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY",
    "FLIP_TABLE_EVAL_RANDOMIZE_MASS",
    "FLIP_TABLE_RANDOMIZE_LIGHTING",
    "FLIP_TABLE_RANDOMIZE_ROOM",
    "FLIP_TABLE_RANDOMIZE_ROOM_PROPS",
    "FLIP_TABLE_CV_WARMUP_STEPS",
    "FLIP_TABLE_CV_SETTLED_SELECTION_STEPS",
    # A zero-amplitude scripted run is a reset/contact-stability diagnostic.
    # The worker validates the small bounded range below.
    "FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE",
    # AVP jobs use the same persistent Isaac application but recreate only the
    # Gym environment with direct camera buffers.  These are bounded runtime
    # controls, not a route to change the action-manager contract.
    "FLIP_TABLE_TELEOP_PORT",
    "FLIP_TABLE_TELEOP_PERSISTENT",
    "FLIP_TABLE_TELEOP_PREVIEW_HZ",
    "FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ",
    "FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS",
    "FLIP_TABLE_REQUIRE_WAIST_LOCK",
    "FLIP_TABLE_SIM_BODY_MODE",
    "FLIP_TABLE_LOCK_LOWER_BODY",
    "FLIP_TABLE_LOCK_ROBOT_ROOT",
    "FLIP_TABLE_FIX_ROOT_LINK",
    "FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS",
    "FLIP_TABLE_RL_RANDOMIZATION_LEVEL",
    "FLIP_TABLE_SIM_PHYSICS_HZ",
    "FLIP_TABLE_SIM_RENDER_INTERVAL",
    # Recreate only the Gym/task environment for an offline calibration job.
    # SimulationApp remains alive, avoiding a cold Isaac startup while keeping
    # authored USD transforms and task caches isolated between candidates.
    "FLIP_TABLE_PERSISTENT_RECREATE_ENV",
}


def _environment_mode(policy_name: str) -> str:
    """Return the observation layout required by one queued policy.

    AVP reads camera tensors directly from Isaac sensors.  Its image terms
    must therefore be removed from the observation manager after reset to
    avoid cloning four camera tensors at every servo tick.  Other policies
    retain their normal observation layout.  Switching modes recreates the
    Gym environment, but deliberately keeps the expensive SimulationApp.
    """

    if policy_name == "AvpTeleopPolicy":
        return "avp_direct"
    # The action-manager layout is selected when the Gym environment is
    # constructed. Recreate only that environment (not SimulationApp) before
    # a 31-D calibration replay so it cannot accidentally reuse 16-D WBC.
    if policy_name == "RecordedFullBodyTargetPolicy":
        return "full_body_diagnostic"
    return "standard"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--poll-interval-s", type=float, default=0.25)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _ready_payload(*, state: str, job_id: str | None = None) -> dict[str, Any]:
    """Expose immutable worker foundation settings alongside its state."""

    result: dict[str, Any] = {
        "schema_version": READY_SCHEMA_VERSION,
        "state": state,
        "pid": os.getpid(),
        "started_unix_s": time.time(),
        "foundation_environment": {
            key: os.environ.get(key, "") for key in FOUNDATION_ENVIRONMENT_KEYS
        },
    }
    if state == "ready":
        result["supported_policies"] = sorted(SUPPORTED_POLICIES)
    if job_id is not None:
        result["job_id"] = job_id
    return result


def _lifecycle_payload(*, stage: str, detail: str | None = None) -> dict[str, Any]:
    """Record startup progress even when Isaac exits before becoming ready.

    The persistent launcher intentionally writes its normal stdout to one log
    file.  Isaac's asynchronous extension startup can make that log sparse
    when the application exits early, so this small atomic sidecar is the
    authoritative startup breadcrumb for remote diagnosis.  It contains no
    simulator state and is never consumed by a policy or evaluation job.
    """

    payload: dict[str, Any] = {
        "schema_version": "team_ramen_persistent_evaluation_lifecycle/v1",
        "pid": os.getpid(),
        "stage": stage,
        "updated_unix_s": time.time(),
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


def _lifecycle_stage(path: Path) -> str:
    """Return the last durable startup stage without trusting malformed JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return "unknown"
    stage = payload.get("stage") if isinstance(payload, dict) else None
    return stage if isinstance(stage, str) else "unknown"


def _relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must not escape the persistent job root")
    return path


def _load_job(path: Path, job_root: Path) -> tuple[dict[str, Any], dict[str, str], Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("job has an unsupported schema_version")
    job_id = raw.get("job_id")
    if not isinstance(job_id, str) or not job_id or any(
        char not in "-_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        for char in job_id
    ):
        raise ValueError("job_id must be a non-empty safe file name")
    policy_name = raw.get("policy_name")
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported persistent policy: {policy_name!r}")
    timeout_steps = raw.get("time_out_limit")
    if not isinstance(timeout_steps, int) or timeout_steps < 1:
        raise ValueError("time_out_limit must be a positive integer")
    seed = raw.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    output_relpath = _relative_path(raw.get("output_relpath"), field="output_relpath")
    output_dir = (job_root / output_relpath).resolve()
    if job_root not in output_dir.parents:
        raise ValueError("output_relpath escapes job root")

    environment = raw.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("environment must be an object")
    unknown = sorted(set(environment) - ALLOWED_ENVIRONMENT_KEYS)
    if unknown:
        raise ValueError(f"job environment contains unsupported keys: {unknown}")
    normalized: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"environment {key} must be scalar")
        rendered = str(value).lower() if isinstance(value, bool) else str(value)
        if key == "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON":
            if len(rendered) > 32_768:
                raise ValueError("calibration candidate JSON exceeds the persistent job limit")
            try:
                candidates = json.loads(rendered)
            except json.JSONDecodeError as exc:
                raise ValueError("calibration candidate must be valid JSON") from exc
            if not isinstance(candidates, list) or not candidates or len(candidates) > 64:
                raise ValueError("calibration candidate must be a non-empty list of at most 64 items")
        if key == "FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE":
            try:
                amplitude = float(rendered)
            except ValueError as exc:
                raise ValueError("FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE must be numeric") from exc
            if not np.isfinite(amplitude) or not 0.0 <= amplitude <= 0.25:
                raise ValueError(
                    "FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE must be finite and within [0.0, 0.25]"
                )
        if key == "FLIP_TABLE_PERSISTENT_RECREATE_ENV" and rendered not in {"true", "false"}:
            raise ValueError("FLIP_TABLE_PERSISTENT_RECREATE_ENV must be true or false")
        normalized[key] = rendered

    replay_path = normalized.get("FLIP_TABLE_REPLAY_ACTION_PATH")
    if replay_path:
        replay_relpath = _relative_path(replay_path, field="FLIP_TABLE_REPLAY_ACTION_PATH")
        replay_file = (job_root / replay_relpath).resolve()
        if job_root not in replay_file.parents or not replay_file.is_file():
            raise ValueError("FLIP_TABLE_REPLAY_ACTION_PATH must reference a queued input file")
        normalized["FLIP_TABLE_REPLAY_ACTION_PATH"] = str(replay_file)
    elif policy_name in {"RecordedJointTargetPolicy", "RecordedFullBodyTargetPolicy"}:
        raise ValueError(f"{policy_name} requires FLIP_TABLE_REPLAY_ACTION_PATH")
    return raw, normalized, output_dir


def _clear_job_environment() -> None:
    for key in ALLOWED_ENVIRONMENT_KEYS:
        os.environ.pop(key, None)


def _close_policy(policy: Any) -> None:
    video_writer = getattr(policy, "_video_writer", None)
    if video_writer is not None:
        video_writer.__exit__(None, None, None)
        policy._video_writer = None
    close = getattr(policy, "close", None)
    if callable(close):
        close()


def _restore_output_ownership(path: Path) -> None:
    """Hand generated evidence back to the host user after a root container run."""

    uid = os.environ.get("FLIP_TABLE_PERSISTENT_HOST_UID")
    gid = os.environ.get("FLIP_TABLE_PERSISTENT_HOST_GID")
    if uid is None or gid is None:
        return
    try:
        owner = int(uid)
        group = int(gid)
    except ValueError as exc:
        raise ValueError("persistent host UID/GID must be integers") from exc
    if owner < 0 or group < 0:
        raise ValueError("persistent host UID/GID must be non-negative")
    if not path.exists():
        return
    for candidate in (path, *path.rglob("*")):
        os.chown(candidate, owner, group)


def _create_environment(env_server: Any, base_config: dict[str, Any], mode: str) -> Any:
    """Build one reset-isolated Gym environment without restarting Isaac."""

    environment = env_server.make_env(
        _env_config(base_config["env_cfg"]), env_server.app_launcher_args
    )
    environment.seed(int(base_config.get("seed", 42)))
    _verify_runtime_rates(environment)
    _verify_unique_upper_body_actuators(environment)
    _enforce_realtime_render_settings()
    if mode == "avp_direct":
        _remove_avp_camera_observation_terms(environment)
    elif mode not in {"standard", "full_body_diagnostic"}:
        environment.close()
        raise ValueError(f"unsupported persistent environment mode: {mode!r}")
    return environment


def _run_job(
    env: Any,
    base_config: dict[str, Any],
    raw_job: dict[str, Any],
    environment: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    _clear_job_environment()
    os.environ.update(environment)
    policy_name = str(raw_job["policy_name"])
    seed = int(raw_job["seed"])
    timeout_steps = int(raw_job["time_out_limit"])
    # ``replay materialize`` has already created this directory and placed its
    # immutable action bundle there. Refuse only evidence from a prior worker
    # execution, which would otherwise make stale files look like this job.
    if (output_dir / "test_0").exists() or (output_dir / "eval_results.json").exists():
        raise FileExistsError(f"persistent job output already contains evaluation evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_output_dir = output_dir / "test_0"
    # ``BasePolicy`` opens its MP4 writer before it creates any other evidence.
    # The one-shot evaluator creates this directory as part of its outer loop;
    # the persistent worker owns that loop, so it must preserve this contract.
    episode_output_dir.mkdir(exist_ok=False)

    config = copy.deepcopy(base_config)
    config["policy_name"] = policy_name
    config["test_num"] = 1
    config["time_out_limit"] = timeout_steps
    config["actions_dim"] = int(env.action_space.shape[-1])
    config["decimation"] = int(env.unwrapped.cfg.decimation)
    config["save_path"] = str(episode_output_dir)

    import policy as policy_module

    policy = getattr(policy_module, policy_name)(config)
    try:
        observation, _ = env.reset(seed=seed)
        policy.reset_model()
        result = policy.eval(
            env,
            observation,
            config,
            episode_output_dir / "record_video.mp4",
        )
        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        success = bool(np.atleast_1d(np.asarray(result).astype(bool))[0])
        result_payload = {
            "schema_version": "team_ramen_persistent_evaluation_result/v1",
            "job_id": raw_job["job_id"],
            "policy_name": policy_name,
            "seed": seed,
            "success": success,
            "output_relpath": raw_job["output_relpath"],
        }
        _atomic_json(output_dir / "eval_results.json", result_payload)
        return result_payload
    finally:
        _close_policy(policy)
        _clear_job_environment()


def main() -> int:
    args = _args()
    if args.poll_interval_s <= 0.0:
        raise ValueError("--poll-interval-s must be positive")
    job_root = args.job_root.resolve()
    queue_dir = job_root / "persistent_jobs" / "queue"
    running_dir = job_root / "persistent_jobs" / "running"
    completed_dir = job_root / "persistent_jobs" / "completed"
    failed_dir = job_root / "persistent_jobs" / "failed"
    control_dir = job_root / "persistent_jobs"
    # Docker runs the worker as root, whereas the host-side submitter runs as
    # the workstation user. Only this file-backed IPC area needs to cross that
    # boundary; evaluation artifacts retain their normal permissions.
    control_dir.mkdir(parents=True, exist_ok=True)
    control_dir.chmod(0o777)
    for directory in (queue_dir, running_dir, completed_dir, failed_dir):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)
    stop_path = control_dir / "STOP"
    stop_path.unlink(missing_ok=True)
    lifecycle_path = control_dir / "last_lifecycle.json"
    _atomic_json(lifecycle_path, _lifecycle_payload(stage="directories_ready"))

    base_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(base_config, dict) or not isinstance(base_config.get("env_cfg"), dict):
        raise ValueError("evaluation config must contain env_cfg")

    _atomic_json(lifecycle_path, _lifecycle_payload(stage="importing_env_server"))
    original_argv = sys.argv
    sys.argv = ["env_server.py", "--headless", "--enable_cameras", "--rendering_mode", "performance"]
    try:
        from robofinals.scripts import env_server
    finally:
        sys.argv = original_argv

    env = None
    active_mode = "standard"
    try:
        try:
            _install_realtime_render_config(env_server)
            _atomic_json(lifecycle_path, _lifecycle_payload(stage="creating_standard_environment"))
            env = _create_environment(env_server, base_config, active_mode)
        except BaseException as exc:  # noqa: BLE001
            _atomic_json(
                lifecycle_path,
                _lifecycle_payload(
                    stage="standard_environment_initialization_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise
        _atomic_json(lifecycle_path, _lifecycle_payload(stage="standard_environment_ready"))
        _atomic_json(
            control_dir / "ready.json",
            _ready_payload(state="ready"),
        )
        print("[flip_table] persistent evaluation worker is ready", flush=True)
        while not stop_path.exists():
            jobs = sorted(queue_dir.glob("*.job.json"))
            if not jobs:
                time.sleep(args.poll_interval_s)
                continue
            queued_path = jobs[0]
            running_path = running_dir / queued_path.name
            try:
                queued_path.replace(running_path)
                raw_job, environment, output_dir = _load_job(running_path, job_root)
                requested_mode = _environment_mode(str(raw_job["policy_name"]))
                recreate_environment = (
                    requested_mode != active_mode
                    or environment.get("FLIP_TABLE_PERSISTENT_RECREATE_ENV") == "true"
                )
                if recreate_environment:
                    # Keep SimulationApp (and its cold-start cost) alive while
                    # rebuilding the Gym environment. This is required for
                    # offline calibration candidates: a task reset restores
                    # physics state, but not every authored USD xform/cache
                    # held by the previous task instance.
                    # Job environment variables must be installed before
                    # construction because the renderer clock is read by the
                    # factory hook.
                    _clear_job_environment()
                    os.environ.update(environment)
                    env.close()
                    env = _create_environment(env_server, base_config, requested_mode)
                    active_mode = requested_mode
                _atomic_json(
                    control_dir / "ready.json",
                    _ready_payload(state="running", job_id=str(raw_job["job_id"])),
                )
                result = _run_job(env, base_config, raw_job, environment, output_dir)
                _restore_output_ownership(output_dir)
                _atomic_json(completed_dir / running_path.name, result)
            except BaseException as exc:  # noqa: BLE001
                failure: dict[str, Any] = {
                    "schema_version": "team_ramen_persistent_evaluation_result/v1",
                    "state": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                try:
                    payload = json.loads(running_path.read_text(encoding="utf-8"))
                    failure["job_id"] = payload.get("job_id")
                    failed_output = payload.get("output_relpath")
                    if isinstance(failed_output, str):
                        _restore_output_ownership(job_root / _relative_path(
                            failed_output, field="output_relpath"
                        ))
                except Exception:  # noqa: BLE001
                    pass
                _atomic_json(failed_dir / running_path.name, failure)
                print("[flip_table] persistent job failed:\n" + failure["traceback"], flush=True)
            finally:
                running_path.unlink(missing_ok=True)
                _atomic_json(
                    control_dir / "ready.json",
                    _ready_payload(state="ready"),
                )
        _atomic_json(
            lifecycle_path,
            _lifecycle_payload(stage="stopping", detail="stop_file_observed"),
        )
        print("[flip_table] persistent evaluation worker stopping", flush=True)
        return 0
    finally:
        previous_stage = _lifecycle_stage(lifecycle_path)
        # Do not overwrite an initialization exception with a generic normal
        # shutdown marker.  That failure record is the only durable evidence
        # when Isaac raises SystemExit before its Python logger flushes.
        if not previous_stage.endswith("_failed"):
            closing_detail = (
                "stop_file_observed"
                if stop_path.exists()
                else f"environment_shutdown_after_{previous_stage}"
            )
            _atomic_json(
                lifecycle_path,
                _lifecycle_payload(stage="closing", detail=closing_detail),
            )
        if env is not None:
            env.close()
        if env_server.simulation_app is not None:
            env_server.simulation_app.close()
            env_server.simulation_app = None


if __name__ == "__main__":
    raise SystemExit(main())
