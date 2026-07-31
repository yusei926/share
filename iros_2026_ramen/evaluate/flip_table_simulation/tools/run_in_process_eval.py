#!/usr/bin/env python3
"""Run one RoboFinals policy in the Isaac Sim process.

The organizer's generic evaluator sends every CUDA observation through a
``multiprocessing.Manager`` RPC. That is useful for isolating learned policies,
but copying five 640x480 camera tensors on every control step prevents the AVP
teleoperation loop from running in real time. This runner uses the organizer's
environment factory unchanged and evaluates only the selected policy in the
same process as Isaac Sim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import yaml


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--time-out-limit", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def _env_config(raw: dict):
    from robofinals.distributed.restful import DotDict

    value = DotDict(raw)
    defaults = {
        "scene_backend": "robocasa",
        "task_backend": "robocasa",
        "device": "cuda:0",
        "robot_scale": 1.0,
        "first_person_view": False,
        "disable_fabric": False,
        "num_envs": 1,
        "usd_simplify": False,
        "video": False,
        "for_rl": False,
        "variant": "Visual",
        "concatenate_terms": False,
        "distributed": False,
        "physics_backend": None,
        "seed": 42,
        "sources": None,
        "object_projects": None,
        "execute_mode": "eval",
        "replay_cfgs": {"add_camera_to_observation": True},
    }
    for key, default in defaults.items():
        if key not in value:
            value[key] = default
    return value


def _requested_render_interval(physics_hz: int, control_hz: int) -> int:
    value = os.environ.get("FLIP_TABLE_SIM_RENDER_INTERVAL")
    return physics_hz // control_hz if value in {None, ""} else int(value)


def _install_realtime_render_config(env_server) -> None:
    """Use Isaac Lab's supported performance preset for live stereo RGB."""

    original = env_server.make_env_cfg
    def make_env_cfg(config):
        # Read this at environment construction time rather than when the
        # wrapper is installed.  The persistent worker retains the Isaac app
        # but deliberately recreates its Gym environment when switching
        # between normal evaluation (200/50 Hz) and AVP collection
        # (100/50 Hz).  Capturing the old values here silently rebuilt a
        # 200 Hz environment while the job claimed to request 100 Hz.
        physics_hz = int(os.environ.get("FLIP_TABLE_SIM_PHYSICS_HZ", "200"))
        control_hz = 50
        render_interval = _requested_render_interval(physics_hz, control_hz)
        if physics_hz < control_hz or physics_hz % control_hz:
            raise ValueError("FLIP_TABLE_SIM_PHYSICS_HZ must be a multiple of 50 Hz")
        if render_interval < 1 or render_interval > physics_hz // 10:
            raise ValueError("FLIP_TABLE_SIM_RENDER_INTERVAL must preserve at least 10 Hz")
        task_name, env_cfg = original(config)
        env_cfg.sim.dt = 1.0 / physics_hz
        env_cfg.decimation = physics_hz // control_hz
        env_cfg.sim.render_interval = render_interval
        render = env_cfg.sim.render
        render.rendering_mode = "performance"
        render.antialiasing_mode = "DLSS"
        render.dlss_mode = 0
        render.enable_dlssg = False
        render.enable_dl_denoiser = False
        render.enable_translucency = False
        render.enable_reflections = False
        render.enable_global_illumination = False
        render.enable_direct_lighting = True
        render.samples_per_pixel = 1
        render.enable_shadows = True
        render.enable_ambient_occlusion = False
        return task_name, env_cfg

    env_server.make_env_cfg = make_env_cfg


def _enforce_realtime_render_settings() -> None:
    """Reassert settings changed by RoboFinals' generic optimize helper."""

    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx-transient/dlssg/enabled", False)
    settings.set_bool("/rtx-transient/dldenoiser/enabled", False)
    settings.set_bool("/rtx/translucency/enabled", False)
    settings.set_bool("/rtx/reflections/enabled", False)
    settings.set_bool("/rtx/indirectDiffuse/enabled", False)
    settings.set_bool("/rtx/directLighting/enabled", True)
    settings.set_int("/rtx/directLighting/sampledLighting/samplesPerPixel", 1)
    settings.set_bool("/rtx/shadows/enabled", True)
    settings.set_bool("/rtx/ambientOcclusion/enabled", False)
    settings.set_int("/rtx/post/dlss/execMode", 0)


def _remove_avp_camera_observation_terms(env) -> None:
    """Keep camera sensors alive without cloning every image into observations.

    Isaac Lab initializes camera render products while constructing the
    observation manager. Removing the terms after the first reset preserves
    those sensors, but avoids five GPU image clones on every 50 Hz control
    step. ``AvpTeleopPolicy`` reads only the cameras due for transmission
    directly from the sensor buffers.
    """

    manager = env.unwrapped.observation_manager
    camera_terms = {
        "first_person_camera_rgb",
        "head_right_camera_rgb",
        "left_hand_camera_rgb",
        "right_hand_camera_rgb",
        "global_camera_rgb",
    }
    required_terms = camera_terms - {"global_camera_rgb"}
    removed: set[str] = set()
    for group_name, names in manager._group_obs_term_names.items():
        keep_indices = [
            index for index, name in enumerate(names) if name not in camera_terms
        ]
        removed.update(name for name in names if name in camera_terms)
        if len(keep_indices) == len(names):
            continue
        manager._group_obs_term_names[group_name] = [names[index] for index in keep_indices]
        manager._group_obs_term_cfgs[group_name] = [
            manager._group_obs_term_cfgs[group_name][index] for index in keep_indices
        ]
        manager._group_obs_term_dim[group_name] = [
            manager._group_obs_term_dim[group_name][index] for index in keep_indices
        ]
        history = manager._group_obs_term_history_buffer[group_name]
        for term_name in camera_terms:
            history.pop(term_name, None)
        if manager._group_obs_concatenate[group_name]:
            raise RuntimeError(
                "AVP camera removal requires non-concatenated observation groups"
            )
        manager._group_obs_dim[group_name] = manager._group_obs_term_dim[group_name]
    missing = required_terms - removed
    if missing:
        raise RuntimeError(
            f"AVP camera observation terms are missing before optimization: {sorted(missing)}"
        )
    manager._obs_buffer = None
    print(
        "[flip_table] AVP direct camera path enabled; removed observation clones: "
        f"{sorted(removed)}",
        flush=True,
    )


def _verify_unique_upper_body_actuators(env) -> None:
    """Reject competing drives on any joint in the teleoperation contract."""

    robot = env.unwrapped.scene["robot"]
    joint_names = tuple(str(name) for name in robot.data.joint_names)
    actuator_cfgs = env.unwrapped.cfg.scene.robot.actuators
    upper_names = {
        name
        for name in joint_names
        if name.startswith("waist_")
        or "_shoulder_" in name
        or "_elbow_" in name
        or "_wrist_" in name
        or "_dex1_finger_" in name
    }
    owners = {name: [] for name in upper_names}
    for actuator_name, actuator_cfg in actuator_cfgs.items():
        patterns = getattr(actuator_cfg, "joint_names_expr", ())
        if isinstance(patterns, str):
            patterns = (patterns,)
        for joint_name in upper_names:
            if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
                owners[joint_name].append(str(actuator_name))
    if len(owners) != 21:
        raise RuntimeError(
            "the G1 + Dex1 upper-body contract must contain 21 joints; "
            f"found {len(owners)}: {sorted(owners)}"
        )
    invalid = {
        name: names for name, names in sorted(owners.items()) if len(names) != 1
    }
    if invalid:
        raise RuntimeError(
            f"upper-body joints must have exactly one actuator each: {invalid}"
        )
    print(
        "[flip_table] verified unique upper-body actuators: "
        + ", ".join(f"{name}={owners[name][0]}" for name in sorted(owners)),
        flush=True,
    )


def _verify_runtime_rates(env) -> None:
    """Prove the live environment retained the requested 100/50 Hz clock."""

    cfg = env.unwrapped.cfg
    physics_hz = 1.0 / float(cfg.sim.dt)
    control_hz = physics_hz / int(cfg.decimation)
    expected_physics_hz = float(os.environ.get("FLIP_TABLE_SIM_PHYSICS_HZ", "200"))
    expected_render_interval = _requested_render_interval(
        round(expected_physics_hz), 50
    )
    if not math.isclose(physics_hz, expected_physics_hz, abs_tol=1.0e-6):
        raise RuntimeError(
            f"simulator physics rate is {physics_hz:.3f} Hz, expected {expected_physics_hz:.3f} Hz"
        )
    if not math.isclose(control_hz, 50.0, abs_tol=1.0e-6):
        raise RuntimeError(f"simulator control rate is {control_hz:.3f} Hz, expected 50 Hz")
    if int(cfg.sim.render_interval) != expected_render_interval:
        raise RuntimeError(
            "simulator render interval is "
            f"{int(cfg.sim.render_interval)}, expected {expected_render_interval}"
        )
    print(
        "[flip_table] verified runtime rates: "
        f"physics_hz={physics_hz:.3f}, control_hz={control_hz:.3f}, "
        f"decimation={int(cfg.decimation)}, "
        f"render_interval={int(cfg.sim.render_interval)}, "
        f"render_hz={physics_hz / int(cfg.sim.render_interval):.3f}",
        flush=True,
    )


def main() -> int:
    args = _args()
    if args.test_num != 1:
        raise ValueError("in-process AVP teleoperation requires exactly one environment")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("env_cfg"), dict):
        raise ValueError("evaluation config must contain env_cfg")
    config["policy_name"] = args.policy_name
    config["test_num"] = args.test_num
    if args.time_out_limit is not None:
        config["time_out_limit"] = args.time_out_limit
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint

    # Importing env_server parses AppLauncher arguments. Keep its official
    # camera-enabled startup path while preventing this runner's arguments from
    # leaking into that parser.
    original_argv = sys.argv
    sys.argv = [
        "env_server.py",
        "--headless",
        "--enable_cameras",
        "--rendering_mode",
        "performance",
    ]
    try:
        from robofinals.scripts import env_server
    finally:
        sys.argv = original_argv

    env = None
    policy = None
    try:
        _install_realtime_render_config(env_server)
        env_config = _env_config(config["env_cfg"])
        env = env_server.make_env(env_config, env_server.app_launcher_args)
        # RoboFinals' factory reads the config seed but does not apply it to
        # the Gym environment.  Calibration candidates must differ only in
        # the parameter under test, never in an implicit PhysX/renderer seed.
        seed = int(config.get("seed", getattr(env_config, "seed", 42)))
        env.seed(seed)
        _verify_runtime_rates(env)
        _verify_unique_upper_body_actuators(env)
        _enforce_realtime_render_settings()

        import policy as policy_module

        policy_class = getattr(policy_module, args.policy_name)
        config["actions_dim"] = int(env.action_space.shape[-1])
        config["decimation"] = int(env.unwrapped.cfg.decimation)
        config["save_path"] = "./eval_result/test_0"
        policy = policy_class(config)
        observation, _ = env.reset(seed=seed)
        if args.policy_name in {"AvpTeleopPolicy", "Dex1ForceCalibrationPolicy"}:
            _remove_avp_camera_observation_terms(env)
        policy.reset_model()
        try:
            result = policy.eval(
                env,
                observation,
                config,
                Path("./eval_result/test_0/record_video.mp4"),
            )
        except BaseException:  # noqa: BLE001
            print("[flip_table] in-process policy failed:", flush=True)
            traceback.print_exc()
            raise

        import numpy as np
        import torch

        if torch.is_tensor(result):
            result = result.detach().cpu().numpy()
        success = bool(np.atleast_1d(np.asarray(result).astype(bool))[0])
        output = {
            "test_count": 1,
            "success_count": int(success),
            "success_rate": float(success),
        }
        Path("./eval_result").mkdir(parents=True, exist_ok=True)
        Path("./eval_result/eval_results.json").write_text(
            json.dumps(output, indent=4) + "\n", encoding="utf-8"
        )
        return 0
    finally:
        if policy is not None and getattr(policy, "_video_writer", None) is not None:
            policy._video_writer.__exit__(None, None, None)
            policy._video_writer = None
        if policy is not None and callable(getattr(policy, "close", None)):
            policy.close()
        if env is not None:
            env.close()
        if env_server.simulation_app is not None:
            env_server.simulation_app.close()
            env_server.simulation_app = None


if __name__ == "__main__":
    raise SystemExit(main())
