from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load_overlay_function(name: str):
    source = (ROOT / "container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py").read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<overlay-function>", "exec"), namespace)
    return namespace[name]


def _load_module(path: Path, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_policy_module(monkeypatch, name: str) -> types.ModuleType:
    monkeypatch.setitem(
        sys.modules,
        "mediapy",
        types.SimpleNamespace(write_image=lambda *args, **kwargs: None),
    )
    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    return _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        name,
    )


def test_wbc_continuity_patch_is_real_encoder_compatible() -> None:
    source = (
        ROOT
        / "container_overlay"
        / "patches"
        / "patch_g1_wbc_action_continuity.py"
    ).read_text()

    assert "self._asset.data.joint_pos" in source
    assert "FLIP_TABLE_WBC_MAX_JOINT_SPEED_RAD_S" in source
    assert "FLIP_TABLE_WBC_MAX_JOINT_ACCELERATION_RAD_S2" in source
    assert "stopping_speed" in source
    assert "contact" not in source.lower()
    assert "object_pose" not in source


def test_eval_config_targets_local_assemble_table_overlay() -> None:
    config = yaml.safe_load((ROOT / "configs" / "flip_table_eval.yml").read_text())

    assert config["test_num"] == 10
    assert config["record_video"] is True
    assert config["record_camera"] == [
        "first_person_camera",
        "head_right_camera",
        "left_hand_camera",
        "right_hand_camera",
        "global_camera",
    ]
    assert config["policy_name"] == "LeRobotGrootN17Policy"
    assert config["checkpoint"] == "/workspace/flip_table_policy_checkpoint"
    assert config["seed"] == config["env_cfg"]["seed"] == 42
    assert config["observation_config"]["custom_mapping"] == {
        "observation.images.head_left": "first_person_camera",
        "observation.images.left_wrist": "left_hand_camera",
        "observation.images.right_wrist": "right_hand_camera",
        "observation.state": "joint_pos",
    }
    assert config["instruction"] == "flip table"
    assert config["env_cfg"]["task"] == "AssembleTableTask"
    assert config["env_cfg"]["robot"] == "G1-Gripper-Controller-DecoupledWBC"
    assert config["env_cfg"]["scene_backend"] == "local"
    assert config["env_cfg"]["task_backend"] == "local"
    assert config["env_cfg"]["layout"].endswith("IROS_IKEA_V13_20260702/Scene02.usd")
    assert config["env_cfg"]["num_envs"] == 1
    assert config["env_cfg"]["disable_fabric"] is False
    assert config["height"] == 480
    assert config["width"] == 640


def test_avp_teleop_uses_v1_action_contract_when_cli_omits_action_dim(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_avp_action_contract")

    assert module._teleop_action_dim({}) == 16
    assert module._teleop_action_dim({"actions_dim": 16}) == 16
    with pytest.raises(ValueError, match="requires the 16-D"):
        module._teleop_action_dim({"actions_dim": 19})
    with pytest.raises(ValueError, match="requires the 16-D"):
        module._teleop_action_dim({"actions_dim": 1})


def test_wrist_camera_distribution_tool_ranks_sim_candidates(tmp_path: Path) -> None:
    tool = _load_module(
        ROOT / "tools" / "compare_wrist_camera_distribution.py",
        "compare_wrist_camera_distribution_for_test",
    )

    def write_blocks(path: Path, rows: tuple[tuple[int, int, int], ...]) -> None:
        image = Image.new("RGB", (90, 90))
        for row, color in enumerate(rows):
            block = Image.new("RGB", (90, 30), color)
            image.paste(block, (0, row * 30))
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)

    real_dir = tmp_path / "real"
    good_dir = tmp_path / "good" / "test_0" / "camera_frames" / "frame_0010"
    bad_dir = tmp_path / "bad" / "test_0" / "camera_frames" / "frame_0010"
    for role in ("left_wrist", "right_wrist"):
        write_blocks(real_dir / f"{role}_rgb.png", ((245, 245, 245), (8, 8, 8), (8, 8, 8)))
        write_blocks(real_dir / f"{role}_real_f0030.png", ((245, 245, 245), (245, 245, 245), (8, 8, 8)))
        write_blocks(good_dir / f"{role}_rgb.png", ((240, 240, 240), (10, 10, 10), (10, 10, 10)))
        write_blocks(bad_dir / f"{role}_rgb.png", ((120, 170, 180), (240, 240, 240), (240, 240, 240)))

    report = tool.evaluate(real_dir, [bad_dir.parents[2], good_dir.parents[2]])

    assert report["best_frame_dir"] == str(good_dir)
    assert report["rank_by"] == "nearest"
    assert report["candidates"][0]["mean_score"] < report["candidates"][1]["mean_score"]
    assert report["candidates"][0]["nearest_real_label"] == "real"
    assert len(report["real_samples"]) == 2
    assert "sample_scores" in report["candidates"][0]
    assert report["candidates"][0]["nearest_score"] <= report["candidates"][0]["mean_reference_score"]
    assert report["candidates"][0]["roles"][0]["sim"]["width"] == 90


def test_dataset_reset_pose_tool_checks_overlay_constants() -> None:
    tool = (ROOT / "tools" / "check_dataset_reset_pose.py").read_text()

    assert "observation.state.robot_q_current" in tool
    assert "observation.state.hand_state" in tool
    assert "FLIP_TABLE_DATASET_INITIAL_UPPER_BODY_JOINT_POS" in tool
    assert "max_abs_delta" in tool


def test_prepared_scene_fixes_the_physical_workbench() -> None:
    tool = (ROOT / "tools" / "prepare_assembled_table_scene.py").read_text()

    assert 'WORKBENCH_BODY = "/World/Table278/Table278"' in tool
    assert "CreateKinematicEnabledAttr(True)" in tool
    assert '"workbench_kinematic": True' in tool
    assert "_remove_nested_physics_scenes" in tool
    assert "prim.IsA(UsdPhysics.Scene)" in tool
    assert '"removed_physics_scenes"' in tool
    assert "ASSEMBLED_RIGID_BODIES" in tool
    assert "_zero_rigid_body_velocities" in tool
    assert "CreateVelocityAttr().Set(zero)" in tool
    assert "CreateAngularVelocityAttr().Set(zero)" in tool
    assert '"zeroed_rigid_body_velocities"' in tool
    assert "_activate_leg_contact_reporting" in tool
    assert 'AddAppliedSchema("PhysxContactReportAPI")' in tool
    assert '"physxContactReport:threshold"' in tool
    assert '"source_detailed_collision_geometry_available": True' in tool
    assert 'LEG_SHAFT_COLLIDER_SUFFIX = "/Collisions/Leg001_Collider118"' in tool
    assert "_disambiguate_leg_contact_reporter_names" in tool
    assert 'LEG_VISUAL_MESH_NAME = "Leg001_visual"' in tool
    assert "Usd.NamespaceEditor(stage)" in tool
    assert '"renamed_leg_visuals"' in tool


def test_performance_benchmark_isolated_camera_and_collision_factors() -> None:
    scene_tool = (ROOT / "tools" / "prepare_assembled_table_scene.py").read_text()
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text()

    assert "def _simplify_white_table_collision" in scene_tool
    assert 'parser.add_argument("--simplify-white-collision", action="store_true")' in scene_tool
    assert '"visual_geometry_changed": False' in scene_tool
    assert '"mode": (' in scene_tool
    assert '"task_external_proxy"' in scene_tool
    assert '"internal_thread_colliders_enabled"' in scene_tool
    assert "LEG_EXTERNAL_COLLIDER_SUFFIXES" in scene_tool
    assert "if enabled_after != expected_after" in scene_tool
    assert "for body_path in (TABLE_BODY, *LEG_BODIES)" in scene_tool
    assert '"scene_geometry": scene_geometry' in scene_tool
    assert "class TeleopPerformanceBenchmarkPolicy(NoOpPolicy)" in policy
    assert 'enumerate(("head_on", "all_cameras_off") * 2)' in policy
    assert "env.render_enabled = head_enabled" in policy
    assert '"teleop_performance_benchmark.json"' in policy
    for name in ("run_eval.sh", "run_eval_in_container.sh"):
        runner = (ROOT / name).read_text()
        assert "FLIP_TABLE_SIMPLIFY_WHITE_COLLISION" in runner
        assert 'FLIP_TABLE_ROBOT_BASE_HEIGHT_M:-0.78' in runner
        if name == "run_eval.sh":
            assert 'FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE:-pinocchio' in runner
        assert 'FLIP_TABLE_CV_WARMUP_STEPS:-50' in runner
        assert "FLIP_TABLE_CV_SETTLED_SELECTION_STEPS" in runner
        assert "FLIP_TABLE_BENCHMARK_WARMUP_STEPS" in runner
        assert "FLIP_TABLE_BENCHMARK_MEASURE_STEPS" in runner
        assert "TeleopPerformanceBenchmarkPolicy" in runner


def test_host_runner_forwards_persistent_teleop_flag() -> None:
    runner = (ROOT / "run_eval.sh").read_text()

    assert 'FLIP_TABLE_TELEOP_PERSISTENT="${FLIP_TABLE_TELEOP_PERSISTENT:-false}"' in runner


def test_contact_randomization_resolves_shared_v1_scene_materials() -> None:
    overlay = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    ).read_text()

    assert '"white": "Looks/flip_table_contact_white"' in overlay
    assert '"workbench": "Looks/flip_table_contact_workbench"' in overlay
    assert "expected_white_minimum = (\n                        13" in overlay
    assert 'material_suffixes[surface_name].rsplit("/", 1)[-1]' in overlay
    assert "Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())" in overlay


def test_dex1_direct_target_actuators_match_robot_limits() -> None:
    patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text()

    assert "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V3" in patch
    assert 'G1_GRIPPER_CFG.actuators.pop("gearwbc_implicit_arms", None)' in patch
    assert 'G1_GRIPPER_CFG.actuators["waist"] = IdealPDActuatorCfg(' not in patch
    assert 'G1_GRIPPER_CFG.actuators["arms"] = IdealPDActuatorCfg(' in patch
    assert '".*_shoulder_pitch_joint": 25.0' in patch
    assert '".*_wrist_pitch_joint": 13.4' in patch
    assert "effort_limit_sim={" in patch
    assert "velocity_limit_sim={" in patch
    assert "effort_limit_sim=20.0" in patch
    assert "velocity_limit=0.2" in patch
    assert "velocity_limit_sim=0.2" in patch
    assert '"__FLIP_TABLE_DEX1_STIFFNESS__": repr(800.0 * dex1_stiffness_scale)' in patch
    assert "solver_position_iteration_count = 8" in patch
    assert "solver_velocity_iteration_count = 4" in patch


def test_balanced_wbc_adapter_rejects_waist_in_official_upper_group() -> None:
    adapter = (
        ROOT / "container_overlay" / "mdp" / "team_ramen_balanced_wbc_action.py"
    ).read_text()

    assert 'get_joint_group_indices("upper_body_no_hands")' in adapter
    assert 'get_joint_group_indices("upper_body")' in adapter
    assert "len(wbc_arm_ids) != 14" in adapter
    assert "set(wbc_arm_names) != set(ARM_JOINT_NAMES)" in adapter
    assert 'name.startswith("waist_")' in adapter


def test_wbc_audit_includes_waist_in_controller_ownership() -> None:
    audit = (
        ROOT.parents[1]
        / "model/flip_table_reinforcement_learning/scripts/audit_simulation_contract.py"
    ).read_text()

    assert (
        "WBC_OWNED_JOINT_NAMES = LOWER_BODY_JOINT_NAMES "
        "+ UPPER_BODY_JOINT_NAMES[:3]"
    ) in audit
    assert "_resolve_joint_ids(robot, WBC_OWNED_JOINT_NAMES)" in audit


def test_v1_camera_registry_declares_the_stereo_right_eye() -> None:
    patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text()

    assert "CAMERA_CONFIG_FIELDS_MARKER" in patch
    assert "patch_camera_config_field_generation" in patch
    assert "make_configclass" in patch
    assert "true stereo" in patch


def test_avp_listener_survives_the_framework_episode_reset() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text()
    reset_method = policy.split("    def reset_model(self) -> None:", 1)[1].split(
        "\n\nclass LeRobotACTPolicy", 1
    )[0]

    assert "def _open_server" in policy
    assert "self._open_server()" in policy
    assert "self._server.close()" not in reset_method


def test_avp_uses_direct_sensors_and_wall_clock_camera_rate() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text()
    avp = policy.split("class AvpTeleopPolicy", 1)[1].split(
        "class LeRobotACTPolicy", 1
    )[0]

    assert "def _sensor_rgb" in avp
    assert "camera_period_s = 1.0 / self._preview_hz" in avp
    assert "camera_due = now >= next_camera_time" in avp
    assert "active_period = 1.0 / self._preview_hz" in avp
    assert "if self._sim_recording:" in avp
    assert "self.add_video_frame(" in avp
    assert "def _resolve_review_video_hz" in avp
    assert 'FLIP_TABLE_TELEOP_REVIEW_VIDEO_HZ' in avp
    assert "global_review_hz=" in avp
    assert "review_now >= self._next_review_video_time" in avp
    assert "self._observation_send_times: deque[float] = deque(maxlen=61)" in avp
    assert "cv2.setRNGSeed(noise_seed)" in avp
    assert "ThreadPoolExecutor" not in avp
    assert '"noise_seed": selected[4]' in avp
    assert "def _observation_sender_loop" in avp
    assert "dropped_operator_frames" in avp
    assert "def _warm_live_camera_pipeline" in avp
    assert "self._warm_live_camera_pipeline(task_env)" in avp
    assert 'message_diagnostics["last_applied_command_sequence"]' in avp
    assert "self._sim_hold_arm = measured_arm.copy()" in avp
    assert "fixed_hold_action = _joint_position_hold_action" in avp
    assert "task_env.step(fixed_hold_action)" in avp
    assert "socket.SO_SNDBUF, 4 * 1024 * 1024" in avp


def test_avp_live_and_recording_jpeg_preserve_rgb_color(monkeypatch) -> None:
    import cv2

    module = _load_policy_module(monkeypatch, "flip_table_avp_jpeg_test")
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = (230, 40, 90)

    for recording in (False, True):
        payload = module.AvpTeleopPolicy._jpeg(image, recording=recording)
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        assert decoded.shape == image.shape
        np.testing.assert_allclose(decoded[240, 320], image[240, 320], atol=5)


def test_recorded_replay_video_sampling_does_not_change_control_clock(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_recorded_video_stride_test")
    monkeypatch.setenv("FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ", "10")
    assert module.RecordedJointTargetPolicy._review_video_stride(50.0) == 5
    monkeypatch.setenv("FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ", "0")
    with pytest.raises(ValueError, match="REPLAY_REVIEW_VIDEO_HZ"):
        module.RecordedJointTargetPolicy._review_video_stride(50.0)


def test_in_process_avp_runner_removes_only_camera_observation_clones() -> None:
    runner = (ROOT / "tools" / "run_in_process_eval.py").read_text()

    assert "def _remove_avp_camera_observation_terms" in runner
    assert '"first_person_camera_rgb"' in runner
    assert '"head_right_camera_rgb"' in runner
    assert (
        'if args.policy_name in {"AvpTeleopPolicy", "Dex1ForceCalibrationPolicy"}:'
        in runner
    )
    assert "_remove_avp_camera_observation_terms(env)" in runner
    assert "def _verify_unique_upper_body_actuators" in runner
    assert "_verify_unique_upper_body_actuators(env)" in runner
    assert "if len(owners) != 21:" in runner
    assert "def _verify_runtime_rates" in runner
    assert "_verify_runtime_rates(env)" in runner
    assert '"--headless"' in runner


def test_persistent_evaluation_worker_reuses_only_isaac_startup() -> None:
    worker = (ROOT / "tools" / "persistent_eval_worker.py").read_text(encoding="utf-8")
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    controller = (ROOT / "persistent_eval.sh").read_text(encoding="utf-8")
    replay_runner = (
        ROOT / "real_to_sim_calibration" / "run_anchor_replay.sh"
    ).read_text(encoding="utf-8")

    assert "env.reset(seed=seed)" in worker
    assert "episode_output_dir.mkdir(exist_ok=False)" in worker
    assert "_restore_output_ownership(output_dir)" in worker
    assert "ALLOWED_ENVIRONMENT_KEYS" in worker
    assert "directory.chmod(0o777)" in worker
    assert "FLIP_TABLE_PERSISTENT_EVAL_WORKER" in runner
    assert "persistent_eval_worker.py" in runner
    assert "FLIP_TABLE_PERSISTENT_EVAL_ROOT" in replay_runner
    assert "persistent_eval_client.py" in replay_runner
    assert 'persistent_eval.sh" ensure' in replay_runner
    assert "wait_until_ready" in controller
    assert "nohup setsid env" in controller
    assert "</dev/null" in controller
    assert 'EXIT_FILE="$RUNTIME_DIR/last_exit.json"' in controller
    assert 'LIFECYCLE_FILE="$RUNTIME_DIR/last_lifecycle.json"' in controller
    assert '"exit_code": int(sys.argv[2])' in controller
    assert '"finished_unix_s": time.time()' in controller
    assert 'stage="standard_environment_initialization_failed"' in worker
    assert "environment_shutdown_after_{previous_stage}" in worker
    assert "  ensure)" in controller
    assert "force-stop" in controller
    assert "restart" in controller
    assert "worker_running && [[ -f \"$RUNTIME_DIR/ready.json\" ]]" in controller
    assert "--policy-name CvRuleBasedPolicy" in (
        ROOT / "real_to_sim_calibration" / "README.md"
    ).read_text(encoding="utf-8")


def test_reset_time_arm_identification_updates_the_explicit_pd_layer() -> None:
    task = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    ).read_text(encoding="utf-8")

    start = task.index("def _randomize_upper_body_joint_properties")
    end = task.index("def _sample_camera_image_model", start)
    randomization = task[start:end]
    assert "IdealPDActuator" in randomization
    assert "robot.actuators" in randomization
    assert 'getattr(actuator, property_name)' in randomization
    assert 'write_joint_{property_name}_to_sim_index' in randomization
    assert "simulator_joint_drive" in randomization


def test_in_process_avp_runner_decouples_render_rate_from_servo(monkeypatch) -> None:
    module = _load_module(
        ROOT / "tools" / "run_in_process_eval.py",
        "flip_table_in_process_render_interval_test",
    )
    cfg = types.SimpleNamespace(
        sim=types.SimpleNamespace(
            dt=0.0,
            render_interval=0,
            render=types.SimpleNamespace(),
        ),
        decimation=0,
    )
    env_server = types.SimpleNamespace(make_env_cfg=lambda config: ("task", cfg))
    monkeypatch.setenv("FLIP_TABLE_SIM_PHYSICS_HZ", "100")
    monkeypatch.setenv("FLIP_TABLE_SIM_RENDER_INTERVAL", "3")

    module._install_realtime_render_config(env_server)
    _, configured = env_server.make_env_cfg({})

    assert configured.sim.dt == 0.01
    assert configured.decimation == 2
    assert configured.sim.render_interval == 3
    launcher = (ROOT / "run_eval.sh").read_text()
    assert '-e FLIP_TABLE_SIM_RENDER_INTERVAL=' in launcher


def test_in_process_avp_runner_rejects_overlapping_upper_body_actuators() -> None:
    module = _load_module(
        ROOT / "tools" / "run_in_process_eval.py",
        "flip_table_in_process_actuator_test",
    )
    joint_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    for side in ("left", "right"):
        joint_names.extend(
            [
                f"{side}_shoulder_pitch_joint",
                f"{side}_shoulder_roll_joint",
                f"{side}_shoulder_yaw_joint",
                f"{side}_elbow_joint",
                f"{side}_wrist_roll_joint",
                f"{side}_wrist_pitch_joint",
                f"{side}_wrist_yaw_joint",
                f"{side}_dex1_finger_joint_1",
                f"{side}_dex1_finger_joint_2",
            ]
        )
    actuators = {
        "waist": types.SimpleNamespace(joint_names_expr=["waist_.*_joint"]),
        "arms": types.SimpleNamespace(
            joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
        ),
        "grippers": types.SimpleNamespace(joint_names_expr=".*_dex1_finger_joint_.*"),
    }
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_names=joint_names))
    unwrapped = types.SimpleNamespace(
        scene={"robot": robot},
        cfg=types.SimpleNamespace(
            scene=types.SimpleNamespace(robot=types.SimpleNamespace(actuators=actuators))
        ),
    )
    env = types.SimpleNamespace(unwrapped=unwrapped)

    module._verify_unique_upper_body_actuators(env)
    actuators["legacy_arms"] = types.SimpleNamespace(
        joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
    )
    with pytest.raises(RuntimeError, match="exactly one actuator"):
        module._verify_unique_upper_body_actuators(env)


def test_avp_observation_sender_keeps_latest_operator_frame(monkeypatch) -> None:
    import queue

    module = _load_policy_module(monkeypatch, "flip_table_avp_observation_sender")
    policy = module.AvpTeleopPolicy.__new__(module.AvpTeleopPolicy)
    delivered: list[int] = []
    policy._observation_queue = queue.Queue(maxsize=1)
    policy._observation_sender_thread = None
    policy._observation_sender_error = None
    policy._dropped_operator_frames = 0
    policy._sim_recording = False
    policy._deliver_observation = lambda payload: delivered.append(payload["sequence"])

    policy._start_observation_sender()
    policy._queue_observation({"sequence": 1})
    policy._flush_observation_sender()
    policy._stop_observation_sender()

    assert delivered == [1]
    assert policy._observation_sender_error is None


def test_avp_pre_ready_disconnect_retries_before_any_control(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_avp_pre_ready_retry")
    policy = module.AvpTeleopPolicy.__new__(module.AvpTeleopPolicy)
    attempts: list[float] = []
    dropped: list[bool] = []

    def accept(*, timeout_s: float) -> None:
        attempts.append(timeout_s)

    def wait_ready(*, timeout_s: float) -> None:
        if len(attempts) == 1:
            raise RuntimeError("teleoperation transport disconnected")

    policy._accept_client = accept
    policy._wait_for_client_ready = wait_ready
    policy._drop_unready_client = lambda: dropped.append(True)
    monkeypatch.setenv("FLIP_TABLE_TELEOP_ACCEPT_TIMEOUT_S", "1")
    monkeypatch.setenv("FLIP_TABLE_TELEOP_PRE_READY_TIMEOUT_S", "0.1")

    policy._accept_ready_client()

    assert len(attempts) == 2
    assert dropped == [True]


def test_avp_initial_scene_preflight_reads_white_table_motion(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_avp_scene_preflight")
    diagnostics = {
        "white_table": {
            "position_world_m": [-1.4, 2.3, 0.79],
            "linear_velocity_m_s": [0.03, 0.04, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.2],
        }
    }

    position, linear_speed, angular_speed = (
        module.AvpTeleopPolicy._initial_table_motion(diagnostics)
    )

    np.testing.assert_allclose(position, [-1.4, 2.3, 0.79])
    assert linear_speed == pytest.approx(0.05)
    assert angular_speed == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="omit the white table"):
        module.AvpTeleopPolicy._initial_table_motion({})


def test_avp_force_audit_samples_every_servo_step(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_avp_force_audit")
    policy = module.AvpTeleopPolicy.__new__(module.AvpTeleopPolicy)
    policy.teleop_config = types.SimpleNamespace(
        rates=types.SimpleNamespace(servo_hz=50.0)
    )
    policy._force_audit = policy._new_force_audit()
    policy._applied_hand = np.asarray((0.0, 0.0))

    sample = {
        "gripper_contact_force_n": {
            "available": True,
            "left_max_n": 3.0,
            "right_max_n": 0.0,
        },
        "dex1_drive_force_n": {
            "available": True,
            "left_max_n": 12.0,
            "right_max_n": 0.2,
        },
    }
    env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(
            flip_table_teleop_force_diagnostics=lambda: sample
        )
    )
    policy._sample_force_diagnostics(env)
    policy._sample_force_diagnostics(env)
    diagnostics = policy._force_audit_diagnostics()

    assert diagnostics["gripper_contact_force_n"]["left_max_n"] == 3.0
    assert diagnostics["dex1_drive_force_n"]["left_max_n"] == 12.0
    assert diagnostics["dex1_grasp_force_audit"][
        "sustained_contact_max_s_left_right"
    ] == [0.04, 0.0]
    assert diagnostics["dex1_grasp_force_audit"][
        "closed_without_load_count_left_right"
    ] == [0, 2]


def test_d405_wrist_camera_calibration_tool_checks_patch_and_step(tmp_path: Path) -> None:
    tool = ROOT / "tools" / "verify_d405_wrist_camera_calibration.py"
    step = tmp_path / "Dex1_1_Realsense_D405_Camera_Mount_M5010.STEP"
    points = "\n".join(
        f"#{idx} = CARTESIAN_POINT ( 'NONE',  ( {idx % 90:.3f}, {idx % 40:.3f}, {idx % 25:.3f} ) ) ;"
        for idx in range(1, 620)
    )
    step.write_text(
        "ISO-10303-21;\nDATA;\n"
        + points
        + "\n#700 = CIRCLE ( 'NONE', #701, 25.50000000000000000 ) ;\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    output = tmp_path / "calibration_report.json"

    subprocess.run(
        [
            sys.executable,
            str(tool),
            "--step",
            str(step),
            "--output",
            str(output),
            "--fail-on-warning",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["camera_patch"]["width"] == 640
    assert report["camera_patch"]["height"] == 480
    # These are the pinned raw D405 color-calibration intrinsics, not the old
    # generic product-sheet FOV that produced an unrealistically wide image.
    assert abs(report["camera_patch"]["derived_horizontal_fov_deg"] - 72.6784) < 0.05
    assert abs(report["camera_patch"]["derived_vertical_fov_deg"] - 57.7728) < 0.05
    assert report["step"]["cartesian_point_count"] > 500
    assert report["distribution_compare"]["status"] == "skipped"
    assert "hand-eye calibration" in " ".join(report["limitations"])


def test_runner_mounts_overlay_and_output_dir() -> None:
    runner = (ROOT / "run_eval.sh").read_text()

    assert "container_overlay/robofinals_tasks/local_auto_tasks/assemble_table_task.py" in runner
    assert "container_overlay/policy/flip_table_eval_policy.py" in runner
    assert "/workspace/robofinals/robofinals_tasks/local_auto_tasks/assemble_table_task.py:ro" in runner
    assert "/workspace/robofinals/policy/flip_table_eval_policy.py:ro" in runner
    assert "/workspace/robofinals/eval_result" in runner
    assert "prepare_assembled_table_scene.py" in runner
    assert "SCENE_PREPARE_TOOL" in runner
    assert "CONFIG_PATH=/tmp/flip_table_eval.yml" in runner
    assert 'config["env_cfg"]["layout"] = sys.argv[2]' in runner
    assert '"$CONFIG_PATH"' in runner
    assert "robofinals/scripts/env_server.py" in runner
    assert "args_cli.disable_fabric = cfg.disable_fabric" in runner
    assert "optimize_rendering(env, args_cli)" in runner
    assert "robofinals/scripts/policy/eval_policy.py" in runner
    start_marker = '  -lc "$(cat <<\'CONTAINER_SCRIPT\'\n'
    end_marker = "\nCONTAINER_SCRIPT\n)\""
    assert start_marker in runner
    assert end_marker in runner
    container_script = runner.split(start_marker, 1)[1].split(end_marker, 1)[0]
    subprocess.run(
        ["bash", "-n"],
        input=container_script,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "  bash -lc '" not in runner
    assert "FLIP_TABLE_POLICY_NAME" in runner
    assert "CHECKPOINT_HOST" in runner
    assert "checkpoint_mount_args" in runner
    assert "/workspace/flip_table_policy_checkpoint" in runner
    assert "FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL" in runner
    assert "FLIP_TABLE_WORKBENCH_FRONT_AXIS" in runner
    assert "FLIP_TABLE_ROBOT_YAW_OFFSET_RAD" in runner
    assert "FLIP_TABLE_ROBOT_ROOT_POS_LOCAL" in runner
    assert "FLIP_TABLE_ROBOT_ROOT_YAW_RAD" in runner
    assert "FLIP_TABLE_USE_DEFAULT_ROBOT_POSE" in runner
    assert "FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS" in runner
    assert "FLIP_TABLE_DEFAULT_ROBOT_FORWARD_CELLS" in runner
    assert "FLIP_TABLE_DEBUG_GRID_CELL_M" in runner
    assert "patch_g1_global_camera.py" in runner
    assert "patch_g1_wbc_action_continuity.py" in runner
    assert 'BALANCED_WBC_ACTION="$FEATURE_DIR/container_overlay/mdp/team_ramen_balanced_wbc_action.py"' in runner
    assert 'WBC_CONTINUITY_PATCH="$FEATURE_DIR/container_overlay/patches/patch_g1_wbc_action_continuity.py"' in runner
    assert "/workspace/flip_table_simulation/team_ramen_balanced_wbc_action.py:ro" in runner
    assert "target_wbc_adapter=/workspace/robofinals/robofinals/core/mdp/actions/team_ramen_balanced_wbc_action.py" in runner
    assert "FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA" in runner
    assert "FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS" in runner
    assert "FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL" in runner
    assert "FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES" in runner
    assert "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION" in runner
    assert "FLIP_TABLE_RANDOMIZE_ROOM" in runner
    assert "FLIP_TABLE_ROOM_TILE_SIZE_M" in runner
    assert "FLIP_TABLE_ROOM_FLOOR_PATTERNS" in runner
    assert "FLIP_TABLE_ROOM_WALL_PATTERNS" in runner
    assert 'FLIP_TABLE_POLICY_NAME:-}" =~ ' in runner
    assert "NoOpPolicy|ScriptedJointPolicy|RecordedJointTargetPolicy|RecordedFullBodyTargetPolicy|AvpTeleopPolicy|LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy" in runner
    assert "RecordedJointTargetPolicy" in runner
    assert '"RecordedFullBodyTargetPolicy": ".flip_table_eval_policy"' in runner
    assert "FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS" in runner
    assert "FLIP_TABLE_CAMERA_WIDTH" in runner
    assert "FLIP_TABLE_CAMERA_HEIGHT" in runner
    assert "FLIP_TABLE_CV_HEAD_CAMERA_FK_MODE" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE" in runner
    assert "FLIP_TABLE_SAVE_CAMERA_FRAMES" in runner
    assert "FLIP_TABLE_CAMERA_FRAME_INDEX" in runner
    assert "FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL" in runner
    assert "FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS" in runner
    assert 'FLIP_TABLE_TABLE_LONG_RANGE_M:-0.12' in runner
    assert 'FLIP_TABLE_TABLE_DEPTH_RANGE_M:-0.035' in runner
    assert 'FLIP_TABLE_TABLE_YAW_RANGE_RAD:-3.141592653589793' in runner
    assert 'FLIP_TABLE_ROBOT_DISTANCE_M:-0.26' in runner
    assert 'FLIP_TABLE_ROBOT_DISTANCE_RANGE_M:-0.04' in runner
    assert 'FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M:-0.62' in runner
    assert 'FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M:-0.20' in runner
    assert 'FLIP_TABLE_ROBOT_LATERAL_RANGE_M:-0.10' in runner
    assert 'FLIP_TABLE_ROBOT_YAW_RANGE_RAD:-0.08' in runner
    assert 'FLIP_TABLE_USE_DEFAULT_ROBOT_POSE:-false' in runner
    assert 'FLIP_TABLE_JOINT_NOISE_RAD:-0.02' in runner
    assert 'FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE:-true' in runner
    assert 'FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE:-0.5' in runner
    assert 'FLIP_TABLE_DEX1_FINGER_NOISE_M:-0.002' in runner
    assert 'FLIP_TABLE_RANDOMIZE_ROOM:-true' in runner
    assert 'FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS:-true' in runner
    assert 'FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE:-0.65,0.95' in runner
    assert 'FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE:-0.35,0.46' in runner
    assert "FLIP_TABLE_EVAL_RANDOMIZE_MASS=false" in runner
    assert 'FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE:-0.02,0.08' in runner
    assert 'FLIP_TABLE_RANDOMIZE_ROOM_PROPS:-true' in runner
    assert 'FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M:-2.20' in runner
    assert 'FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M:-0.30' in runner
    assert 'FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M:-0.20' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M:-0.50' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG:-80' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_AXIS:-+x' in runner
    assert 'FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K:-3800,6500' in runner
    assert 'FLIP_TABLE_LIGHT_EXPOSURE_RANGE:--0.35,0.35' in runner
    assert 'FLIP_TABLE_SUN_ELEVATION_DEG:-18,72' in runner
    assert '/workspace/flip_table_room_assets:ro' in runner
    assert 'FLIP_TABLE_LIGHT_INTENSITY_RANGE:-450,1200' in runner
    assert 'FLIP_TABLE_LIGHT_COLOR_RANGE:-0.82,1.0' in runner
    assert "FLIP_TABLE_G1_GLOBAL_CAMERA_PRIM_PATH" not in runner
    assert "FLIP_TABLE_G1_GLOBAL_CAMERA_OFFSET_POS" not in runner
    assert "FLIP_TABLE_G1_GLOBAL_CAMERA_OFFSET_RPY_RAD" not in runner
    assert "paperc/robofinals:RoboFinals-IKEA-V1" in runner


def test_feature_scripts_do_not_embed_workstation_paths() -> None:
    paths = [ROOT / "README.md", *sorted((ROOT / "tools").glob("*.py"))]

    for path in paths:
        assert "/home/suzuki" not in path.read_text(), path


def test_eval_runners_share_the_same_evaluation_modes() -> None:
    expected = "nominal, randomized, or unseen_dr"
    for name in ("run_eval.sh", "run_eval_in_container.sh"):
        runner = (ROOT / name).read_text(encoding="utf-8")
        assert expected in runner
        assert "FLIP_TABLE_GROOT_DR_PROFILE" in runner


def test_groot_validation_and_held_out_dr_profiles_are_disjoint() -> None:
    profile_script = ROOT / "groot_dr_profiles.sh"

    def read_profile(name: str) -> list[str]:
        command = (
            f'source "{profile_script}"; '
            f"groot_apply_dr_profile {name}; "
            "printf '%s\\n' "
            '"$FLIP_TABLE_GROOT_DR_PROFILE" '
            '"$FLIP_TABLE_ROOM_FLOOR_MATERIALS" '
            '"$FLIP_TABLE_ROOM_WALL_MATERIALS" '
            '"$FLIP_TABLE_ROOM_FLOOR_PATTERNS" '
            '"$FLIP_TABLE_ROOM_WALL_PATTERNS" '
            '"$FLIP_TABLE_ROOM_PROP_ASSETS" '
            '"$FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE"'
        )
        return subprocess.run(
            ["bash", "-c", command],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()

    validation = read_profile("validation_v1")
    held_out = read_profile("held_out_v1")
    assert validation[0] == "validation_v1"
    assert held_out[0] == "held_out_v1"
    for validation_values, held_out_values in zip(validation[1:6], held_out[1:6]):
        assert set(validation_values.split(",")).isdisjoint(
            held_out_values.split(",")
        )
    assert validation[6] == "0.70,0.90"
    assert held_out[6] == "0.65,0.70"


def test_in_container_runner_installs_overlay_and_runs_eval() -> None:
    runner = (ROOT / "run_eval_in_container.sh").read_text()

    assert "ROBOFINALS_ROOT" in runner
    assert "FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT" in runner
    assert "FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES=true" in runner
    assert "robofinalsbak" in runner
    assert "original_assemble_table_task" in runner
    assert "flip_table_eval_policy.py" in runner
    assert "patch_g1_global_camera.py" in runner
    assert "patch_g1_wbc_action_continuity.py" not in runner
    assert "team_ramen_balanced_wbc_action.py" in runner
    assert "FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS" in runner
    assert "FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL" in runner
    assert "FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES" in runner
    assert "FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS" in runner
    assert "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION" in runner
    assert 'FLIP_TABLE_POLICY_NAME:-}" =~ ' in runner
    assert "NoOpPolicy|ScriptedJointPolicy|RecordedJointTargetPolicy|RecordedFullBodyTargetPolicy|AvpTeleopPolicy|LeRobotACTPolicy|FlowMatchingBCPolicy|FlowMatchingRLPDPolicy" in runner
    assert '"RecordedFullBodyTargetPolicy": ".flip_table_eval_policy"' in runner
    assert '"FlowMatchingBCPolicy": ".flip_table_eval_policy"' in runner
    assert '"FlowMatchingRLPDPolicy": ".flip_table_eval_policy"' in runner
    assert "FLIP_TABLE_FLOW_N_ACTION_STEPS" in runner
    assert 'cp -a "$FLOW_PACKAGE" "$ROBOFINALS_ROOT/policy/flow_matching"' in runner
    assert 'cp -a "$RLPD_PACKAGE" "$ROBOFINALS_ROOT/policy/rlpd"' in runner
    assert "RecordedJointTargetPolicy" in runner
    assert "FLIP_TABLE_CAMERA_WIDTH" in runner
    assert "FLIP_TABLE_CAMERA_HEIGHT" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_POS" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_OFFSET_ROT" in runner
    assert "FLIP_TABLE_DEX1_WRIST_CAMERA_HORIZONTAL_APERTURE" in runner
    assert "FLIP_TABLE_SAVE_CAMERA_FRAMES" in runner
    assert "FLIP_TABLE_CAMERA_FRAME_INDEX" in runner
    assert "FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL" in runner
    assert "FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS" in runner
    assert 'FLIP_TABLE_TABLE_LONG_RANGE_M:-0.12' in runner
    assert 'FLIP_TABLE_TABLE_DEPTH_RANGE_M:-0.035' in runner
    assert 'FLIP_TABLE_TABLE_YAW_RANGE_RAD:-3.141592653589793' in runner
    assert 'FLIP_TABLE_ROBOT_DISTANCE_M:-0.26' in runner
    assert 'FLIP_TABLE_ROBOT_DISTANCE_RANGE_M:-0.04' in runner
    assert 'FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M:-0.62' in runner
    assert 'FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M:-0.20' in runner
    assert 'FLIP_TABLE_ROBOT_LATERAL_RANGE_M:-0.10' in runner
    assert 'FLIP_TABLE_ROBOT_YAW_RANGE_RAD:-0.08' in runner
    assert 'FLIP_TABLE_USE_DEFAULT_ROBOT_POSE:-false' in runner
    assert 'FLIP_TABLE_JOINT_NOISE_RAD:-0.02' in runner
    assert 'FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE:-true' in runner
    assert 'FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE:-0.5' in runner
    assert 'FLIP_TABLE_DEX1_FINGER_NOISE_M:-0.002' in runner
    assert 'FLIP_TABLE_RANDOMIZE_ROOM:-true' in runner
    assert 'FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS:-true' in runner
    assert 'FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE:-0.65,0.95' in runner
    assert 'FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE:-0.35,0.46' in runner
    assert "FLIP_TABLE_EVAL_RANDOMIZE_MASS=false" in runner
    assert 'FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE:-0.02,0.08' in runner
    assert 'FLIP_TABLE_RANDOMIZE_ROOM_PROPS:-true' in runner
    assert 'FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M:-2.20' in runner
    assert 'FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M:-0.30' in runner
    assert 'FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M:-0.20' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M:-0.50' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG:-80' in runner
    assert 'FLIP_TABLE_ROOM_PROP_FRONT_AXIS:-+x' in runner
    assert 'FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K:-3800,6500' in runner
    assert 'FLIP_TABLE_LIGHT_EXPOSURE_RANGE:--0.35,0.35' in runner
    assert 'FLIP_TABLE_SUN_ELEVATION_DEG:-18,72' in runner
    assert 'FLIP_TABLE_LIGHT_INTENSITY_RANGE:-450,1200' in runner
    assert 'FLIP_TABLE_LIGHT_COLOR_RANGE:-0.82,1.0' in runner
    assert "NoOpPolicy" in runner
    assert "robofinals/scripts/env_server.py" in runner
    assert "args_cli.disable_fabric = cfg.disable_fabric" in runner
    assert "optimize_rendering(env, args_cli)" in runner
    assert "robofinals/scripts/policy/eval_policy.py" in runner
    assert "--livestream 2" in runner


def test_vast_sync_script_excludes_generated_artifacts() -> None:
    sync_script = ROOT / "tools" / "sync_to_vast.sh"
    script = sync_script.read_text()
    readme = (ROOT / "README.md").read_text()

    assert os.access(sync_script, os.X_OK)
    assert "REMOTE_REPO_DIR" in script
    assert "rsync -az --delete --delete-delay" in script
    assert "--exclude='.git/'" in script
    assert "--exclude='.checkpoints/'" in script
    assert "--exclude='outputs/'" in script
    assert "--exclude='logs/'" in script
    assert "--exclude='wandb/'" in script
    assert "--exclude='model/subtask_policy_training/.venv/'" in script
    assert "--exclude='model/subtask_policy_training/.venv_lerobot060/'" in script
    assert "LOCAL_CHECKPOINT" in script
    assert "REMOTE_CHECKPOINT_DIR" in script
    assert "act_flip_table_upper_body" in script
    assert "FLIP_TABLE_FLOW_CHECKPOINT" in script
    assert "FLIP_TABLE_RLPD_COMBINED_CHECKPOINT" in script
    assert "sync_checkpoint" in script
    assert "evaluate/flip_table_simulation/run_eval_in_container.sh" in script
    # Vast mirroring remains an archival utility, but the maintained release
    # workflow is the local RTX 5090 worker.  Do not reintroduce a remote
    # execution instruction through this optional helper.
    assert "RTX 5090" in readme
    assert "Vast.ai を通常手順・再現手順として使いません" in readme


def test_overlay_keeps_registered_class_name() -> None:
    overlay = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    ).read_text()

    assert "class AssembleTableTask" in overlay
    assert 'task_name: str = "FlipTableEvalTask"' in overlay
    assert "flip_table_eval_reset" in overlay
    assert "FlipTableEvalEventCfg" in overlay
    assert "self.events_cfg = FlipTableEvalEventCfg" in overlay
    assert "UsdPhysics.FixedJoint" in overlay
    assert "removed stale table joint" in overlay
    assert "FlipTableEvalFixedJoint_" in overlay
    assert "_quat_conjugate_xyzw" in overlay
    assert "_base_table_pos_local" in overlay
    assert "_base_leg_positions_local" in overlay
    assert "base_table_quat[:, 0] = 1.0" in overlay
    assert "assembly_delta_quat" in overlay
    assert "assembly_delta_rot" in overlay
    assert "FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL" in overlay
    assert 'workbench_prim_path: str = "Table278"' in overlay
    assert "_sample_table_pose_on_workbench" in overlay
    assert "_sample_robot_pose_for_workbench_front" in overlay
    assert "FLIP_TABLE_TABLE_PLACEMENT_BUFFER_M" in overlay
    assert "Constrain the complete tabletop footprint" in overlay
    assert "footprint_allowance" in overlay
    assert "FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M" in overlay
    assert "FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION" in overlay
    assert "within (0, 1]" in overlay
    assert "Rejection-sample within the configured yaw" in overlay
    assert "a calibration tabletop pose does not satisfy the bounded workbench support condition" in overlay
    assert "Table yaw must never move G1 around the workbench" in overlay
    assert "manipulation_target_xy" not in overlay
    assert "FLIP_TABLE_ROBOT_NEAR_EDGE_DISTANCE_M" not in overlay
    assert "_place_robot_near_table" not in overlay
    assert "_log_robot_default_pose" in overlay
    assert "_apply_default_robot_pose_offset" in overlay
    assert "FLIP_TABLE_USE_DEFAULT_ROBOT_POSE" in overlay
    assert "FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS" in overlay
    assert '"FLIP_TABLE_WORKBENCH_FRONT_AXIS", (0.0, -1.0)' in overlay
    assert "robot_yaw[i] = approach_yaw + yaw_offset + heading_jitter" in overlay
    assert '"FLIP_TABLE_ROBOT_DISTANCE_M", 0.26' in overlay
    assert '"FLIP_TABLE_ROBOT_DISTANCE_RANGE_M", 0.04' in overlay
    assert '"FLIP_TABLE_ROBOT_LATERAL_RANGE_M", 0.10' in overlay
    assert "FLIP_TABLE_ROBOT_WBC_SETTLE_CLEARANCE_M" in overlay
    assert "scale = placement_min_table_distance / table_distance" in overlay
    assert "_set_robot_planar_base_pose" in overlay
    assert '"base_x_joint"' in overlay
    assert '"base_y_joint"' in overlay
    assert '"base_yaw_joint"' in overlay
    assert "_refresh_camera_sensors" in overlay
    assert "update_latest_camera_pose = True" in overlay
    assert "G1_MATERIAL_COLORS" in overlay
    assert "FLIP_TABLE_APPLY_G1_VISUAL_MATERIALS" in overlay
    assert "_ensure_robot_visual_materials" in overlay
    assert '"dex1_base": (0.41176, 0.41176, 0.41176)' in overlay
    assert '"dex1_finger_1": (0.79216, 0.81961, 0.93333)' in overlay
    assert "UsdPreviewSurface" in overlay
    assert "MaterialBindingAPI.Apply" in overlay
    assert "current_joint_pos = as_torch(data.joint_pos)[env_ids].clone()" in overlay
    assert "joint_pos[:, lower_ids] = current_joint_pos[:, lower_ids]" in overlay
    assert "_write_robot_root_pose" in overlay
    assert "_aim_global_camera_at_table" not in overlay
    assert "_reapply_global_camera_pose" not in overlay
    assert "FLIP_TABLE_SUCCESS_DOT_THRESHOLD" in overlay
    assert '"FLIP_TABLE_SUCCESS_DOT_THRESHOLD", -0.95' in overlay
    assert "FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M" in overlay
    assert "FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S" in overlay
    assert "FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S" in overlay
    assert "FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M" in overlay
    assert "usable_half_length" in overlay
    assert "usable_half_depth" in overlay
    assert "workbench_footprint_margin_m" in overlay
    assert "FLIP_TABLE_SUCCESS_HOLD_STEPS" in overlay
    assert "FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS" in overlay
    assert "self._stable_success_result" in overlay
    assert "_stable_flip_success_components" in overlay
    assert "self._stable_success_streak" in overlay
    assert "within_workbench" in overlay
    assert "FLIP_TABLE_SUCCESS_DEBUG_EVERY" in overlay
    assert "table_yaw_delta=" in overlay
    assert "[FlipTableEvalTask] light randomization:" in overlay
    assert "_debug_global_camera_pose" in overlay
    assert 'self._find_prim_by_suffix(env, "global_camera", env_id=0)' in overlay
    assert "_capture_lower_body_lock" in overlay
    assert "_apply_lower_body_lock" in overlay
    assert "joint_ids=lower_ids_i32" in overlay
    assert "locked_vel = torch.zeros_like(locked_pos)" in overlay
    assert "FLIP_TABLE_LOCK_LOWER_BODY" in overlay
    assert "FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS" in overlay
    assert '"base_,hip,knee,ankle,waist"' in overlay
    assert "FLIP_TABLE_REQUIRE_WAIST_LOCK" in overlay
    for runner_name in ("run_eval.sh", "run_eval_in_container.sh"):
        runner = (ROOT / runner_name).read_text()
        assert "base_,hip,knee,ankle,waist" in runner
    assert "FLIP_TABLE_DATASET_INITIAL_UPPER_BODY_JOINT_POS" in overlay
    assert "FLIP_TABLE_DATASET_INITIAL_DEX1_FINGER_JOINT_POS" in overlay
    assert "_env_named_float_map" in overlay
    assert "FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS" in overlay
    assert "FLIP_TABLE_USE_DATASET_INITIAL_UPPER_BODY" in overlay
    assert "FLIP_TABLE_UPPER_BODY_INITIAL_POSE_RANGES_RAD" in overlay
    assert "FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE" in overlay
    assert '"left_shoulder_pitch_joint": 0.12' in overlay
    assert '"waist_roll_joint": 0.04' in overlay
    assert '"right_wrist_pitch_joint": 0.08' in overlay
    assert '"FLIP_TABLE_DEX1_FINGER_NOISE_M", 0.002' in overlay
    assert '"FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M", 0.62' in overlay
    assert '"FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M", 0.20' in overlay
    assert "projected_half_depth" not in overlay
    assert "table_distance < placement_min_table_distance" in overlay
    assert 'for side in ("left", "right")' in overlay
    assert "joint_noise[:, joint_names.index(finger_name)] = hand_noise" in overlay
    assert '"right_wrist_yaw_joint": 0.6012' in overlay
    assert '"right_dex1_finger_joint_1": 0.02437' in overlay
    assert "FLIP_TABLE_RANDOMIZE_ROOM" in overlay
    assert "ROOM_FLOOR_MATERIALS" in overlay
    assert "ROOM_WALL_MATERIALS" in overlay
    assert "ROOM_PROP_ASSETS" in overlay
    assert "UsdUVTexture" in overlay
    assert "UsdLux.SphereLight" in overlay
    overlay_tree = ast.parse(overlay)
    lighting_method = next(
        node
        for node in ast.walk(overlay_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_randomize_lighting"
    )
    lighting_source = ast.get_source_segment(overlay, lighting_method)
    assert lighting_source is not None
    assert "for env_id in env_ids.tolist()" in lighting_source
    assert "for env_id in range(env.num_envs)" not in lighting_source
    assert "UsdLux.DistantLight" not in lighting_source
    assert "FLIP_TABLE_LOCAL_SUN_DISTANCE_M" in overlay
    assert "GetLightLinkCollectionAPI" in lighting_source
    assert "GetShadowLinkCollectionAPI" in lighting_source
    assert "FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M" in overlay
    assert "_randomize_contact_materials" in overlay
    assert "_rl_randomization_level" in overlay
    assert "_curriculum_range" in overlay
    assert "_curriculum_choices" in overlay
    assert "_randomize_policy_camera_mounts" in overlay
    assert "FLIP_TABLE_RL_CAMERA_POSITION_JITTER_M" in overlay
    assert '"FLIP_TABLE_RANDOMIZE_LIGHTING", True' in overlay
    assert '"Scene/Table278/Table278/Collisions/Table278_Collider5"' in overlay
    assert "ComputeBoundMaterial(\"physics\")" in overlay
    assert "Usd.TraverseInstanceProxies()" in overlay
    assert 'collision_shape_counts["workbench"] != 1' in overlay
    scene_prepare = (ROOT / "tools" / "prepare_assembled_table_scene.py").read_text()
    assert 'WORKBENCH_TOP_COLLIDER = f"{WORKBENCH_BODY}/Collisions/Table278_Collider5"' in scene_prepare
    assert 'AddAppliedSchema("PhysxMaterialAPI")' in scene_prepare
    assert '"physxMaterial:frictionCombineMode"' in scene_prepare

    g1_patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text()
    assert "patch_g1_gripper_contact_material" in g1_patch
    assert 'CONTACT_MATERIAL_PATCH_ENV = "FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL"' in g1_patch
    assert "if _env_bool(CONTACT_MATERIAL_PATCH_ENV, True):" in g1_patch
    assert "CONTACT_RANDOMIZATION_ENV" not in g1_patch
    assert g1_patch.count('"left_dex1_finger_link_1/collisions"') == 1
    assert g1_patch.count('"right_dex1_finger_link_2/collisions"') == 1


def test_overlay_joint_targets_support_partial_environment_resets() -> None:
    overlay_path = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    )
    tree = ast.parse(overlay_path.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "set_joint_position_target"
    ]

    assert calls
    assert all(any(keyword.arg == "env_ids" for keyword in call.keywords) for call in calls)


def test_overlay_reset_clears_stale_actuator_targets() -> None:
    overlay_path = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    )
    source = overlay_path.read_text()
    tree = ast.parse(source)

    event_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlipTableEvalEventCfg"
    )
    reset_assignment = next(
        node
        for node in event_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "reset_all" for target in node.targets)
    )
    assert isinstance(reset_assignment.value, ast.Call)
    assert isinstance(reset_assignment.value.func, ast.Name)
    assert reset_assignment.value.func.id == "EventTerm"
    reset_func = next(keyword.value for keyword in reset_assignment.value.keywords if keyword.arg == "func")
    assert isinstance(reset_func, ast.Name)
    assert reset_func.id == "_reset_scene_to_default_and_targets"

    reset_function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_scene_to_default_and_targets"
    )
    reset_target_calls = {
        node.func.attr
        for node in ast.walk(reset_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"set_joint_position_target", "set_joint_velocity_target"}
    }
    assert reset_target_calls == {"set_joint_position_target", "set_joint_velocity_target"}

    randomize_function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_randomize_robot_joints"
    )
    target_calls = {
        node.func.attr
        for node in ast.walk(randomize_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"set_joint_position_target", "set_joint_velocity_target"}
    }
    assert target_calls == {"set_joint_position_target", "set_joint_velocity_target"}


def test_partial_reset_preserves_other_environments_initial_table_normal() -> None:
    overlay_path = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    )
    tree = ast.parse(overlay_path.read_text())
    reset_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_flip_table_scene"
    )
    allocation_guards = []
    for node in ast.walk(reset_method):
        if not isinstance(node, ast.If):
            continue
        assigns_normal = any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_initial_table_normal"
                for target in child.targets
            )
            for child in node.body
        )
        if assigns_normal:
            allocation_guards.append(ast.unparse(node.test))

    assert len(allocation_guards) == 1
    assert "self._initial_table_normal is None" in allocation_guards[0]
    assert "self._initial_table_normal.shape != (env.num_envs, 3)" in allocation_guards[0]
    assert any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "_initial_table_normal"
            for target in node.targets
        )
        for node in ast.walk(reset_method)
    )


def test_rl_partial_reset_runner_uses_script_cli_spelling() -> None:
    runner = (
        ROOT.parents[1]
        / "model"
        / "flip_table_reinforcement_learning"
        / "run_train_in_container.sh"
    ).read_text()

    assert '--task_config flip_table_rl --num_envs "$NUM_ENVS"' in runner
    assert '--task_config flip_table_rl --num-envs "$NUM_ENVS"' not in runner

    adapter = (ROOT / "container_overlay/mdp/team_ramen_balanced_wbc_action.py").read_text()
    assert '"torso_orientation_rpy_cmd": np.zeros(' in adapter
    assert "(self.num_envs, 3)" in adapter


def test_table_body_pose_returns_stable_snapshots() -> None:
    overlay_path = (
        ROOT
        / "container_overlay"
        / "robofinals_tasks"
        / "local_auto_tasks"
        / "assemble_table_task.py"
    )
    tree = ast.parse(overlay_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_table_body_pose"
    )
    returns = [node for node in method.body if isinstance(node, ast.Return)]
    assert any(
        isinstance(return_node.value, ast.Tuple)
        and len(return_node.value.elts) == 2
        and all(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "clone"
            for call in return_node.value.elts
        )
        for return_node in returns
    )


def test_contact_pair_average_inverse_is_exact() -> None:
    solve = _load_overlay_function("_surface_values_for_average_pairs")
    hand, white, workbench = solve(0.72, 0.46, 0.64)

    assert abs((hand + white) / 2.0 - 0.72) < 1.0e-12
    assert abs((white + workbench) / 2.0 - 0.46) < 1.0e-12
    assert abs((workbench + hand) / 2.0 - 0.64) < 1.0e-12
    assert abs(hand - 0.90) < 1.0e-12
    assert abs(white - 0.54) < 1.0e-12
    assert abs(workbench - 0.38) < 1.0e-12


def test_room_domain_randomization_assets_are_tracked_and_self_contained() -> None:
    asset_root = ROOT / "assets" / "room"
    props = (asset_root / "room_props.usda").read_text(encoding="utf-8")
    for asset_name in ("Chair", "Desk", "Shelf", "Cabinet", "Crates", "Plant"):
        assert f'def Xform "{asset_name}"' in props
    assert "PhysicsRigidBodyAPI" not in props
    assert "CollisionAPI" not in props

    texture_names = (
        "oak_wood.png",
        "rough_concrete.png",
        "ceramic_tile.png",
        "industrial_vinyl.png",
        "painted_plaster.png",
        "red_brick.png",
    )
    for texture_name in texture_names:
        data = (asset_root / "textures" / texture_name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(data) > 1024

    generator = (ROOT / "tools" / "generate_room_textures.py").read_text(encoding="utf-8")
    assert "random.Random" in generator
    assert "http://" not in generator
    assert "https://" not in generator
    overlay = (ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py").read_text()
    assert "ROOM_BACKGROUND_PALETTES" in overlay
    assert "ROOM_FLOOR_PATTERNS" in overlay
    assert "ROOM_WALL_PATTERNS" in overlay
    assert "_randomize_room_background" in overlay
    assert "TileLine_" in overlay
    assert "FloorChecker_" in overlay
    assert "WallStripe_" in overlay
    assert "WallWainscot" in overlay
    assert "FrontWall" in overlay


def test_no_root_level_script_path_in_feature_docs() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "scripts/run" not in readme
    assert "python scripts/" not in readme
    assert os.path.exists(ROOT / "run_eval.sh")
    assert os.path.exists(ROOT / "run_eval_in_container.sh")


def test_policy_overlay_defines_lightweight_eval_policies() -> None:
    overlay = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text()

    assert "class NoOpPolicy" in overlay
    assert "class ScriptedJointPolicy" in overlay
    assert "class CvRuleBasedPolicy" in overlay
    assert "class RecordedJointTargetPolicy" in overlay
    assert "class RecordedWBCPolicy" in overlay
    assert "class LeRobotACTPolicy" in overlay
    assert "class FlowMatchingBCPolicy(LeRobotACTPolicy)" in overlay
    assert "class FlowMatchingRLPDPolicy(FlowMatchingBCPolicy)" in overlay
    assert "class LeRobotGrootN17Policy" in overlay
    assert "_ACT_STATE_DIM = 19" in overlay
    assert "_ACT_ACTION_DIM = 16" in overlay
    assert "_WBC_ACTION_DIM = 16" in overlay
    assert "_fk_wrist_pose" not in overlay
    assert "FLIP_TABLE_ACT_MAP_WAIST_TO_TORSO" not in overlay
    assert "FLIP_TABLE_ACT_FK_USE_WAIST" not in overlay
    assert "_to_wbc_action" in overlay
    assert "pinocchio" in overlay
    assert "task_env.step(action)" in overlay
    assert "FLIP_TABLE_SAVE_CAMERA_FRAMES" in overlay
    assert "FLIP_TABLE_CAMERA_FRAME_INDEX" in overlay
    assert "FLIP_TABLE_SAVE_ACTION_STATE_TRACE" in overlay
    assert "FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE" in overlay
    assert "FLIP_TABLE_SCRIPTED_DEBUG_EVERY" in overlay
    assert "CAMERA_SAVE_ROLES" in overlay
    assert '"observation.images.head_right": "first_person_camera"' not in overlay
    assert "simulator_object_pose_used" in overlay
    assert "_update_head_camera_calibration" in overlay
    assert "framesForwardKinematics" in overlay
    assert "media.write_image" in overlay
    assert "FLIP_TABLE_ACT_CONVERT_DEX1_HAND" in overlay
    assert "FLIP_TABLE_ACT_N_ACTION_STEPS" in overlay
    assert "FLIP_TABLE_ACT_POLICY_HZ" in overlay
    assert "FLIP_TABLE_FLOW_N_ACTION_STEPS" in overlay
    assert "self.model.sample_actions" in overlay
    assert "apply_residual_to_base(base_target, residual)" in overlay
    assert "FLIP_TABLE_ACT_SIM_CONTROL_HZ" in overlay
    assert "FLIP_TABLE_ACT_DEVICE" in overlay
    assert "FLIP_TABLE_ACT_TARGET_VELOCITY_SCALE" in overlay
    assert "FLIP_TABLE_ACT_TARGET_ACCELERATION_RAD_S2" in overlay
    assert "_dex1_joint_pos_to_policy_hand" in overlay
    assert "_policy_hand_to_dex1_command" in overlay
    assert "_GROOT_STATE_DIM = 49" in overlay
    assert "_GROOT_ACTION_DIM = 53" in overlay
    assert "predict_action_chunk" not in overlay
    assert '"global": "global_camera"' not in overlay


def test_cv_rule_based_policy_is_wired_to_pink_in_both_runners() -> None:
    for name in ("run_eval.sh", "run_eval_in_container.sh"):
        runner = (ROOT / name).read_text(encoding="utf-8")
        assert "CvRuleBasedPolicy" in runner
        assert "FLIP_TABLE_USE_PINK_EEF_ACTION=true" in runner
        assert "FLIP_TABLE_CV_REDETECT_INTERVAL_STEPS" in runner
        assert "FLIP_TABLE_CV_REDETECT_ALPHA" in runner
        assert "FLIP_TABLE_CV_DEX1_GRASP_BLOCK_THRESHOLD_RAD" in runner
        assert "FLIP_TABLE_CV_GRASP_LOSS_LIMIT_STEPS" in runner
        assert "FLIP_TABLE_CV_FIRST_ROLL_ADVANCE_INTERVAL" in runner
        assert "episode_000000_motion" not in runner
        assert "policy/cv_rule_based" in runner


def test_cv_rule_based_grasp_verification_uses_only_dex1_encoders(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_cv_grasp_encoder_test")
    policy = module.CvRuleBasedPolicy.__new__(module.CvRuleBasedPolicy)
    policy.dex1_grasp_block_threshold_rad = -0.017
    joint_pos = np.zeros((1, 33), dtype=np.float32)
    joint_pos[0, 29:33] = (-0.02, -0.02, -0.01, -0.02)

    fingers, blocked = policy._dex1_grasp_observation(
        {"embodiment_general_obs": {"joint_pos": joint_pos}}
    )

    np.testing.assert_allclose(fingers[0], (-0.02, -0.02))
    np.testing.assert_allclose(fingers[1], (-0.01, -0.02))
    assert blocked == (False, False)


def test_cv_rule_based_holds_right_hand_clear_during_first_roll() -> None:
    overlay = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )

    assert "late_left_flip" not in overlay


def test_groot_runtime_isolated_and_wired_into_both_runners() -> None:
    requirements = (ROOT / "groot_runtime" / "requirements.txt").read_text(encoding="utf-8")
    setup = (ROOT / "groot_runtime" / "setup_runtime.sh").read_text(encoding="utf-8")
    server = (ROOT / "groot_runtime" / "groot_inference_server.py").read_text(encoding="utf-8")
    runners = [
        (ROOT / "run_eval.sh").read_text(encoding="utf-8"),
        (ROOT / "run_eval_in_container.sh").read_text(encoding="utf-8"),
    ]

    assert "lerobot[groot]==0.6.0" in requirements
    assert "torch==2.10.0" in requirements
    assert "torchvision==0.25.0" in requirements
    assert "--torch-backend cu128" in setup
    assert "--system-site-packages" not in setup
    assert "predict_action_chunk(processed)" in server
    assert "decoded_chunk = self.postprocessor(normalized_chunk)" in server
    assert "select_action" not in server
    for runner in runners:
        assert "LeRobotGrootN17Policy" in runner
        assert "groot_inference_server.py" in runner
        assert "FLIP_TABLE_GROOT_N_ACTION_STEPS" in runner
        assert "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION=true" in runner
        assert "awk '/^policy_name:/" in runner


def test_host_runner_resolves_default_groot_policy_and_requires_checkpoint(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("FLIP_TABLE_POLICY_NAME", None)
    env.pop("FLIP_TABLE_POLICY_CHECKPOINT", None)
    env["FLIP_TABLE_IPC_LOCK_PATH"] = str(tmp_path / "isolated_ipc.lock")
    result = subprocess.run(
        ["bash", str(ROOT / "run_eval.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "LeRobotGrootN17Policy requires FLIP_TABLE_POLICY_CHECKPOINT" in result.stderr


def test_act_adapter_converts_between_dataset_and_dex1_hand_scales(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")

    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))

    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)

    overlay_path = ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py"
    spec = importlib.util.spec_from_file_location("flip_table_eval_policy_for_test", overlay_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    open_close = torch.tensor([policy._DEX1_OPEN_POS, policy._DEX1_CLOSE_POS], dtype=torch.float32)

    assert torch.allclose(
        policy._dex1_joint_pos_to_policy_hand(open_close),
        torch.tensor([policy._POLICY_HAND_OPEN, policy._POLICY_HAND_CLOSED]),
        atol=1e-6,
    )
    assert torch.allclose(
        policy._policy_hand_to_dex1_command(torch.tensor([policy._POLICY_HAND_OPEN, policy._POLICY_HAND_CLOSED])),
        torch.tensor([-1.0, 1.0]),
        atol=1e-6,
    )


def test_runtime_control_rate_reads_remote_scalars_without_pickling_sim_cfg(monkeypatch) -> None:
    import pytest

    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))
    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    class Service:
        values = {
            "unwrapped.cfg.sim.dt": 0.005,
            "unwrapped.cfg.decimation": 4,
        }

        def __init__(self) -> None:
            self.paths = []

        def getattr_value(self, path):
            self.paths.append(path)
            return self.values[path]

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    module = _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        "flip_table_eval_policy_remote_rate_test",
    )
    service = Service()
    task_env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(_svc=service))

    assert module._runtime_control_hz(task_env) == 50.0
    assert service.paths == ["unwrapped.cfg.sim.dt", "unwrapped.cfg.decimation"]


def test_act_adapter_prefers_named_dex1_joints_over_vendor_gripper_pos(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))

    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    module = _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        "flip_table_eval_policy_hand_source_test",
    )

    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.device = torch.device("cpu")
    policy.input_state_dim = 19
    policy.state_source = "joint_pos"
    policy.state_indices = []
    policy.gripper_state_source = "gripper_pos"
    policy.convert_dex1_hand = True
    joint_pos = torch.zeros((1, 33), dtype=torch.float32)
    joint_pos[0, 29:31] = torch.tensor([0.01659, 0.01659])
    joint_pos[0, 31:33] = torch.tensor([0.02437, 0.02437])
    merged = {
        "joint_pos": joint_pos,
        "gripper_pos": torch.tensor([[0.0245, 0.0245]]),
    }

    state = policy._state_tensor(merged)
    expected = torch.tensor([3.6998, 4.4868], dtype=torch.float32)
    assert torch.allclose(state[17:19], expected, atol=1e-3)


def test_camera_resolution_rejects_ambiguous_or_invalid_observations(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_camera_contract_test")
    observations = {
        "left_hand_camera_depth": torch.zeros((1, 480, 640, 1)),
        "left_hand_camera_rgb": torch.zeros((1, 480, 640, 4)),
    }

    assert module._resolve_camera_rgb_key(observations, "left_hand_camera") == "left_hand_camera_rgb"
    with pytest.raises(ValueError, match="batch size 1"):
        module._camera_image_uint8(torch.zeros((2, 480, 640, 4)))
    with pytest.raises(ValueError, match="NaN or Inf"):
        module._camera_image_uint8(torch.full((1, 480, 640, 3), float("nan")))
    with pytest.raises(ValueError, match="exactly one RGB"):
        module._resolve_camera_rgb_key(
            {
                "camera_color_rgb": torch.zeros((1, 480, 640, 3)),
                "camera_linear_rgb": torch.zeros((1, 480, 640, 3)),
            },
            "camera",
        )


def test_act_state_adapter_fails_closed_on_unknown_or_incomplete_state(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_act_state_contract_test")
    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.device = torch.device("cpu")
    policy.input_state_dim = 19
    policy.state_source = "joint_pos"
    policy.state_indices = []
    policy.gripper_state_source = "gripper_pos"
    policy.convert_dex1_hand = True

    with pytest.raises(ValueError, match="is missing"):
        policy._state_tensor({})
    with pytest.raises(ValueError, match="named 33-D"):
        policy._state_tensor({"joint_pos": torch.zeros((1, 20))})
    with pytest.raises(ValueError, match="requires left/right Dex1"):
        policy._state_tensor({"joint_pos": torch.zeros((1, 29))})


def test_act_action_adapter_rejects_incompatible_dimensions(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_act_action_contract_test")
    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.device = torch.device("cpu")
    policy.output_action_dim = 16
    policy.convert_dex1_hand = True

    with pytest.raises(ValueError, match="ACT output has 15"):
        policy._to_wbc_action(torch.zeros((1, 15)), 16)
    with pytest.raises(ValueError, match="canonical 16-D"):
        policy._to_wbc_action(torch.zeros((1, 16)), 17)


def test_act_evaluator_rejects_multiple_environments(monkeypatch) -> None:
    import pytest

    module = _load_policy_module(monkeypatch, "flip_table_act_single_env_contract_test")
    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)

    with pytest.raises(ValueError, match="requires num_envs=1"):
        policy.eval(
            object(),
            {},
            {"env_cfg": {"num_envs": 2}, "actions_dim": 16, "time_out_limit": 1},
            None,
        )


def test_act_checkpoint_loading_is_strict() -> None:
    source = (
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py"
    ).read_text(encoding="utf-8")

    assert 'load_kwargs = {"strict": True}' in source
    assert "ACT checkpoint does not exactly match its config" in source


def test_act_checkpoint_contract_requires_exact_cameras_and_normalization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pytest

    module = _load_policy_module(monkeypatch, "flip_table_act_checkpoint_contract_test")
    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.checkpoint = str(tmp_path)
    policy.input_state_dim = 19
    policy.output_action_dim = 16
    policy.camera_mapping = {
        "observation.images.head_left": "first_person_camera",
        "observation.images.left_wrist": "left_hand_camera",
        "observation.images.right_wrist": "right_hand_camera",
    }
    config = {
        "type": "act",
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
        "input_features": {
            "observation.state": {"shape": [19]},
            **{key: {"shape": [3, 480, 640]} for key in policy.camera_mapping},
        },
        "output_features": {"action": {"shape": [16]}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    policy._load_checkpoint_feature_dims()
    config["input_features"]["observation.images.global"] = {"shape": [3, 480, 640]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="cameras must be exactly"):
        policy._load_checkpoint_feature_dims()


def test_act_normalizer_contract_rejects_missing_or_wrong_stats(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_act_stats_contract_test")
    required = {
        "observation.state.mean": (19,),
        "observation.state.std": (19,),
    }

    with pytest.raises(ValueError, match="missing keys"):
        module.LeRobotACTPolicy._validate_stats({}, required, "preprocessor")
    with pytest.raises(ValueError, match="must have shape"):
        module.LeRobotACTPolicy._validate_stats(
            {
                "observation.state.mean": torch.zeros(18),
                "observation.state.std": torch.ones(19),
            },
            required,
            "preprocessor",
        )


def test_act_processor_state_is_resolved_from_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pytest

    module = _load_policy_module(monkeypatch, "flip_table_act_processor_manifest_test")
    state_path = tmp_path / "custom-normalizer.safetensors"
    state_path.write_bytes(b"state")
    manifest_path = tmp_path / "policy_preprocessor.json"
    manifest_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": state_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved, eps = module.LeRobotACTPolicy._resolve_processor_state(
        tmp_path,
        manifest_path.name,
        "normalizer_processor",
    )
    assert resolved == state_path.resolve()
    assert eps == pytest.approx(1e-8)

    manifest_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": "../outside.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="remain inside"):
        module.LeRobotACTPolicy._resolve_processor_state(
            tmp_path,
            manifest_path.name,
            "normalizer_processor",
        )


def test_act_target_safety_enforces_position_velocity_and_acceleration(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))
    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    module = _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        "flip_table_eval_policy_safety_test",
    )

    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.device = torch.device("cpu")
    policy.policy_hz = 10.0
    policy.target_velocity_scale = 1.0
    policy.target_acceleration_rad_s2 = 1.0
    policy._pin_model = types.SimpleNamespace(
        lowerPositionLimit=[-1.0] * 17,
        upperPositionLimit=[1.0] * 17,
        velocityLimit=[20.0] * 17,
    )
    policy._pin_joint_indices = {name: index for index, name in enumerate(policy._UPPER_BODY_JOINT_NAMES[:17])}
    policy._last_safe_target = None
    policy._last_safe_velocity = None
    policy._last_safe_hand_target = None
    policy._last_safe_hand_velocity = None
    policy._safety_clip_count = 0

    current = torch.zeros((1, 19))
    raw = torch.tensor([[10.0] * 14 + [99.0, -99.0]])
    safe = policy._clip_policy_joint_targets(raw, current)

    # dt=0.1, velocity=20 rad/s, but acceleration=1 rad/s^2 from rest.
    assert torch.all(safe[0, :14].abs() <= 0.01 + 1e-6)
    assert torch.all((safe[0, :14] >= -1.0) & (safe[0, :14] <= 1.0))
    assert torch.all(
        (safe[0, 14:16] >= policy._POLICY_HAND_MIN)
        & (safe[0, 14:16] <= policy._POLICY_HAND_MAX)
    )
    # The hand command is acceleration-limited as well as range-clipped.
    assert torch.allclose(safe[0, 14:16], torch.tensor([2.0, 0.0]))
    assert policy._safety_clip_count > 0


def test_groot_contract_matches_training_mapping(monkeypatch) -> None:
    import pytest

    pytest.importorskip("torch")
    policy_module = _load_policy_module(monkeypatch, "flip_table_groot_contract_test")
    mapping = _load_module(
        ROOT.parents[1] / "model" / "subtask_policy_training" / "gr00t" / "g1_full_body_mapping.py",
        "groot_training_mapping_for_sim_contract_test",
    )
    policy = policy_module.LeRobotGrootN17Policy

    assert policy._GROOT_STATE_DIM == mapping.REAL_G1_RELATIVE_EEF_STATE_DIM == 49
    assert policy._GROOT_ACTION_DIM == mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM == 53
    assert policy._GROOT_EMBODIMENT_TAG == mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
    assert policy._GROOT_STATE_SLICES == mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES
    assert policy._GROOT_ACTION_SLICES == mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES
    assert set(policy._GROOT_REQUIRED_CAMERA_KEYS) == set(mapping.STANDARD_POLICY_VIDEO_KEYS)


def test_groot_builds_exact_real_g1_49d_state_from_named_sim_joints(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_state_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy.device = torch.device("cpu")
    policy.state_source = "joint_pos"
    policy.convert_dex1_hand = True

    joint_pos = torch.zeros((1, 33), dtype=torch.float32)
    lookup = {name: index for index, name in enumerate(policy._G1_GRIPPER_33_JOINT_ORDER)}
    for value, name in enumerate(policy._UPPER_BODY_JOINT_NAMES, start=1):
        joint_pos[0, lookup[name]] = float(value) / 10.0
    joint_pos[0, list(module._G1_LEFT_DEX1_JOINT_INDICES)] = policy._DEX1_OPEN_POS
    joint_pos[0, list(module._G1_RIGHT_DEX1_JOINT_INDICES)] = policy._DEX1_CLOSE_POS
    merged = {"joint_pos": joint_pos}

    left_eef = torch.arange(9, dtype=torch.float32) + 10
    right_eef = torch.arange(9, dtype=torch.float32) + 20

    def fake_fk(self, state, side):
        assert state.shape == (19,)
        return left_eef if side == "left" else right_eef

    policy._fk_eef_xyz_rot6d = types.MethodType(fake_fk, policy)
    state = policy._groot_state_tensor(merged)

    assert state.shape == (49,)
    assert torch.equal(state[0:9], left_eef)
    assert torch.equal(state[9:18], right_eef)
    assert state[18:25].tolist() == pytest.approx(
        module.dex1_to_hand(4.5, side="left", kind="state")
    )
    assert state[25:32].tolist() == pytest.approx(
        module.dex1_to_hand(0.0, side="right", kind="state")
    )
    assert torch.allclose(state[32:39], torch.arange(4, 11, dtype=torch.float32) / 10)
    assert torch.allclose(state[39:46], torch.arange(11, 18, dtype=torch.float32) / 10)
    assert torch.allclose(state[46:49], torch.arange(1, 4, dtype=torch.float32) / 10)


def test_groot_encode_obs_excludes_global_and_head_right_cameras(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_camera_input_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy.device = torch.device("cpu")
    policy.instruction = "Flip the table."
    policy._groot_state_tensor = lambda merged: torch.zeros(49)
    policy._camera_history = __import__("collections").deque(maxlen=64)
    policy._current_sim_time = 0.0
    frame = torch.zeros((1, 480, 640, 3), dtype=torch.uint8)
    observation = {
        "policy": {
            "first_person_camera_rgb": frame,
            "left_hand_camera_rgb": frame,
            "right_hand_camera_rgb": frame,
            "global_camera_rgb": torch.full_like(frame, 1),
            "head_right_camera_rgb": torch.full_like(frame, 2),
        },
        "embodiment_general_obs": {"joint_pos": torch.zeros((1, 33))},
    }

    payload = policy.encode_obs(observation)

    assert set(payload) == {"state", "task", "head_left", "left_wrist", "right_wrist"}
    assert payload["head_left"].shape == (2, 480, 640, 3)
    assert payload["head_left"].dtype == np.uint8


def test_groot_fk_uses_unitree_wrist_yaw_plus_five_centimeter_eef(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_fk_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy.device = torch.device("cpu")
    policy._pin_model = types.SimpleNamespace()
    placement = types.SimpleNamespace(rotation=np.eye(3), translation=np.asarray([1.0, 2.0, 3.0]))
    policy._pin_data = types.SimpleNamespace(oMf={4: placement, 5: placement})
    policy._pin_q = np.zeros(17, dtype=np.float64)
    policy._pin_joint_indices = {
        name: index for index, name in enumerate(policy._UPPER_BODY_JOINT_NAMES)
    }
    policy._pin_frame_ids = {"left": 4, "right": 5}
    policy._pin = types.SimpleNamespace(framesForwardKinematics=lambda *args: None)
    policy._ensure_fk_model = lambda: None

    pose = policy._fk_eef_xyz_rot6d(torch.zeros(19), "left")

    assert torch.allclose(pose[:3], torch.tensor([1.05, 2.0, 3.0]), atol=1e-7)
    assert torch.equal(pose[3:], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))


def test_groot_decoded_53d_maps_only_real_upper_body_targets(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_action_mapping_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy.device = torch.device("cpu")
    decoded = torch.arange(2 * 53, dtype=torch.float32).reshape(2, 53)
    decoded[:, 18:25] = torch.tensor(
        module.dex1_to_hand(1.25, side="left", kind="action")
    )
    decoded[:, 25:32] = torch.tensor(
        module.dex1_to_hand(3.75, side="right", kind="action")
    )

    targets = policy._decoded_chunk_to_joint_targets(decoded)

    assert targets.shape == (2, 16)
    assert torch.equal(targets[:, 0:7], decoded[:, 32:39])
    assert torch.equal(targets[:, 7:14], decoded[:, 39:46])
    assert targets[:, 14].tolist() == pytest.approx([1.25, 1.25])
    assert targets[:, 15].tolist() == pytest.approx([3.75, 3.75])
    selected = set(range(18, 46))
    assert not selected.intersection(range(0, 18))
    assert not selected.intersection(range(49, 53))


def test_groot_runtime_postprocesses_the_entire_relative_chunk_once(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    server = _load_module(
        ROOT / "groot_runtime" / "groot_inference_server.py",
        "flip_table_groot_runtime_full_chunk_test",
    )
    runtime = server.GrootRuntime.__new__(server.GrootRuntime)
    runtime.torch = torch
    calls: list[tuple[str, tuple[int, ...]]] = []

    class Preprocessor:
        def __call__(self, raw):
            calls.append(("pre", tuple(raw["observation.state"].shape)))
            return {"processed": torch.ones(1)}

    class Model:
        def predict_action_chunk(self, processed):
            assert set(processed) == {"processed"}
            calls.append(("predict", ()))
            return torch.arange(40 * 53, dtype=torch.float32).reshape(1, 40, 53)

    class Postprocessor:
        def __call__(self, chunk):
            calls.append(("post", tuple(chunk.shape)))
            return chunk + 1000

    runtime.preprocessor = Preprocessor()
    runtime.model = Model()
    runtime.postprocessor = Postprocessor()
    request = {
        "state": np.zeros(49, dtype=np.float32),
        "head_left": np.zeros((2, 480, 640, 3), dtype=np.uint8),
        "left_wrist": np.zeros((2, 480, 640, 3), dtype=np.uint8),
        "right_wrist": np.zeros((2, 480, 640, 3), dtype=np.uint8),
        "task": np.asarray("flip table"),
    }

    decoded, normalized, elapsed = runtime.predict(request)

    assert calls == [("pre", (49,)), ("predict", ()), ("post", (1, 40, 53))]
    assert decoded.shape == normalized.shape == (40, 53)
    assert np.allclose(decoded, normalized + 1000)
    assert elapsed >= 0


def test_groot_runtime_reset_reseeds_all_inference_rngs() -> None:
    torch = pytest.importorskip("torch")
    server = _load_module(
        ROOT / "groot_runtime" / "groot_inference_server.py",
        "flip_table_groot_runtime_seed_test",
    )
    runtime = server.GrootRuntime.__new__(server.GrootRuntime)
    runtime.torch = torch
    runtime.model = types.SimpleNamespace(reset=lambda: None)
    runtime.current_seed = 0

    runtime.reset(95001)
    first = (
        server.random.random(),
        float(server.np.random.random()),
        float(torch.rand(())),
    )
    runtime.reset(95001)
    second = (
        server.random.random(),
        float(server.np.random.random()),
        float(torch.rand(())),
    )

    assert first == pytest.approx(second)
    assert runtime.current_seed == 95001
    with pytest.raises(ValueError, match="uint32"):
        runtime.reset(2**32)


def test_groot_policy_assigns_independent_reproducible_episode_seeds(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_groot_episode_seed_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy._inference_seed_base = 95001
    policy._inference_episode_index = 0

    assert [policy._next_inference_episode_seed() for _ in range(3)] == [
        95001,
        95002,
        95003,
    ]


def test_groot_client_requires_seed_acknowledgement(monkeypatch) -> None:
    module = _load_policy_module(monkeypatch, "flip_table_groot_seed_ack_test")
    client = module._GrootInferenceClient.__new__(module._GrootInferenceClient)
    requests: list[dict[str, np.ndarray]] = []

    def request(**arrays):
        requests.append(arrays)
        return {"ok": np.asarray([1]), "seed": arrays["seed"].copy()}

    client._request = request
    client.reset(95001)

    assert str(requests[0]["kind"]) == "reset"
    assert requests[0]["seed"].dtype == np.uint64
    assert requests[0]["seed"].tolist() == [95001]


def test_groot_runtime_validates_native_relative_processor_contract() -> None:
    import pytest

    pytest.importorskip("torch")
    server = _load_module(
        ROOT / "groot_runtime" / "groot_inference_server.py",
        "flip_table_groot_runtime_processor_contract_test",
    )
    processor = pytest.importorskip("lerobot.policies.groot.processor_groot")
    GrootN17ActionDecodeStep = processor.GrootN17ActionDecodeStep
    GrootN17PackInputsStep = processor.GrootN17PackInputsStep

    state_dims = (9, 9, 7, 7, 7, 7, 3)
    action_dims = state_dims + (1, 3)
    state_stats = {
        key: {"mean": [0.0] * dim} for key, dim in zip(server.STATE_GROUPS, state_dims)
    }
    action_stats = {
        key: {"mean": [0.0] * dim} for key, dim in zip(server.ACTION_GROUPS, action_dims)
    }
    relative_stats = {
        key: {"mean": [[0.0] * action_dims[server.ACTION_GROUPS.index(key)] for _ in range(40)]}
        for key in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_arm", "right_arm")
    }
    action_configs = [
        {
            "rep": expected[0],
            "type": expected[1],
            "format": expected[2],
            "state_key": expected[3],
        }
        for expected in server.ACTION_CONFIGS.values()
    ]
    modality = {
        "state": {"modality_keys": list(server.STATE_GROUPS)},
        "action": {
            "modality_keys": list(server.ACTION_GROUPS),
            "action_configs": action_configs,
        },
    }
    pack = GrootN17PackInputsStep(
        action_horizon=40,
        valid_action_horizon=40,
        video_horizon=2,
        embodiment_tag=server.EMBODIMENT_TAG,
        video_modality_keys=["head_left", "left_wrist", "right_wrist"],
    )
    decode = GrootN17ActionDecodeStep(
        env_action_dim=53,
        raw_stats={
            "state": state_stats,
            "action": action_stats,
            "relative_action": relative_stats,
        },
        modality_config=modality,
        use_relative_action=True,
        pack_step=pack,
    )
    preprocessor = types.SimpleNamespace(steps=[pack])
    postprocessor = types.SimpleNamespace(steps=[decode])

    server.validate_processor_contract(preprocessor, postprocessor, required_horizon=16)
    decode.pack_step = None
    with pytest.raises(ValueError, match="not connected"):
        server.validate_processor_contract(preprocessor, postprocessor, required_horizon=16)


def test_groot_unix_socket_protocol_preserves_full_chunks(tmp_path: Path, monkeypatch) -> None:
    import socket
    import threading

    import pytest

    pytest.importorskip("torch")
    policy_module = _load_policy_module(monkeypatch, "flip_table_groot_socket_client_test")
    server_module = _load_module(
        ROOT / "groot_runtime" / "groot_inference_server.py",
        "flip_table_groot_socket_server_test",
    )
    socket_path = tmp_path / "groot.sock"
    ready = threading.Event()
    received: dict[str, np.ndarray] = {}

    def server_thread() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(socket_path))
            listener.listen(1)
            ready.set()
            connection, _ = listener.accept()
            with connection:
                request = server_module.receive_archive(connection)
                assert request is not None
                received.update(request)
                server_module.send_archive(
                    connection,
                    ok=np.asarray([1], dtype=np.uint8),
                    action=np.arange(2 * 53, dtype=np.float32).reshape(2, 53),
                    normalized_action=np.zeros((2, 53), dtype=np.float32),
                    inference_seconds=np.asarray([0.125], dtype=np.float64),
                )

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    assert ready.wait(timeout=2)
    client = policy_module._GrootInferenceClient(socket_path, timeout_seconds=2)
    action, normalized, elapsed = client.predict(
        {
            "state": np.zeros(49, dtype=np.float32),
            "head_left": np.zeros((2, 480, 640, 3), dtype=np.uint8),
            "left_wrist": np.zeros((2, 480, 640, 3), dtype=np.uint8),
            "right_wrist": np.zeros((2, 480, 640, 3), dtype=np.uint8),
            "task": np.asarray("Flip the table."),
        }
    )
    client.close()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert str(received["kind"]) == "predict"
    assert action.shape == normalized.shape == (2, 53)
    assert action[-1, -1] == pytest.approx(105.0)
    assert elapsed == pytest.approx(0.125)


def test_groot_checkpoint_contract_rejects_extra_policy_camera(tmp_path: Path, monkeypatch) -> None:
    import pytest

    pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_checkpoint_test")
    server = _load_module(
        ROOT / "groot_runtime" / "groot_inference_server.py",
        "flip_table_groot_server_checkpoint_test",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        (checkpoint / name).write_bytes(b"fixture")
    config = {
        "type": "groot",
        "model_version": "n1.7",
        "embodiment_tag": "real_g1_relative_eef_relative_joints",
        "use_relative_actions": True,
        "relative_exclude_joints": ["hand", "waist", "base_height", "navigate"],
        "chunk_size": 40,
        "max_state_dim": 132,
        "max_action_dim": 132,
        "input_features": {
            "observation.state": {"shape": [49]},
            "observation.images.head_left": {"shape": [3, 480, 640]},
            "observation.images.left_wrist": {"shape": [3, 480, 640]},
            "observation.images.right_wrist": {"shape": [3, 480, 640]},
        },
        "output_features": {"action": {"shape": [53]}},
    }
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert module.LeRobotGrootN17Policy._validate_checkpoint_contract(checkpoint) == config
    assert server.validate_checkpoint(checkpoint) == config
    config["type"] = "furniture_groot"
    config["valid_action_dim"] = 46
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="base_model_revision"):
        module.LeRobotGrootN17Policy._validate_checkpoint_contract(checkpoint)
    with pytest.raises(ValueError, match="base_model_revision"):
        server.validate_checkpoint(checkpoint)
    config["base_model_revision"] = (
        "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    )
    config["base_model_path"] = "nvidia/GR00T-N1.7-3B"
    config.update(
        {
            "tune_llm": False,
            "tune_visual": False,
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_vlln": True,
            "tune_top_llm_layers": 0,
        }
    )
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    from model.subtask_policy_training.gr00t.dex1_hand_synergy import ASSET_PATH
    from model.subtask_policy_training.gr00t.n17_contract import (
        DATASET_REPO_ID,
        DATASET_REVISION,
        DEX1_SYNERGY_SHA256,
        EXPECTED_SHA256,
    )

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (checkpoint / "dex1_g1_synergy.json").write_bytes(ASSET_PATH.read_bytes())
    eef_fk_audit = checkpoint / "eef_fk_audit.json"
    eef_fk_audit.write_text(
        json.dumps(
            {
                "source_repo_id": DATASET_REPO_ID,
                "source_revision": DATASET_REVISION,
                "pass": True,
                "action_fk_residual_pass": True,
                "frame_assignment_pass": True,
                "coverage": {"episode_count": 174},
                "temporal_alignment": {
                    "pass": True,
                    "selected_offset_frames": 0,
                },
                "training_contract": {
                    "teacher_pair_status": (
                        "compatible_with_expected_ik_realization_residual"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "training_run_record.json").write_text(
        json.dumps(
            {
                "contract": {
                    "repo_id": "nvidia/GR00T-N1.7-3B",
                    "revision": config["base_model_revision"],
                    "sha256": EXPECTED_SHA256,
                },
                "dex1_adapter": {"sha256": DEX1_SYNERGY_SHA256},
                "training_scope": {
                    "config_flags": {
                        "tune_llm": False,
                        "tune_visual": False,
                        "tune_projector": True,
                        "tune_diffusion_model": True,
                        "tune_vlln": True,
                        "tune_top_llm_layers": 0,
                    },
                    "frozen": ["llm", "visual"],
                    "trainable": [
                        "projector",
                        "diffusion_model",
                        "vlln",
                        "progress_head_when_enabled",
                    ],
                },
                "eef_fk_audit": {
                    "sha256": sha256(eef_fk_audit),
                    "source_repo_id": DATASET_REPO_ID,
                    "source_revision": DATASET_REVISION,
                    "episode_count": 174,
                    "action_fk_residual_pass": True,
                    "frame_assignment_pass": True,
                    "selected_offset_frames": 0,
                    "teacher_pair_status": (
                        "compatible_with_expected_ik_realization_residual"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "training_manifest.json").write_text(
        json.dumps(
            {
                "dataset": {
                    "repo_id": DATASET_REPO_ID,
                    "revision": DATASET_REVISION,
                    "counts": {"train": 139, "validation": 17, "test": 18},
                },
                "contract": {
                    "logical_state_dim": 49,
                    "logical_action_dim": 53,
                    "packed_state_dim": 132,
                    "packed_action_dim": 132,
                    "valid_action_dim": 46,
                    "action_horizon": 40,
                    "policy_cameras": ["head_left", "left_wrist", "right_wrist"],
                    "head_right_used": False,
                    "progress_in_action": False,
                    "progress_head_shape": None,
                },
                "checkpoint": {
                    "model_safetensors_sha256": sha256(
                        checkpoint / "model.safetensors"
                    ),
                    "config_sha256": sha256(checkpoint / "config.json"),
                },
                "candidate_selection": {"selection_data": "validation_only"},
                "wandb_url": "https://wandb.ai/test/project/runs/test",
            }
        ),
        encoding="utf-8",
    )
    # Finalized Furniture-GR00T provenance is covered by the dedicated
    # finalization tests. Keep this fixture focused on the shared camera gate.
    config["type"] = "groot"
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert module.LeRobotGrootN17Policy._validate_checkpoint_contract(checkpoint) == config
    assert server.validate_checkpoint(checkpoint) == config
    config["input_features"]["observation.images.global"] = {"shape": [3, 480, 640]}
    (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly head_left"):
        module.LeRobotGrootN17Policy._validate_checkpoint_contract(checkpoint)
    with pytest.raises(ValueError, match="exactly head_left"):
        server.validate_checkpoint(checkpoint)


def test_groot_policy_clock_advances_30_targets_during_50_sim_steps(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_groot_policy_clock_test")
    policy = module.LeRobotGrootN17Policy.__new__(module.LeRobotGrootN17Policy)
    policy.device = torch.device("cpu")
    policy.policy_hz = 30.0
    policy.sim_control_hz = 50.0
    policy._ACT_ACTION_DIM = 16
    reset_seeds: list[int] = []
    policy._client = types.SimpleNamespace(
        reset=lambda seed=None: reset_seeds.append(seed)
    )
    policy._inference_seed_base = 95001
    policy._inference_episode_index = 0
    policy.n_action_steps = 10
    policy._temporal_ensemble = module.PhysicalTargetTemporalEnsembler(decay_lambda=-0.1)
    policy._camera_history = __import__("collections").deque(maxlen=64)
    policy._capture_camera_history = lambda *args, **kwargs: None
    policy._last_env_action = None
    policy._last_safe_target = None
    policy._last_safe_velocity = None
    policy._last_safe_joint_target = None
    policy._last_raw_action = None
    policy._last_decoded_action = None
    policy._last_normalized_chunk = None
    policy._last_decoded_chunk = None
    policy._last_inference_seconds = None
    policy._last_groot_state = torch.zeros(49)
    policy._safety_clip_count = 0
    policy.add_video_frame = lambda *args, **kwargs: None
    policy._current_joint_target_state = lambda merged: torch.zeros(19)
    policy._groot_state_tensor = lambda merged: torch.zeros(49)
    policy._clip_policy_joint_targets = lambda target, current: target
    policy._to_wbc_action = lambda target, dim: target

    def predict_and_add_chunk(observation, *, origin_step):
        policy._policy_inference_count += 1
        policy._last_inference_seconds = 0.01
        policy._last_normalized_chunk = torch.zeros(40, 53)
        policy._last_decoded_chunk = torch.zeros(40, 53)
        policy._temporal_ensemble.add_chunk(
            origin_step=origin_step,
            absolute_targets=np.full((40, 16), float(origin_step)),
        )

    policy._predict_and_add_chunk = predict_and_add_chunk
    observation = {
        "policy": {},
        "embodiment_general_obs": {"joint_pos": torch.zeros((1, 33))},
    }

    class Env:
        def __init__(self):
            self.actions = []

        def step(self, action):
            self.actions.append(action.clone())
            return observation, None, torch.tensor([False]), None, {}

    env = Env()
    result = policy.eval(
        env,
        observation,
        {
            "actions_dim": 16,
            "time_out_limit": 50,
            "record_camera": [],
            "env_cfg": {"num_envs": 1},
        },
        None,
    )

    assert result.tolist() == [False]
    assert policy._action_advance_count == 30
    assert policy._policy_inference_count == 3
    assert reset_seeds == [95001]
    assert len(env.actions) == 50
    assert all(action.shape == (1, 16) for action in env.actions)


def test_act_policy_clock_advances_30_targets_and_tracks_chunk_inference(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_act_policy_clock_test")
    policy = module.LeRobotACTPolicy.__new__(module.LeRobotACTPolicy)
    policy.device = torch.device("cpu")
    policy.policy_hz = 30.0
    policy.sim_control_hz = 50.0
    policy.input_state_dim = 19
    policy._WBC_ACTION_DIM = 16
    policy._last_normalized_action = None
    policy._last_raw_action = None
    policy._last_wbc_action = None
    action_calls = 0
    policy.debug_steps = 0
    policy.add_video_frame = lambda *args, **kwargs: None
    policy._state_tensor = lambda merged: torch.zeros(19)
    policy.encode_obs = lambda observation: {}

    def get_action(observation, action_dim, current_state):
        nonlocal action_calls
        policy._last_model_inference = action_calls % 10 == 0
        action_calls += 1
        action = torch.zeros((1, action_dim))
        policy._last_raw_action = action
        policy._last_wbc_action = action
        return action

    policy.get_action = get_action
    observation = {
        "policy": {},
        "embodiment_general_obs": {"joint_pos": torch.zeros((1, 33))},
    }

    class Env:
        def step(self, action):
            return observation, None, torch.tensor([False]), None, {}

    result = policy.eval(
        Env(),
        observation,
        {
            "actions_dim": 16,
            "time_out_limit": 50,
            "record_camera": [],
            "env_cfg": {"num_envs": 1},
        },
        None,
    )

    assert result.tolist() == [False]
    assert policy._action_advance_count == 30
    assert policy._policy_inference_count == 3


def test_flow_matching_adapter_consumes_short_chunk_before_replanning(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    module = _load_policy_module(monkeypatch, "flip_table_flow_policy_queue_test")
    policy = module.FlowMatchingBCPolicy.__new__(module.FlowMatchingBCPolicy)
    policy.device = torch.device("cpu")
    policy.n_action_steps = 2
    policy._action_queue = __import__("collections").deque()
    policy._last_model_inference = False
    policy._clip_policy_joint_targets = lambda action, current: action
    policy._to_wbc_action = lambda action, dimension: action

    class Model:
        def __init__(self):
            self.calls = 0

        def sample_actions(self, images, state):
            self.calls += 1
            return torch.stack(
                [torch.full((16,), float(index)) for index in range(4)], dim=0
            ).unsqueeze(0)

        @staticmethod
        def normalize_action(action):
            return action / 10.0

    policy.model = Model()
    observation = {
        "images": torch.zeros(1, 3, 3, 48, 64),
        "state": torch.zeros(1, 19),
    }
    current_state = torch.zeros(1, 19)

    first = policy.get_action(observation, 16, current_state)
    second = policy.get_action(observation, 16, current_state)
    third = policy.get_action(observation, 16, current_state)

    assert policy.model.calls == 2
    assert torch.all(first == 0.0)
    assert torch.all(second == 1.0)
    assert torch.all(third == 0.0)
    assert len(policy._action_queue) == 1


def test_scripted_probe_initializes_valid_wbc_quaternions(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))
    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    module = _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        "flip_table_eval_policy_scripted_action_test",
    )

    policy = module.NoOpPolicy.__new__(module.NoOpPolicy)
    policy.device = torch.device("cpu")
    action = policy._blank_action(1, 23)
    assert torch.allclose(action[0, 5:9], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(action[0, 12:16], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_joint_position_probe_holds_measured_pose_and_maps_both_dex1_hands(monkeypatch) -> None:
    import pytest

    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=lambda *args, **kwargs: None))
    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    module = _load_module(
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py",
        "flip_table_eval_policy_joint_probe_test",
    )

    joint_pos = torch.arange(33, dtype=torch.float32).unsqueeze(0) / 100.0
    joint_pos[0, 29:31] = module._DEX1_OPEN_POS
    joint_pos[0, 31:33] = module._DEX1_CLOSE_POS
    observation = {"embodiment_general_obs": {"joint_pos": joint_pos}}

    action = module._joint_position_hold_action(observation, torch.device("cpu"))

    assert action.shape == (1, 16)
    assert torch.allclose(
        action[0, :14],
        joint_pos[0, list(module._G1_ARM_JOINT_INDICES)],
    )
    assert torch.allclose(action[0, 14:16], torch.tensor([-1.0, 1.0]))


def test_camera_frame_export_saves_once_per_episode_path(monkeypatch, tmp_path: Path) -> None:
    import pytest

    torch = pytest.importorskip("torch")

    written: list[Path] = []

    def write_image(path: Path, image: object) -> None:
        written.append(Path(path))
        Path(path).write_bytes(b"png")

    monkeypatch.setitem(sys.modules, "mediapy", types.SimpleNamespace(write_image=write_image))

    policy_module = types.ModuleType("policy")
    base_module = types.ModuleType("policy.base")

    class BasePolicy:
        pass

    base_module.BasePolicy = BasePolicy
    monkeypatch.setitem(sys.modules, "policy", policy_module)
    monkeypatch.setitem(sys.modules, "policy.base", base_module)
    monkeypatch.setenv("FLIP_TABLE_SAVE_CAMERA_FRAMES", "true")
    monkeypatch.setenv("FLIP_TABLE_CAMERA_FRAME_INDEX", "10")

    overlay_path = ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py"
    spec = importlib.util.spec_from_file_location("flip_table_eval_policy_camera_export_test", overlay_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    policy = types.SimpleNamespace()
    observation = {
        "policy": {
            "left_hand_camera_rgb": torch.zeros((1, 480, 640, 3), dtype=torch.uint8),
        }
    }

    args0 = {"record_camera": ["left_hand_camera"], "save_path": str(tmp_path / "test_0")}
    args1 = {"record_camera": ["left_hand_camera"], "save_path": str(tmp_path / "test_1")}

    module._maybe_save_camera_frames(policy, observation, args0, 10)
    module._maybe_save_camera_frames(policy, observation, args0, 10)
    module._maybe_save_camera_frames(policy, observation, args1, 10)

    assert written == [
        tmp_path / "test_0" / "camera_frames" / "frame_0010" / "left_wrist_rgb.png",
        tmp_path / "test_1" / "camera_frames" / "frame_0010" / "left_wrist_rgb.png",
    ]
    assert (tmp_path / "test_0" / "camera_frames" / "frame_0010" / "metadata.json").exists()
    assert (tmp_path / "test_1" / "camera_frames" / "frame_0010" / "metadata.json").exists()


def test_g1_camera_patch_rejects_invalid_runtime_boolean(monkeypatch) -> None:
    import pytest

    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_patch_boolean_test",
    )
    monkeypatch.setenv("FLIP_TABLE_ENABLE_ROBOT_COLLISIONS", "invalid")

    with pytest.raises(ValueError, match="must be a boolean"):
        module._validate_patched_runtime_booleans()


def test_g1_patch_bootstraps_dex1_wbc_from_pristine_v1_text() -> None:
    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_pristine_bootstrap_test",
    )
    assets_source = '''
from isaaclab.actuators import IdealPDActuatorCfg

G1_GEARWBC_CFG = ArticulationCfg()

G1_WUJI_ASSET_PATH = robofinals_DATA_PATH / "assets" / "g1_wuji.usd"
'''
    g1_source = '''
import robofinals.core.mdp as mdp
from .assets_cfg import G1_GEARWBC_CFG, G1_HIGH_PD_CFG, G1_WUJI_CFG

def _newton_g1_gripper_action_cfg(side: str):
    return side

def _set_g1_hand_action_cfg(action_config, gripper_cfg, hand_action_mode: str) -> None:
    if get_strategy(get_context().physics_backend).use_newton_gripper_action():
        return

def _g1_ee_target_frame_path(side: str) -> str:
    link_name = side
    return f"{{ENV_REGEX_NS}}/Robot/{link_name}"

@configclass
class G1SceneCfg:
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
    )

class UnitreeG1ControllerDecoupledWBCEnvCfg:
    def customize_physics_cfg(self, env_cfg) -> None:
        from isaaclab_newton.physics.newton_manager_cfg import VBDSolverCfg

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        self.observation_cameras = {}

class UnitreeG1WujiEnvCfg:
    pass
'''

    patched_assets = module._ensure_gripper_asset_cfg_text(assets_source)
    patched_g1 = module._ensure_gripper_controller_text(g1_source)
    patched_assets_with_variants = module._patch_gripper_asset_variants_text(patched_assets)

    ast.parse(patched_assets)
    ast.parse(patched_assets_with_variants)
    ast.parse(patched_g1)
    assert "G1_GRIPPER_CFG = G1_GEARWBC_CFG.copy()" in patched_assets
    assert '"left_dex1_finger_joint_1": 0.0245' in patched_assets
    assert 'joint_names_expr=[".*_dex1_finger_joint_.*"]' in patched_assets
    assert patched_assets_with_variants.count("G1_GRIPPER_CFG.spawn.variants") == 1
    assert "from robofinals.core.models.grippers.dex1 import Dex1GripperCfg" in patched_g1
    assert "G1_GEARWBC_CFG, G1_GRIPPER_CFG," in patched_g1
    assert "configure_g1_hand_action_cfg(" in patched_g1
    assert "use_newton_gripper_action()" not in patched_g1
    assert 'prim_path=_g1_robot_frame_path("pelvis"),' in patched_g1
    assert "customize_g1_controller_physics_cfg(env_cfg)" in patched_g1
    assert "VBDSolverCfg" not in patched_g1
    assert "class UnitreeG1GripperControllerDecoupledWBCEnvCfg" in patched_g1
    assert "self.scene_config.robot.spawn.activate_contact_sensors = True" in patched_g1
    # One sensor config per hand; each regex covers both Dex1 finger links.
    assert patched_g1.count("force_threshold=0.1") == 2
    assert patched_g1.index("class UnitreeG1GripperController") < patched_g1.index(
        "class UnitreeG1WujiEnvCfg"
    )
    assert patched_g1.count('width=640,') == 5
    assert patched_g1.count('height=480,') == 5
    assert module._ensure_gripper_asset_cfg_text(patched_assets) == patched_assets
    legacy_assets = patched_assets.replace(
        "# FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V3",
        "# FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V1",
        1,
    )
    assert module._ensure_gripper_asset_cfg_text(legacy_assets) == patched_assets
    assert module._ensure_gripper_controller_text(patched_g1) == patched_g1


def test_g1_patch_restores_all_robot_files_from_immutable_v1(tmp_path: Path) -> None:
    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_restore_test",
    )
    mutable_root = tmp_path / "mutable"
    official_root = tmp_path / "official"
    expected: dict[str, bytes] = {}
    for index, relative in enumerate(module.OFFICIAL_V1_ROBOT_FILES):
        official_file = official_root / relative
        mutable_file = mutable_root / relative
        official_file.parent.mkdir(parents=True, exist_ok=True)
        mutable_file.parent.mkdir(parents=True, exist_ok=True)
        content = f"official-v1-{index}\n".encode()
        expected[str(relative)] = content
        official_file.write_bytes(content)
        mutable_file.write_bytes(f"stale-patch-{index}\n".encode())

    hashes = module.restore_official_v1_robot_files(mutable_root, official_root)

    assert set(hashes) == set(expected)
    for relative, content in expected.items():
        restored = mutable_root / relative
        assert restored.read_bytes() == content
        assert hashes[relative] == module._sha256(restored)


def test_g1_patch_restores_generated_usds_from_first_patch_backups(tmp_path: Path) -> None:
    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_generated_asset_restore_test",
    )
    expected: dict[str, bytes] = {}
    for index, (relative, suffix) in enumerate(module.GENERATED_ASSET_BACKUP_SPECS):
        target = tmp_path / relative
        backup = Path(str(target) + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = f"generated-v1-original-{index}\n".encode()
        expected[str(relative)] = content
        target.write_bytes(f"repository-patched-{index}\n".encode())
        backup.write_bytes(content)

    hashes = module.restore_generated_v1_assets(tmp_path)

    assert set(hashes) == set(expected)
    for relative, content in expected.items():
        restored = tmp_path / relative
        assert restored.read_bytes() == content
        assert hashes[relative] == module._sha256(restored)


def test_g1_patch_rejects_an_unreviewed_v1_source_hash(tmp_path: Path) -> None:
    import pytest

    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_source_hash_test",
    )
    relative = module.OFFICIAL_V1_ROBOT_FILES[0]
    official_file = tmp_path / "official" / relative
    mutable_file = tmp_path / "mutable" / relative
    official_file.parent.mkdir(parents=True, exist_ok=True)
    mutable_file.parent.mkdir(parents=True, exist_ok=True)
    official_file.write_text("unexpected source\n", encoding="utf-8")
    mutable_file.write_text("current source\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="official V1 source hash mismatch"):
        module.restore_official_v1_robot_files(
            tmp_path / "mutable",
            tmp_path / "official",
            expected_sha256={str(relative): "0" * 64},
        )


def test_g1_contact_patch_removes_unsupported_shape_filters() -> None:
    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_contact_filter_test",
    )
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "Declare four finger sensors plus four optional leg-side sensors" in source
    assert "PhysX GPU filtering accepts rigid-body paths" in source
    assert "white_leg_contact_0" in source
    assert "Finger ``net_forces_w`` remains the all-surface safety signal" in source

    legacy = '''
_FLIP_TABLE_WHITE_TABLE_CONTACT_FILTERS = (
    "{ENV_REGEX_NS}/Scene/Leg001/Leg001/Collisions/Leg001_Collider118",
)

@configclass
class G1SceneCfg:
    left_gripper_contact: ContactSensorCfg | None = None
    right_gripper_contact: ContactSensorCfg | None = None
    left_gripper_contact_2: ContactSensorCfg | None = ContactSensorCfg(filter_prim_paths_expr=[])
    right_gripper_contact_2: ContactSensorCfg | None = ContactSensorCfg(filter_prim_paths_expr=[])

class UnitreeG1GripperControllerDecoupledWBCEnvCfg:
    def __init__(self):
        self.scene_config.left_gripper_contact = ContactSensorCfg(
            filter_prim_paths_expr=[],
        )
        self.scene_config.right_gripper_contact = ContactSensorCfg(
            filter_prim_paths_expr=[],
        )
'''.splitlines(keepends=True)
    assert module._patch_g1_contact_sensor_fields(legacy)
    assert module._remove_unsupported_shape_contact_filters(legacy)
    migrated = "".join(legacy)
    assert "_FLIP_TABLE_WHITE_TABLE_CONTACT_FILTERS" not in migrated
    assert "Leg001_Collider118" not in migrated
    assert migrated.count("white_leg_contact_") == 4
    assert migrated.count("filter_prim_paths_expr=[]") == 4
    assert "filter_prim_paths_expr=list(" not in migrated


def test_g1_contact_patch_configures_all_four_finger_sensors() -> None:
    module = _load_module(
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py",
        "flip_table_g1_four_contact_sensor_test",
    )
    lines = '''
@configclass
class G1SceneCfg:
    left_gripper_contact: ContactSensorCfg | None = None
    right_gripper_contact: ContactSensorCfg | None = None

class UnitreeG1GripperControllerDecoupledWBCEnvCfg:
    def __init__(self):
        self.scene_config.left_gripper_contact = ContactSensorCfg(
            update_period=0.0,
            history_length=1,
        )
        self.scene_config.right_gripper_contact = ContactSensorCfg(
            update_period=0.0,
            history_length=1,
        )
'''.splitlines(keepends=True)

    assert module._patch_g1_contact_sensor_fields(lines)
    assert module._patch_gripper_contact_sensor_thresholds(lines)
    patched = "".join(lines)
    assert patched.count("force_threshold=0.1") == 4
    assert patched.count("update_period=0.0") == 4


def test_dex1_force_calibration_keeps_contact_sensors_enabled() -> None:
    policy = (
        ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py"
    ).read_text(encoding="utf-8")

    assert 'if "camera" not in str(sensor_name).lower()' in policy
    assert "sensor.cfg.update_period = 3600.0" in policy
    calibration_start = policy.index("class Dex1ForceCalibrationPolicy")
    calibration_end = policy.index("class ScriptedJointPolicy", calibration_start)
    calibration = policy[calibration_start:calibration_end]
    assert "for sensor in env.scene.sensors.values()" not in calibration
    assert '"blocker_size_m": [0.160, 0.040, 0.300]' in calibration
    assert "contact_force_max_n_by_sensor" in calibration
    assert "sustained_contact_s_by_finger" in calibration
    assert "finger_prismatic_position_range_m" in calibration
    assert "finger_fixture_coordinate_range_m" in calibration
    assert "finger_fixture_center_displacement_max_m" in calibration
    assert "finger_fixture_center_overlap_max_m" in calibration
    assert '"schema_version": "team_ramen_dex1_force_calibration/v4"' in calibration
    assert "official Dex1 collision-STL bounds plus runtime link FK" in calibration
    assert "_FINGER_COLLISION_CAD_CENTERS_M" in calibration
    assert "FINGER_COLLISION_OFFSETS" not in calibration

    scene_tool = (ROOT / "tools" / "prepare_assembled_table_scene.py").read_text(
        encoding="utf-8"
    )
    assert "UsdGeom.Cube.Define(stage, path)" in scene_tool
    assert "DEX1_FORCE_CALIBRATION_SIZE_M = (0.160, 0.040, 0.300)" in scene_tool


def test_dex1_physx_contact_patch_targets_non_instanced_source_colliders() -> None:
    patch = (
        ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py"
    ).read_text(encoding="utf-8")

    assert "def patch_g1_gripper_physx_collisions" in patch
    assert 'f"/colliders/{side}_dex1_finger_link_{finger}' in patch
    assert 'prim.AddAppliedSchema("PhysxCollisionAPI")' in patch
    assert '"physxCollision:contactOffset"' in patch
    assert "contact_attr.Set(0.002)" in patch
    assert "rest_attr.Set(0.0)" in patch
    assert "self.scene_config.robot.spawn.collision_props" not in patch


def test_g1_global_camera_patch_rewrites_vendor_config(tmp_path: Path) -> None:
    target = tmp_path / "robofinals" / "core" / "robots" / "unitree" / "g1.py"
    target.parent.mkdir(parents=True)
    assets_target = target.parent / "assets_cfg.py"
    assets_target.write_text(
        '''
G1_GRIPPER_CFG = G1_GEARWBC_CFG.copy()
G1_GRIPPER_CFG.spawn.usd_path = str(robofinals_DATA_PATH / "assets" / "g1_urdf_gripper" / "G1_GRIPPER.usd")
G1_GRIPPER_CFG.actuators.pop("hands", None)
''',
        encoding="utf-8",
    )
    original_assets_content = assets_target.read_text(encoding="utf-8")
    target.write_text(
        '''
class UnitreeG1EnvCfg:
    def __init__(self):
        self.observation_cameras = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, 0.02857542, 0.42446595),
                                                    rot=(0.26523914, -0.27106013, -0.66472446, 0.64367383),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=45.55,
                        vertical_aperture=26.61,
                        clipping_range=(0.01, 50.0),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, -0.04657542, 0.42446595),
                                                    rot=(0.26523914, -0.27106013, -0.66472446, 0.64367383),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=62,
                        vertical_aperture=39.8,
                        clipping_range=(0.01, 50.0),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
            "hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.04353, -0.01712, 0.16093),
                                                    rot=(0.27138, -0.01743, -0.70811, 0.65164),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
            "global_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/global_camera",
                    # offset=TiledCameraCfg.OffsetCfg(pos=(0.25, 0.3, 0.38),
                    #                                 rot=(-0.01853, -0.10431, 0.33021, 0.93794),
                    #                                 convention="opengl"),
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.9, 0, 0.24521),
                                                    rot=(0.56538, 0.43517, 0.42607, 0.55627),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=27.7,  # Adjusted for 60 deg FOV
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
        }

class UnitreeG1GripperControllerDecoupledWBCEnvCfg:
    def __init__(self):
        self.observation_cameras = {
            "left_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.06017, 0, 0.14333),
                                                    rot=(0.3925, -0.35855, -0.58749, 0.61012),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=62,
                        vertical_aperture=39.8,
                        clipping_range=(0.01, 50.0),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
            "first_person_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/torso_link/first_person_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, -0.00937542, 0.42446595),
                                                    rot=(0.26523914, -0.27106013, -0.66472446, 0.64367383),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=19.3,
                        focus_distance=400.0,
                        horizontal_aperture=48.53,
                        vertical_aperture=35.37,
                        clipping_range=(0.1, 1.0e5),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
            "right_hand_camera": {
                "camera_cfg": TiledCameraCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_hand_camera",
                    offset=TiledCameraCfg.OffsetCfg(pos=(0.06017, 0, 0.14333),
                                                    rot=(0.3925, -0.35855, -0.58749, 0.61012),
                                                    convention="opengl"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=24.0,
                        focus_distance=400.0,
                        horizontal_aperture=62,
                        vertical_aperture=39.8,
                        clipping_range=(0.01, 50.0),
                        lock_camera=True
                    ),
                    width=224,
                    height=224,
                    update_period=0.05,
                ),
            },
        }
        self.action_config = G1DecoupledWBCActionsCfg()
        self.action_config.left_hand_action = self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]
        self.action_config.right_hand_action = self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]
''',
        encoding="utf-8",
    )
    original_content = target.read_text(encoding="utf-8")
    patch = ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py"
    env = os.environ.copy()
    env["ROBOFINALS_ROOT"] = str(tmp_path)
    env["FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS"] = "false"
    env["FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL"] = "false"
    env["FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES"] = "false"
    env["FLIP_TABLE_ENABLE_ROBOT_COLLISIONS"] = "false"
    env["FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS"] = "false"
    env["FLIP_TABLE_CAMERA_RESOLUTION_NAMES"] = "first_person_camera,left_hand_camera,right_hand_camera,hand_camera,global_camera"

    subprocess.run([sys.executable, str(patch)], env=env, check=True, capture_output=True, text=True)
    patched_once = target.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(patch)], env=env, check=True, capture_output=True, text=True)

    assert target.read_text(encoding="utf-8") == patched_once
    patched_assets_once = assets_target.read_text(encoding="utf-8")
    assert 'G1_GRIPPER_CFG.spawn.variants = {' in patched_assets_once
    assert '    "Physics": "PhysX",' in patched_assets_once
    assert '    "Robot": "Robot",' in patched_assets_once
    assert '    "Sensor": "Sensors",' in patched_assets_once
    assert patched_assets_once.count("G1_GRIPPER_CFG.spawn.variants") == 1
    patch_text = patch.read_text(encoding="utf-8")
    assert "patch_g1_gripper_usd_materials" not in patch_text
    assert "FLIP_TABLE_PATCH_G1_GRIPPER_MATERIALS" not in patch_text
    assert "patch_g1_gripper_material_binding_api" in patch_text
    assert "patch_g1_gripper_unitree_material_values" in patch_text
    assert 'GetRelationship("material:binding")' in patch_text
    assert "UsdShade.MaterialBindingAPI.Apply(prim)" in patch_text
    assert "MATERIAL_BINDING_BACKUP_SUFFIX" in patch_text
    assert "UNITREE_G1_PBR_MATERIALS" in patch_text
    assert Path(str(target) + ".original_flip_table_global_camera").exists()
    assert Path(str(target) + ".original_flip_table_global_camera").read_text(encoding="utf-8") == original_content
    assert Path(str(assets_target) + ".original_flip_table_gripper_variants").exists()
    assert (
        Path(str(assets_target) + ".original_flip_table_gripper_variants").read_text(encoding="utf-8")
        == original_assets_content
    )
    assert 'prim_path="{ENV_REGEX_NS}/global_camera",' in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/torso_link/global_camera",' not in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_hand_camera",' in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/torso_link/right_hand_camera",' in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_hand_camera",' in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/hand_camera",' in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/left_dex1_base_link/left_hand_camera",' not in patched_once
    assert 'prim_path="{ENV_REGEX_NS}/Robot/right_dex1_base_link/right_hand_camera",' not in patched_once
    assert patched_once.count("offset=TiledCameraCfg.OffsetCfg(pos=(0.115, 0, 0.07),") == 2
    assert "rot=(0.28059008, -0.25108559, -0.64082457, 0.66900606)," in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.10000, 0, 0.08000)," not in patched_once
    assert "rot=(0.25114139, -0.22289424, -0.65116685, 0.68060848)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.09000, 0, 0.07000)," not in patched_once
    assert "rot=(0.22121463, -0.19427859, -0.66026959, 0.69091532)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.07500, 0, 0.10000)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.07500, 0, 0.08000)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.06537, 0, 0.06140)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.06017, 0, 0.14333)," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.06017, 0, 0.08000)," not in patched_once
    assert "rot=(0.3925, -0.35855, -0.58749, 0.61012)," not in patched_once
    assert "rot=(0.46535710, -0.26866964, -0.42167579, 0.73037587)," not in patched_once
    assert 'convention="opengl"),' in patched_once
    assert "focal_length=12.0," not in patched_once
    assert "horizontal_aperture=20.0," not in patched_once
    assert "update_period=0.02," not in patched_once
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(0.10209156, -0.04657542, 0.42446595)," in patched_once
    assert "focal_length=24.0," in patched_once
    assert "horizontal_aperture=45.56883749280177," in patched_once
    assert "vertical_aperture=34.176628119601325," in patched_once
    assert "pos=(0.10209156, 0.0207748116, 0.42446595)" in patched_once
    assert "horizontal_aperture=48.53," not in patched_once
    assert "vertical_aperture=35.37," not in patched_once
    assert patched_once.count('"global_camera": {') == 2
    assert patched_once.count('prim_path="{ENV_REGEX_NS}/global_camera",') == 2
    assert "offset=TiledCameraCfg.OffsetCfg(pos=(-0.977742, 2.372581, 1.994000)," in patched_once
    assert "rot=(0.25915552, 0.28195817, 0.64845940, 0.65790457)," in patched_once
    assert "horizontal_aperture=90," in patched_once
    assert patched_once.count("horizontal_aperture=62,") == 1
    assert patched_once.count("vertical_aperture=39.8,") == 1
    assert patched_once.count("horizontal_aperture=35.31010639776536,") == 2
    assert patched_once.count("vertical_aperture=26.482579798324018,") == 2
    assert patched_once.count("vertical_aperture=34.176628119601325,") == 1
    assert patched_once.count("vertical_aperture=34.29727835652327,") == 1
    assert patched_once.count("width=640,") == 9
    assert patched_once.count("height=480,") == 9
    assert "width=224," not in patched_once
    assert "height=224," not in patched_once
    assert "class FlipTableUpperBodyJointActionsCfg" in patched_once
    assert patched_once.count(
        "from robofinals.core.mdp.actions.team_ramen_balanced_wbc_action "
        "import TeamRamenBalancedWBCActionCfg"
    ) == 1
    assert patched_once.count("class FlipTableBalancedWBCActionsCfg") == 1
    assert "class FlipTablePinkEEFActionsCfg" in patched_once
    assert 'os.environ.get("FLIP_TABLE_SIM_BODY_MODE", "balanced_wbc")' in patched_once
    assert 'os.environ.get("FLIP_TABLE_USE_PINK_EEF_ACTION", "")' in patched_once
    assert "self.action_config = FlipTableUpperBodyJointActionsCfg()" in patched_once
    assert "self.action_config = FlipTablePinkEEFActionsCfg()" in patched_once
    assert "self.action_config = FlipTableBalancedWBCActionsCfg()" in patched_once
    assert patched_once.count("class FlipTableUpperBodyJointActionsCfg") == 1
    assert patched_once.count("class FlipTablePinkEEFActionsCfg") == 1
