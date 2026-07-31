from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_contracts():
    path = ROOT / "real_to_sim_calibration" / "contracts.py"
    spec = importlib.util.spec_from_file_location("real_to_sim_contracts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_persistent_worker():
    path = ROOT / "tools" / "persistent_eval_worker.py"
    spec = importlib.util.spec_from_file_location("flip_table_persistent_eval_worker", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _episode(module, index: int, amplitude: float):
    frames = 12
    timestamps = np.arange(frames, dtype=np.float64) / 30.0
    q = np.zeros((frames, 36), dtype=np.float64)
    q[:, 19:36] = np.linspace(0.0, amplitude, frames)[:, None]
    hands = np.column_stack((np.linspace(0.0, 4.5, frames), np.linspace(4.5, 0.0, frames)))
    return module.episode_signals(index, timestamps, q, hands)


def test_16d_action_mapping_excludes_waist_and_preserves_arms_hands() -> None:
    module = _load_contracts()
    q = np.arange(72, dtype=np.float64).reshape(2, 36)
    hands = np.array(((0.0, 4.5), (2.0, 3.0)))
    action = module.source_16d_actions(q, hands)
    assert action.shape == (2, 16)
    np.testing.assert_array_equal(action[:, :14], q[:, 22:36])
    np.testing.assert_array_equal(action[:, 14:], hands)


def test_persistent_job_validation_refuses_path_escape_and_unknown_environment(tmp_path: Path) -> None:
    module = _load_persistent_worker()
    root = tmp_path / "persistent"
    queue = root / "persistent_jobs" / "running"
    queue.mkdir(parents=True)
    action = root / "persistent_jobs" / "inputs" / "job" / "replay_actions.json"
    action.parent.mkdir(parents=True)
    action.write_text("{}", encoding="utf-8")
    valid = {
        "schema_version": module.SCHEMA_VERSION,
        "job_id": "valid_job",
        "policy_name": "RecordedJointTargetPolicy",
        "seed": 42,
        "time_out_limit": 10,
        "output_relpath": "run_001",
        "environment": {"FLIP_TABLE_REPLAY_ACTION_PATH": str(action.relative_to(root))},
    }
    valid_path = queue / "valid.job.json"
    valid_path.write_text(json.dumps(valid), encoding="utf-8")
    _, environment, output_dir = module._load_job(valid_path, root)
    assert environment["FLIP_TABLE_REPLAY_ACTION_PATH"] == str(action.resolve())
    assert output_dir == (root / "run_001").resolve()

    calibration = dict(valid)
    calibration["environment"] = {
        **valid["environment"],
        "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON": json.dumps(
            [{"offset_local_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0}]
        ),
    }
    calibration_path = queue / "calibration.job.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    _, calibration_environment, _ = module._load_job(calibration_path, root)
    assert json.loads(calibration_environment["FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON"]) == [
        {"offset_local_m": [0.0, 0.0, 0.0], "yaw_rad": 0.0}
    ]

    malformed = dict(calibration)
    malformed["environment"] = {
        **valid["environment"],
        "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON": "not-json",
    }
    malformed_path = queue / "malformed.job.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        module._load_job(malformed_path, root)

    invalid = dict(valid)
    invalid["output_relpath"] = "../escape"
    invalid_path = queue / "invalid.job.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="must not escape"):
        module._load_job(invalid_path, root)

    invalid = dict(valid)
    invalid["environment"] = {"UNSAFE_SHELL_VARIABLE": "no"}
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported keys"):
        module._load_job(invalid_path, root)


def test_persistent_worker_accepts_every_generated_replay_environment_key(tmp_path: Path) -> None:
    worker = _load_persistent_worker()
    replay = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.replay"
    )
    actions = np.zeros((4, 19), dtype=np.float64)
    bundle = {
        "source_episode_index": 1,
        "source_episode_name": "episode_000001",
        "fps": 30.0,
        "action_layout": {"name": "test"},
        "recorded_upper_body_target_and_hand_cmd": actions.tolist(),
        "observed_upper_body_state_and_hand_state": actions.tolist(),
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    output_dir = tmp_path / "output"
    replay_payload = replay.materialize(bundle_path, output_dir)
    environment = replay.runtime_environment(replay_payload, output_dir)
    permitted_after_client_normalization = set(environment) - {
        "FLIP_TABLE_POLICY_NAME",
        "FLIP_TABLE_SIM_OUTPUT_DIR",
        "FLIP_TABLE_TEST_NUM",
    }
    assert permitted_after_client_normalization <= worker.ALLOWED_ENVIRONMENT_KEYS


def test_replay_command_delay_is_explicit_and_not_a_policy_feature() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    replay = (ROOT / "real_to_sim_calibration" / "replay.py").read_text(encoding="utf-8")
    worker = _load_persistent_worker()
    assert "FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS" in policy
    assert "source_observed_frame" in policy
    assert '"FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS": "0"' in replay
    assert "FLIP_TABLE_REPLAY_COMMAND_DELAY_STEPS" in worker.ALLOWED_ENVIRONMENT_KEYS


def test_persistent_worker_allows_only_explicit_reset_time_actuator_identification() -> None:
    worker = _load_persistent_worker()

    expected = {
        "FLIP_TABLE_ARM_STIFFNESS_SCALE_RANGE",
        "FLIP_TABLE_ARM_DAMPING_SCALE_RANGE",
        "FLIP_TABLE_ARM_ARMATURE_SCALE_RANGE",
        "FLIP_TABLE_ARM_FRICTION_SCALE_RANGE",
    }

    assert expected <= worker.ALLOWED_ENVIRONMENT_KEYS


def test_persistent_worker_bounds_scripted_reset_stability_probe(tmp_path: Path) -> None:
    worker = _load_persistent_worker()
    root = tmp_path / "persistent"
    root.mkdir()
    valid = {
        "schema_version": worker.SCHEMA_VERSION,
        "job_id": "scripted_static_probe",
        "policy_name": "ScriptedJointPolicy",
        "seed": 1,
        "time_out_limit": 10,
        "output_relpath": "out",
        "environment": {"FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE": "0"},
    }
    path = root / "job.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    _, environment, _ = worker._load_job(path, root)
    assert environment["FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE"] == "0"

    valid["environment"] = {"FLIP_TABLE_SCRIPTED_ACTION_AMPLITUDE": "0.26"}
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match=r"within \[0.0, 0.25\]"):
        worker._load_job(path, root)


def test_persistent_worker_supports_explicit_calibration_environment_recreation(
    tmp_path: Path,
) -> None:
    worker = _load_persistent_worker()
    root = tmp_path / "persistent"
    root.mkdir()
    valid = {
        "schema_version": worker.SCHEMA_VERSION,
        "job_id": "isolated_calibration_probe",
        "policy_name": "ScriptedJointPolicy",
        "seed": 1,
        "time_out_limit": 10,
        "output_relpath": "out",
        "environment": {"FLIP_TABLE_PERSISTENT_RECREATE_ENV": "true"},
    }
    path = root / "job.json"
    path.write_text(json.dumps(valid), encoding="utf-8")
    _, environment, _ = worker._load_job(path, root)
    assert environment["FLIP_TABLE_PERSISTENT_RECREATE_ENV"] == "true"

    valid["environment"] = {"FLIP_TABLE_PERSISTENT_RECREATE_ENV": "yes"}
    path.write_text(json.dumps(valid), encoding="utf-8")
    with pytest.raises(ValueError, match="must be true or false"):
        worker._load_job(path, root)


def test_parallel_scene_probe_requests_task_recreation_for_candidate_isolation(
    tmp_path: Path,
) -> None:
    probe = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.parallel_scene_probe"
    )
    environment = probe.write_probe_environment(
        [
            {
                "offset_local_m": [0.0, 0.0, 0.0],
                "yaw_rad": 0.0,
                "head_stereo_offset_local_m": [0.0, 0.0, 0.0],
                "head_stereo_rotation_rpy_deg": [0.0, 0.0, 0.0],
                "robot_root_pos_local_m": [0.0, 0.0, 0.78],
                "robot_root_yaw_rad": 0.0,
            }
        ],
        output_dir=tmp_path / "probe",
        replay_action_path=tmp_path / "replay.json",
        frame_index=1,
    )
    assert environment["FLIP_TABLE_PERSISTENT_RECREATE_ENV"] == "true"


def test_shared_camera_candidate_rejects_one_episode_visual_regression(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.assess_camera_candidate"
    )

    def alignment(path: Path, *, image: Path, iou: float, edge: float):
        payload = {
            "schema_version": "team_ramen_table_silhouette_alignment/v2",
            "real_image": str(image),
            "simulated_image": str(path.with_suffix(".png")),
            "roi_xyxy": [55, 70, 585, 365],
            "mask_iou": iou,
            "edge_distance_symmetric_px": edge,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return module._load_alignment(path)

    first = tmp_path / "episode_250.png"
    second = tmp_path / "episode_499.png"
    first.write_bytes(b"episode 250")
    second.write_bytes(b"episode 499")
    result = module.assess(
        [
            (
                alignment(tmp_path / "first_base.json", image=first, iou=0.61, edge=26.0),
                alignment(tmp_path / "first_candidate.json", image=first, iou=0.67, edge=24.0),
            ),
            (
                alignment(tmp_path / "second_base.json", image=second, iou=0.53, edge=35.0),
                alignment(tmp_path / "second_candidate.json", image=second, iou=0.49, edge=41.0),
            ),
        ]
    )
    assert result["decision"] == "rejected_cross_episode_visual_regression"
    assert result["accepted_for_shared_simulator_default"] is False


def test_head_right_uses_its_own_pinned_raw_calibration() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text(
        encoding="utf-8"
    )
    assert '"head_right_camera": {' in policy
    assert "336.30012498108425" in policy
    assert "321.60051380995424" in policy
    assert "0.06366431884731834" in policy
    assert "HEAD_RIGHT_HORIZONTAL_APERTURE" in patch
    assert "HEAD_RIGHT_VERTICAL_APERTURE" in patch
    spec = importlib.util.spec_from_file_location(
        "flip_table_camera_patch", ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cameras = {item.camera_name: item for item in module.POLICY_CAMERA_GEOMETRIES}
    assert cameras["first_person_camera"].horizontal_aperture == module.HEAD_LEFT_HORIZONTAL_APERTURE
    assert cameras["head_right_camera"].horizontal_aperture == module.HEAD_RIGHT_HORIZONTAL_APERTURE
    assert cameras["head_right_camera"].vertical_aperture == module.HEAD_RIGHT_VERTICAL_APERTURE


def test_persistent_worker_restores_only_explicit_foundation_keys_after_restart() -> None:
    runner = (ROOT / "persistent_eval.sh").read_text(encoding="utf-8")
    assert 'FOUNDATION_ENV_FILE="${FLIP_TABLE_PERSISTENT_FOUNDATION_ENV_FILE:-$PERSISTENT_ROOT/persistent_foundation.env}"' in runner
    assert "load_foundation_environment" in runner
    assert "FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE" in runner
    assert "FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE" in runner
    assert "load_foundation_environment\n    if worker_running" in runner


def test_persistent_ready_state_records_immutable_foundation_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _load_persistent_worker()
    monkeypatch.setenv("FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE", "2")
    monkeypatch.setenv("FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE", "1.41421356237")
    payload = worker._ready_payload(state="ready")
    assert payload["state"] == "ready"
    assert payload["foundation_environment"]["FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE"] == "2"
    assert payload["foundation_environment"]["FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE"] == "1.41421356237"
    assert "FLIP_TABLE_G1_USD_PATH" not in worker.ALLOWED_ENVIRONMENT_KEYS
    assert "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION" not in worker.ALLOWED_ENVIRONMENT_KEYS


def test_actuator_probe_uses_fixed_reset_parameters_and_bounded_prefix() -> None:
    script = (ROOT / "real_to_sim_calibration" / "run_actuator_probe.sh").read_text(
        encoding="utf-8"
    )

    assert "FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES=true" in script
    assert "FLIP_TABLE_ARM_STIFFNESS_SCALE_RANGE=$STIFFNESS_SCALE,$STIFFNESS_SCALE" in script
    assert "FLIP_TABLE_ARM_DAMPING_SCALE_RANGE=$DAMPING_SCALE,$DAMPING_SCALE" in script
    assert "FLIP_TABLE_ARM_ARMATURE_SCALE_RANGE=$ARMATURE_SCALE,$ARMATURE_SCALE" in script
    assert "FLIP_TABLE_ARM_FRICTION_SCALE_RANGE=1,1" in script
    assert "FLIP_TABLE_PERSISTENT_RECREATE_ENV=true" in script
    assert '--time-out-limit "$CONTROL_STEPS"' in script
    assert '--source-frame-start "$SOURCE_FRAME_START"' in script
    assert '--source-frame-end "$SOURCE_FRAME_END"' in script


def test_head_arm_motion_metric_rewards_projected_motion_support() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.head_arm_motion_alignment"
    )
    before = np.zeros((480, 640, 3), dtype=np.uint8)
    after = before.copy()
    cv2 = pytest.importorskip("cv2")
    cv2.line(after, (100, 220), (200, 220), (255, 255, 255), 5)
    mask = module.motion_support_mask(before, after)
    supported = np.column_stack((np.arange(100, 201, dtype=np.float64), np.full(101, 220.0)))
    distant = supported + np.asarray((0.0, 150.0))
    supported_score = module.motion_distance_score(mask, supported)
    distant_score = module.motion_distance_score(mask, distant)
    assert supported_score[0] < distant_score[0]
    assert supported_score[1] > distant_score[1]


def test_head_arm_motion_sampling_clips_extreme_offscreen_projections() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.head_arm_motion_alignment"
    )
    samples = module.sample_projected_segments(
        {
            "left": np.asarray(
                ((-1.0e12, 240.0), (1.0e12, 240.0)), dtype=np.float64
            )
        }
    )
    assert 2 <= len(samples) <= module._MAX_SAMPLES_PER_SEGMENT
    assert np.isfinite(samples).all()
    assert np.all((0.0 <= samples[:, 0]) & (samples[:, 0] < 640.0))
    assert np.all((0.0 <= samples[:, 1]) & (samples[:, 1] < 480.0))
    assert module.sample_projected_segments(
        {"left": np.asarray(((-1.0e308, 240.0), (1.0e308, 240.0)), dtype=np.float64)}
    ).shape == (0, 2)


def test_head_arm_motion_overlay_clips_extreme_offscreen_projections() -> None:
    cv2 = pytest.importorskip("cv2")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.head_arm_motion_alignment"
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    overlay = module._overlay(
        image,
        {"left": np.asarray(((-1.0e12, 240.0), (1.0e12, 240.0)), dtype=np.float64)},
        (0, 255, 0),
    )
    assert cv2.countNonZero(cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)) > 0


def test_shared_head_mount_bundle_recovers_synthetic_fixed_mount() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_shared_head_mount_bundle"
    )
    rotation = pytest.importorskip("scipy.spatial.transform").Rotation

    def transform(translation, rotation_deg):
        value = np.eye(4)
        value[:3, :3] = rotation.from_euler("XYZ", rotation_deg, degrees=True).as_matrix()
        value[:3, 3] = translation
        return value

    mount = transform((0.10, -0.02, 0.42), (3.0, -2.0, 1.0))
    eye_from_table = transform((0.30, 0.20, 0.60), (0.0, 0.0, 0.0))
    tables = {
        10: transform((0.70, 0.10, 0.01), (0.0, 0.0, 10.0)),
        20: transform((0.72, -0.08, 0.01), (0.0, 0.0, -12.0)),
    }
    observations = []
    for episode, table in tables.items():
        for frame, torso in enumerate(
            (transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), transform((0.0, 0.0, 0.0), (2.0, -1.0, 3.0)))
        ):
            camera_from_table = np.linalg.inv(torso @ mount) @ table
            observations.append(
                module.Observation(
                    episode,
                    frame,
                    "head_left",
                    torso,
                    np.eye(4),
                    mount,
                    camera_from_table,
                )
            )
    result = module.fit_shared_mount(observations)
    recovered = result["final_mount"]
    assert np.linalg.norm(recovered[:3, 3] - mount[:3, 3]) < 1.0e-7
    assert rotation.from_matrix(recovered[:3, :3].T @ mount[:3, :3]).magnitude() < 1.0e-7
    assert result["optimization"]["mount_correction_at_bound"] is False


def test_persistent_worker_switches_avp_observation_layout_without_new_isaac_app() -> None:
    worker = _load_persistent_worker()

    assert "AvpTeleopPolicy" in worker.SUPPORTED_POLICIES
    assert "Dex1ForceCalibrationPolicy" not in worker.SUPPORTED_POLICIES
    assert worker._environment_mode("AvpTeleopPolicy") == "avp_direct"
    assert worker._environment_mode("RecordedJointTargetPolicy") == "standard"
    # These values are consumed only by AVP itself or the next task reset.
    # In particular, an AVP job cannot mutate the fixed action-manager setup.
    required = {
        "FLIP_TABLE_TELEOP_PORT",
        "FLIP_TABLE_TELEOP_PERSISTENT",
        "FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS",
        "FLIP_TABLE_REQUIRE_WAIST_LOCK",
    }
    assert required <= worker.ALLOWED_ENVIRONMENT_KEYS
    assert "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION" not in worker.ALLOWED_ENVIRONMENT_KEYS


def test_19d_observation_mapping_preserves_measured_hand_overshoot() -> None:
    module = _load_contracts()
    q = np.arange(72, dtype=np.float64).reshape(2, 36)
    hands = np.array(((4.53, 4.50), (4.55, 2.0)))
    state = module.source_19d_observation(q, hands)
    np.testing.assert_array_equal(state[:, :17], q[:, 19:36])
    np.testing.assert_array_equal(state[:, 17:], hands)


def test_selection_is_disjoint_and_deterministic() -> None:
    module = _load_contracts()
    signals = [_episode(module, index, float(index + 1)) for index in range(12)]
    first = module.select_episode_roles(signals)
    second = module.select_episode_roles(signals)
    assert first == second
    assert len(first.all_indices()) == 8
    assert len(set(first.all_indices())) == 8


def test_selection_can_require_source_eef_fk_eligible_episodes() -> None:
    module = _load_contracts()
    signals = [_episode(module, index, float(index + 1)) for index in range(16)]
    eligible = {index for index in range(16) if index % 2 == 0}
    selected = module.select_episode_roles(signals, eligible_episode_indices=eligible)
    assert set(selected.all_indices()).issubset(eligible)
    try:
        module.select_episode_roles(signals, eligible_episode_indices=range(7))
    except ValueError as exc:
        assert "EEF/FK-eligible" in str(exc)
    else:
        raise AssertionError("selection must reject fewer than eight EEF/FK-eligible episodes")


def test_reviewed_visual_selection_override_remains_audited_and_eligible(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.prepare"
    )
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "team_ramen_flip_table_selection_override/v1",
                "selection": {
                    "anchor": 417,
                    "calibration": [250, 184],
                    "validation": [100, 501, 334, 143, 341],
                },
                "rationale": "replace a visually unavailable numeric candidate with direct stereo-CAD evidence",
                "visual_evidence": [
                    {
                        "episode_index": 509,
                        "decision": "rejected",
                        "reason": "fewer than three stereo-consistent direct CAD pairs",
                    },
                    {
                        "episode_index": 184,
                        "decision": "accepted",
                        "reason": "four direct stereo-CAD pairs passed",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    selection, provenance = module._selection_from_override(
        path,
        available_episode_indices=set(range(531)),
        eligible_episode_indices=set(range(531)),
    )
    assert selection.calibration == (250, 184)
    assert provenance["mode"] == "reviewed_visual_evidence_override"
    assert provenance["numeric_audit_still_executed"] is True


def test_joint_metrics_fail_closed_for_unavailable_lower_body() -> None:
    module = _load_contracts()
    target = np.zeros((5, 16), dtype=np.float64)
    actual = np.zeros((5, 19), dtype=np.float64)
    actual[:, 3] = 0.02
    metrics = module.compute_joint_replay_metrics(target, actual)
    assert metrics.upper_body_rmse_rad > 0.0
    assert metrics.lower_body_rmse_rad is None
    assert metrics.passed() is False


def test_metric_gate_requires_all_real_to_sim_metrics() -> None:
    module = _load_contracts()
    report = module.compare_metric_gate({"mask_iou": 0.95})
    assert report["passed"] is False
    assert report["metrics"]["camera_reprojection_median_px"]["status"] == "missing"
    passed = module.compare_metric_gate(
        {
            "camera_reprojection_median_px": 2.0,
            "camera_reprojection_p95_px": 7.0,
            "table_translation_rmse_m": 0.01,
            "table_rotation_rmse_deg": 2.0,
            "phase_timing_max_error_s": 0.08,
            "mask_iou": 0.92,
        }
    )
    assert passed["passed"] is True


def test_replay_runner_mounts_host_action_file() -> None:
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    assert 'OUTPUT_DIR="$(realpath -m "${FLIP_TABLE_SIM_OUTPUT_DIR:-$ROOT_DIR/outputs/flip_table_simulation/eval_result}")"' in runner
    assert "REPLAY_ACTION_CONTAINER=\"/workspace/flip_table_replay/replay_actions.json\"" in runner
    assert "replay_mount_args=(-v" in runner
    assert "FLIP_TABLE_REPLAY_ACTION_PATH=\"$REPLAY_ACTION_CONTAINER\"" in runner
    assert "FLIP_TABLE_REPLAY_WARMUP_STEPS" in runner
    assert "FLIP_TABLE_INITIAL_UPPER_BODY_STATE" in runner
    assert "FLIP_TABLE_CAMERA_FRAME_INDICES" in runner
    assert "full_body_diagnostic" in runner
    assert "RecordedFullBodyTargetPolicy" in runner


def test_full_body_diagnostic_replay_preserves_all_recorded_body_targets(tmp_path: Path) -> None:
    contracts = _load_contracts()
    q = np.arange(72, dtype=np.float64).reshape(2, 36)
    hands = np.array(((0.0, 4.5), (2.0, 3.0)))
    actions = contracts.source_31d_actions(q, hands)
    observed = contracts.source_31d_observation(q, hands)
    assert actions.shape == (2, 31)
    assert observed.shape == (2, 31)
    np.testing.assert_array_equal(actions[:, :29], q[:, 7:])
    np.testing.assert_array_equal(actions[:, 29:], hands)

    replay = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    bundle = {
        "source_episode_index": 1,
        "source_episode_name": "episode_000001",
        "fps": 30.0,
        "action_layout": "arms",
        "recorded_full_body_hand_target_31d": actions.tolist(),
        "observed_full_body_state_and_hand_state": observed.tolist(),
    }
    bundle_path = tmp_path / "full_body_replay_bundle.json"
    output_path = tmp_path / "full_body_replay_actions.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    payload = replay.materialize(bundle_path, output_path, full_body_diagnostic=True)
    assert payload["body_mode"] == "full_body_diagnostic"
    assert payload["policy_name"] == "RecordedFullBodyTargetPolicy"
    assert np.asarray(payload["actions"]).shape == (2, 31)
    environment = replay.runtime_environment(payload, tmp_path / "full_body_runtime")
    assert environment["FLIP_TABLE_SIM_BODY_MODE"] == "full_body_diagnostic"
    assert environment["FLIP_TABLE_POLICY_NAME"] == "RecordedFullBodyTargetPolicy"
    assert "FLIP_TABLE_INITIAL_FULL_BODY_STATE" in environment
    assert "FLIP_TABLE_INITIAL_UPPER_BODY_STATE" not in environment


def test_full_body_trace_reports_root_drift_without_hiding_it(tmp_path: Path) -> None:
    replay = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    trace = tmp_path / "full_body_trace.jsonl"
    initial = np.zeros(31, dtype=np.float64)
    rows = []
    for step, root_z in enumerate((0.78, 0.12)):
        rows.append(
            {
                "replay_warmup": False,
                "source_action_31d": initial.tolist(),
                "source_observed_state_31d": initial.tolist(),
                "actual_state_31d": initial.tolist(),
                "simulator_scene_diagnostics": {
                    "root_pose_world_xyzw": [0.0, 0.0, root_z, 0.0, 0.0, 0.0, 1.0],
                    "white_table": {"position_world_m": [0.0, 0.0, 0.8 + step * 0.01]},
                },
            }
        )
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = replay.analyze_full_body_trace(trace, tmp_path / "full_body_report.json")
    assert report["command_tracking"]["upper_body_joint_rad"]["rmse"] == 0.0
    assert report["floating_base_diagnostics"]["root_height_final_m"] == 0.12
    assert report["floating_base_diagnostics"]["root_displacement_max_m"] > 0.6


def test_root_trajectory_diagnostic_uses_relative_motion_only(tmp_path: Path) -> None:
    replay = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    bundle = tmp_path / "bundle.json"
    _write_json(
        bundle,
        {
            "source_episode_index": 1,
            "fps": 30.0,
            "observed_root_pose_xyz_wxyz": [
                [10.0, 20.0, 0.7, 1.0, 0.0, 0.0, 0.0],
                [10.1, 20.0, 0.7, 1.0, 0.0, 0.0, 0.0],
                [10.2, 20.0, 0.7, 1.0, 0.0, 0.0, 0.0],
            ],
        },
    )
    rows = [
        {
            "replay_warmup": False,
            "simulator_scene_diagnostics": {
                "root_pose_world_xyzw": [3.0, -4.0, 0.8, 0.0, 0.0, 0.0, 1.0]
            },
        },
        {
            "replay_warmup": False,
            "simulator_scene_diagnostics": {
                "root_pose_world_xyzw": [3.06, -4.0, 0.8, 0.0, 0.0, 0.0, 1.0]
            },
        },
    ]
    report = replay._root_trajectory_diagnostics(rows, bundle)
    assert report["available"] is True
    assert report["source_displacement_max_m"] == pytest.approx(0.06)
    assert report["simulator_displacement_max_m"] == pytest.approx(0.06)
    assert report["rmse_m"] == pytest.approx(0.0)


def test_recorded_replay_keeps_dataset_dex1_open_closed_convention() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    start = policy.index("class RecordedJointTargetPolicy")
    end = policy.index("class CvRuleBasedPolicy")
    replay = policy[start:end]
    assert "1.0\n                - 2.0" in replay
    assert "* 2.0\n                - 1.0" not in replay
    assert "FLIP_TABLE_REPLAY_WARMUP_STEPS" in replay


def test_replay_runtime_starts_from_a_fixed_calibration_scene() -> None:
    replay = (ROOT / "real_to_sim_calibration" / "replay.py").read_text(encoding="utf-8")
    assert '"FLIP_TABLE_RANDOMIZE_LIGHTING": "false"' in replay
    assert '"FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS": "false"' in replay
    assert '"FLIP_TABLE_TABLE_YAW_RANGE_RAD": "0"' in replay
    assert '"FLIP_TABLE_REPLAY_WARMUP_STEPS": str(warmup_steps)' in replay
    assert '"FLIP_TABLE_INITIAL_UPPER_BODY_STATE"' in replay
    assert '"FLIP_TABLE_CAMERA_FRAME_INDEX": str(warmup_steps - 1)' in replay
    assert '"FLIP_TABLE_CAMERA_FRAME_INDICES": ",".join(str(step) for step in camera_steps)' in replay
    assert '"FLIP_TABLE_SAVE_CAMERA_FRAMES": "true"' in replay
    assert '"FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY": "true"' in replay
    assert '"FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE": "true"' in replay
    assert '"FLIP_TABLE_TIME_OUT_LIMIT": str(time_out_limit)' in replay


def test_replay_camera_frame_map_preserves_source_evidence_timeline() -> None:
    module = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    mapping = module.camera_frame_map(101)
    assert mapping[0] == {"source_frame": 0, "simulator_step": 119}
    assert mapping[1] == {"source_frame": 10, "simulator_step": 137}
    assert mapping[-1] == {"source_frame": 100, "simulator_step": 287}
    assert [item["source_frame"] for item in mapping] == [0, 10, 25, 50, 75, 100]


def test_initial_pose_only_runtime_stops_before_the_first_recorded_action(tmp_path: Path) -> None:
    replay = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    payload = {
        "fps": 30.0,
        "actions": np.zeros((4, 16), dtype=np.float64).tolist(),
        "initial_state_19d": np.zeros(19, dtype=np.float64).tolist(),
        "camera_frame_map": [
            {"source_frame": 0, "simulator_step": 119},
            {"source_frame": 1, "simulator_step": 122},
        ],
    }
    environment = replay.runtime_environment(payload, tmp_path, initial_pose_only=True)
    assert environment["FLIP_TABLE_REPLAY_WARMUP_STEPS"] == "120"
    assert environment["FLIP_TABLE_TIME_OUT_LIMIT"] == "120"
    assert environment["FLIP_TABLE_CAMERA_FRAME_INDEX"] == "119"


def test_initialization_probe_report_rejects_a_terminal_warmup_state_outside_q_tolerance(
    tmp_path: Path,
) -> None:
    replay = importlib.import_module("evaluate.flip_table_simulation.real_to_sim_calibration.replay")
    trace = tmp_path / "trace.jsonl"
    row = {
        "replay_warmup": True,
        "source_action_16d": [0.0] * 16,
        "source_observed_state_19d": [0.0] * 19,
        "state_after": [0.0] * 33,
    }
    for index in (2, 5, 8, 11, 15):
        row["state_after"][index] = 0.1
    trace.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "initialization.json"
    report = replay.analyze_initialization_trace(trace, output)
    assert report["camera_frame"] == "last_warmup_step"
    assert report["passed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is False


def test_scene_candidate_grid_preserves_fixed_root_and_scales_only_calibration_fields() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.scene_candidate_grid"
    )
    scene = {
        "candidates": [
            {
                "offset_local_m": [0.1, -0.2, 0.03],
                "yaw_rad": 0.4,
                "robot_root_pos_local_m": [-0.8, 2.3, 0.78],
                "robot_root_yaw_rad": -3.14,
            }
        ]
    }
    head = {
        "candidates": [
            {
                "head_stereo_offset_local_m": [0.01, 0.02, -0.03],
                "head_stereo_rotation_rpy_deg": [1.0, -2.0, 3.0],
            }
        ]
    }
    candidates = module.build_grid(
        scene, head, scene_scales=(0.0, 0.5, 1.0), head_scales=(0.0, 1.0)
    )
    assert len(candidates) == 6
    half_scene = candidates[2]
    assert half_scene["offset_local_m"] == [0.05, -0.1, 0.015]
    assert half_scene["yaw_rad"] == pytest.approx(0.2)
    assert half_scene["head_stereo_offset_local_m"] == [0.0, 0.0, 0.0]
    assert half_scene["robot_root_pos_local_m"] == [-0.8, 2.3, 0.78]
    full = candidates[-1]
    assert full["head_stereo_rotation_rpy_deg"] == [1.0, -2.0, 3.0]


def test_table_offset_grid_changes_only_workbench_local_table_xy() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.table_offset_grid"
    )
    report = {
        "candidates": [
            {
                "offset_local_m": [0.1, -0.2, 0.03],
                "yaw_rad": 0.4,
                "robot_root_pos_local_m": [-0.8, 2.3, 0.78],
                "robot_root_yaw_rad": -3.14,
                "head_stereo_offset_local_m": [-0.04, 0.0, 0.0],
            }
        ]
    }
    candidates = module.build_grid(report, x_offsets_m=(-0.1, 0.1), y_offsets_m=(0.0,))
    assert [candidate["offset_local_m"] for candidate in candidates] == [
        [0.0, -0.2, 0.03],
        [0.2, -0.2, 0.03],
    ]
    assert all(candidate["yaw_rad"] == 0.4 for candidate in candidates)
    assert all(candidate["robot_root_pos_local_m"] == [-0.8, 2.3, 0.78] for candidate in candidates)
    assert all(candidate["head_stereo_offset_local_m"] == [-0.04, 0.0, 0.0] for candidate in candidates)


def test_multiframe_head_comparison_can_read_the_immutable_replay_mapping(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.compare_multiframe_head"
    )
    actions = tmp_path / "replay_actions.json"
    _write_json(
        actions,
        {
            "camera_frame_map": [
                {"source_frame": 0, "simulator_step": 119},
                {"source_frame": 10, "simulator_step": 137},
            ]
        },
    )
    assert module.frame_map_from_replay_actions(actions) == ((0, 119), (10, 137))


def test_anchor_replay_requires_the_actual_trace_and_camera_evidence_paths() -> None:
    runner = (ROOT / "real_to_sim_calibration" / "run_anchor_replay.sh").read_text(encoding="utf-8")
    assert 'TRACE_PATH="$OUTPUT_DIR/test_0/action_state_trace.jsonl"' in runner
    assert 'CAMERA_DIR="$OUTPUT_DIR/test_0/camera_frames"' in runner
    assert '"ERROR: calibration replay finished without camera-frame evidence' in runner


def test_calibration_bundle_retains_eef_dual_labels_for_offline_fk_checks() -> None:
    prepare = (ROOT / "real_to_sim_calibration" / "prepare.py").read_text(encoding="utf-8")
    contracts = (ROOT / "real_to_sim_calibration" / "contracts.py").read_text(encoding="utf-8")
    assert '"observation.state.ee_state"' in prepare
    assert '"action.ee_action"' in prepare
    assert '"observed_ee_state_xyz_euler_xyz_rad"' in prepare
    assert '"recorded_ee_action_xyz_euler_xyz_rad"' in prepare
    assert "never a replay controller input" in prepare
    assert "--eef-fk-audit" in prepare
    assert "fitted_tool_transforms_imported" in prepare
    assert 'SOURCE_EEF_POSE_FORMAT = "xyz_euler_xyz_rad"' in contracts


def test_replay_scene_diagnostics_are_explicitly_off_policy() -> None:
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    task = (
        ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py"
    ).read_text(encoding="utf-8")
    recorded = policy[policy.index("class RecordedJointTargetPolicy") : policy.index("class CvRuleBasedPolicy")]
    assert '"FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE"' in policy
    assert '"simulator_scene_diagnostics": scene_diagnostics' in recorded
    assert '"actuator_drive_diagnostics"' in task
    assert "never enter an observation, action, policy" in policy
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    assert "-e FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE=" in runner


def test_eval_runners_reap_the_fixed_port_server_on_failure() -> None:
    for runner_name in ("run_eval.sh", "run_eval_in_container.sh"):
        runner = (ROOT / runner_name).read_text(encoding="utf-8")
        assert "cleanup_env_server()" in runner
        assert 'trap cleanup_env_server EXIT INT TERM' in runner
        assert 'wait "$SERVER_PID"' in runner


def test_host_runner_fails_before_starting_a_second_fixed_port_evaluator() -> None:
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    assert "FLIP_TABLE_IPC_LOCK_PATH" in runner
    assert "flock -n 9" in runner
    assert "another RoboFinals evaluator owns IPC port 50000" in runner


def test_container_runner_preserves_multi_frame_camera_export() -> None:
    runner = (ROOT / "run_eval_in_container.sh").read_text(encoding="utf-8")
    assert 'export FLIP_TABLE_CAMERA_FRAME_INDICES="${FLIP_TABLE_CAMERA_FRAME_INDICES:-}"' in runner


def test_host_camera_export_path_is_remapped_to_the_output_mount() -> None:
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    assert 'CAMERA_FRAME_OUTPUT_CONTAINER=""' in runner
    assert 'FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR must be inside FLIP_TABLE_SIM_OUTPUT_DIR' in runner
    assert '-e FLIP_TABLE_CAMERA_FRAME_OUTPUT_DIR="$CAMERA_FRAME_OUTPUT_CONTAINER"' in runner


def test_parallel_scene_probe_is_explicit_and_preserves_normal_camera_layout() -> None:
    task = (ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py").read_text(
        encoding="utf-8"
    )
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    assert "def _calibration_table_pose_candidates()" in task
    assert "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON length must equal the active" in task
    assert "offsets[:, 2]" in task
    assert "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT" in policy
    assert 'env_dir = frame_dir / f"env_{batch_index:03d}" if batch_size > 1 else frame_dir' in policy
    assert "FLIP_TABLE_CALIBRATION_NUM_ENVS" in runner
    assert 'config["env_cfg"]["num_envs"] = count' in runner
    assert "FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION" in runner


def test_parallel_probe_environment_is_accepted_by_persistent_client(tmp_path: Path) -> None:
    from evaluate.flip_table_simulation.real_to_sim_calibration import parallel_scene_probe
    from evaluate.flip_table_simulation.tools import persistent_eval_client

    candidates = [
        {
            "label": "nominal",
            "offset_local_m": [0.0, 0.01, 0.0],
            "yaw_rad": 0.0,
            "robot_root_pos_local_m": [-0.8, 2.37, 0.78],
            "robot_root_yaw_rad": -3.14,
        }
    ]
    replay = tmp_path / "replay_actions.json"
    replay.write_text("{}", encoding="utf-8")

    expected = parallel_scene_probe.write_probe_environment(
        candidates,
        output_dir=tmp_path / "probe",
        replay_action_path=replay,
        frame_index=119,
    )

    assert persistent_eval_client._read_generated_environment(
        tmp_path / "probe" / "parallel_probe.env"
    ) == expected
    encoded = json.loads(expected["FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON"])
    assert encoded[0]["robot_root_pos_local_m"] == [-0.8, 2.37, 0.78]
    assert encoded[0]["robot_root_yaw_rad"] == -3.14


def test_parallel_probe_supports_only_episode_fixed_stereo_rig_candidates() -> None:
    task = (ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py").read_text(
        encoding="utf-8"
    )
    probe = (ROOT / "real_to_sim_calibration" / "parallel_scene_probe.py").read_text(
        encoding="utf-8"
    )
    assert "head_stereo_offset_local_m" in task
    assert "head_stereo_rotation_rpy_deg" in task
    assert "group_name == \"head_stereo\"" in task
    assert "head_stereo_offset_local_m" in probe
    assert "head_stereo_rotation_rpy_deg" in probe


def test_parallel_probe_records_candidates_but_never_accepts_one() -> None:
    source = (ROOT / "real_to_sim_calibration" / "parallel_scene_probe.py").read_text(
        encoding="utf-8"
    )
    assert '"policy_use": "forbidden: offline camera/scene calibration only"' in source
    assert '"accepted_candidate": None' in source
    assert '"decision": "diagnostic_only_pending_multiview_and_heldout_validation"' in source
    assert '"FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT": "true"' in source
    assert '"FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE": "true"' in source
    assert '"FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY": "true"' in source
    assert '"FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION": "0.70"' in source
    assert 'flat_image = frame_dir / "head_left_rgb.png"' in source
    assert "len(candidates) == 1 and flat_image.is_file()" in source
    assert '"head_left_saved_in_recorded_raw_geometry": True' in source


def test_parallel_probe_rejects_low_quality_pnp_candidates_from_fit_ranking() -> None:
    source = (ROOT / "real_to_sim_calibration" / "parallel_scene_probe.py").read_text(
        encoding="utf-8"
    )
    assert "_PNP_REPROJECTION_ERROR_MAX_PX = 8.0" in source
    assert '"pnp_reliable": _is_reliable_pnp(simulated)' in source
    assert "not item.get(\"pnp_reliable\", False)" in source


def test_trace_scene_refinement_is_offline_camera_relative_and_support_gated() -> None:
    source = (
        ROOT / "real_to_sim_calibration" / "refine_scene_candidate_from_trace.py"
    ).read_text(encoding="utf-8")
    assert "single offline camera-relative realized-reset correction" in source
    assert "unobserved free-base world pose" in source
    assert "_workbench_support_envelope" in source
    assert '"policy_use": "forbidden: offline reset calibration only"' in source
    assert "policy, planner, or reward" in source
    assert "unapplied_vertical_offset_workbench_local_m" in source
    assert "fixed_to_zero_for_physical_workbench_support" in source
    assert "delta_local[2] = 0.0" in source


def test_parallel_probe_uses_silhouette_alignment_without_accepting_a_fit() -> None:
    source = (ROOT / "real_to_sim_calibration" / "parallel_scene_probe.py").read_text(
        encoding="utf-8"
    )
    visual = (ROOT / "real_to_sim_calibration" / "visual_alignment.py").read_text(
        encoding="utf-8"
    )
    assert "compare_images(real_image, image)" in source
    assert '"accepted_candidate": None' in source
    assert "edge_distance_symmetric_px" in source
    assert "policy, planner, reward, or inference-time branch" in visual


def test_camera_comparison_requires_explicit_image_geometry_provenance() -> None:
    scene = (ROOT / "real_to_sim_calibration" / "compare_scene_candidates.py").read_text(
        encoding="utf-8"
    )
    temporal = (ROOT / "real_to_sim_calibration" / "compare_multiframe_head.py").read_text(
        encoding="utf-8"
    )
    assert "sim_recorded_geometry" in scene
    assert "--sim-recorded-geometry" in scene
    assert "--environment-index" in temporal
    assert "silhouette_alignment" in temporal
    assert "_sim_head_image_path" in temporal
    assert "sim_recorded_geometry" in temporal
    assert "--sim-recorded-geometry" in temporal


def test_replay_uses_observed_state_for_reset_and_warmup() -> None:
    task = (ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py").read_text(
        encoding="utf-8"
    )
    policy = (ROOT / "container_overlay" / "policy" / "flip_table_eval_policy.py").read_text(
        encoding="utf-8"
    )
    assert 'FLIP_TABLE_INITIAL_UPPER_BODY_STATE' in task
    assert 'FLIP_TABLE_INITIAL_UPPER_BODY_ACTION is not valid for real-to-sim replay' in task
    assert 'step < self._replay_warmup_steps' in policy
    assert '"source_action_16d"' in policy
    assert '"source_observed_state_19d"' in policy
    assert "observed_states_19d" in policy
    assert "FLIP_TABLE_CAMERA_FRAME_INDICES" in policy


def test_replay_metrics_distinguish_commands_from_real_observations() -> None:
    replay = (ROOT / "real_to_sim_calibration" / "replay.py").read_text(encoding="utf-8")
    assert '"observed_states_19d"' in replay
    assert '"replay_command_tracking"' in replay
    assert '"replay_observation_matching"' in replay
    assert '"comparison_initialization"' in replay
    assert '"eligible_for_initial_frame_comparison"' in replay
    assert "source_observed_state_19d" in replay


def test_in_process_runner_applies_config_seed_to_environment() -> None:
    runner = (ROOT / "tools" / "run_in_process_eval.py").read_text(encoding="utf-8")
    assert "env.seed(seed)" in runner
    assert "env.reset(seed=seed)" in runner


def test_camera_patch_accepts_head_and_per_wrist_intrinsic_overrides() -> None:
    patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text(
        encoding="utf-8"
    )
    assert '"first_person_camera": "FLIP_TABLE_HEAD_LEFT_CAMERA"' in patch
    assert '"head_right_camera": "FLIP_TABLE_HEAD_RIGHT_CAMERA"' in patch
    assert '"left_hand_camera": "FLIP_TABLE_LEFT_WRIST_CAMERA"' in patch

    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    container_runner = (ROOT / "run_eval_in_container.sh").read_text(encoding="utf-8")
    for name in (
        "FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_POS",
        "FLIP_TABLE_HEAD_LEFT_CAMERA_OFFSET_ROT",
        "FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_POS",
        "FLIP_TABLE_HEAD_RIGHT_CAMERA_OFFSET_ROT",
        "FLIP_TABLE_LEFT_WRIST_CAMERA_FOCAL_LENGTH",
        "FLIP_TABLE_RIGHT_WRIST_CAMERA_FOCAL_LENGTH",
        "FLIP_TABLE_RANDOMIZE_LIGHTING",
    ):
        assert name in runner
        assert name in container_runner


def test_camera_patch_normalizes_bare_offset_tuples_before_writing_python() -> None:
    path = ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py"
    spec = importlib.util.spec_from_file_location("flip_table_camera_patch", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module._env_camera_tuple("TEST_CAMERA", "1,2,3,4", 4) == "(1, 2, 3, 4)"


def test_visual_evidence_frame_selection_keeps_start_and_end() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.extract_visual_evidence"
    )
    assert module.frame_indices(867)[0] == 0
    assert module.frame_indices(867)[-1] == 866
    assert 10 in module.frame_indices(867)


def test_replay_camera_frame_map_includes_requested_cad_frames() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.replay"
    )
    mapping = module.camera_frame_map(100, additional_source_frames=(20, 30, 40, 50))
    values = {item["source_frame"]: item["simulator_step"] for item in mapping}
    assert values[0] == 119
    assert values[20] == 153
    assert values[50] == 203


def test_table_motion_comparison_uses_only_paired_stereo_source_poses(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.table_motion_comparison"
    )

    def pose(x: float) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

    frames = []
    for frame, x in ((0, 0.0), (10, 0.03), (20, 0.06)):
        frames.append(
            {
                "frame_index": frame,
                "stereo_agreement": {"accepted": True},
                "eyes": {
                    "head_left": {"accepted": True, "root_from_table": pose(x)},
                    "head_right": {"accepted": True, "root_from_table": pose(x)},
                },
            }
        )
    alignment = tmp_path / "alignment.json"
    _write_json(
        alignment,
        {
            "accepted_for_fixed_scene_proposal": True,
            "source": {"episode_index": 9},
            "stereo_agreement": {
                "passes_internal_gate": True,
                "accepted_translation_p95_m": 0.001,
                "accepted_rotation_p95_deg": 0.1,
            },
            "frames": frames,
        },
    )
    actions = tmp_path / "replay_actions.json"
    _write_json(
        actions,
        {
            "camera_frame_map": [
                {"source_frame": 0, "simulator_step": 119},
                {"source_frame": 10, "simulator_step": 137},
                {"source_frame": 20, "simulator_step": 153},
            ]
        },
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "simulator_scene_diagnostics": {
                        "root_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "white_table": {
                            "position_world_m": [x, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                }
            )
            for step, x in ((119, 0.0), (137, 0.03), (153, 0.06))
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.compare(alignment, actions, trace)
    assert report["decision"] == "measured"
    assert report["source_episode_index"] == 9
    assert report["metrics"]["table_translation_rmse_m"] == pytest.approx(0.0)
    assert report["metrics"]["table_rotation_rmse_deg"] == pytest.approx(0.0)
    assert report["metrics"]["phase_timing_max_error_s"] == pytest.approx(1.0 / 37.5)


def test_table_motion_comparison_refuses_unvalidated_temporal_tracker(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.table_motion_comparison"
    )
    tracker = tmp_path / "temporal_tracker.json"
    _write_json(
        tracker,
        {
            "schema_version": "team_ramen_flip_table_temporal_stereo_cad_tracker/v1",
            "source": {"episode_index": 9},
            "summary": {"accepted_for_table_motion_metric": False},
            "measurement_uncertainty": {
                "independent_metric_bound": {"status": "unavailable", "passed": False}
            },
            "records": [],
        },
    )
    report = module.compare(tracker, tmp_path / "unused_actions.json", tmp_path / "unused_trace.jsonl")
    assert report["decision"] == "temporal_source_uncertainty_not_independently_validated"
    assert report["metrics"] == {}


def test_table_motion_comparison_uses_only_hash_verified_foundationpose_observations(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.table_motion_comparison"
    )

    def write_array(name: str, value: np.ndarray) -> dict[str, object]:
        path = tmp_path / name
        np.save(path, value, allow_pickle=False)
        return {
            "path": path.name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    poses = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    poses[:, 0, 3] = (0.0, 0.03, 0.06)
    tracker = tmp_path / "foundationpose_manifest.json"
    _write_json(
        tracker,
        {
            "schema_version": "team_ramen_foundationpose_table_track/v48",
            "episode_index": 9,
            "accepted": True,
            "gate": {
                "pass": True,
                "bidirectional_pass": True,
                "rendered_alignment_pass": True,
                "pose_evidence_pass": True,
                "p95_bidirectional_translation_error_m": 0.001,
                "p95_bidirectional_rotation_error_rad": np.deg2rad(0.1),
            },
            "arrays": {
                "source_frame_indices": write_array(
                    "source_frame_indices.npy", np.asarray((0, 10, 20), dtype=np.int64)
                ),
                "table_pose_root_sampled": write_array(
                    "table_pose_root_sampled.npy", poses
                ),
                "bidirectional_translation_error_m": write_array(
                    "bidirectional_translation_error_m.npy", np.asarray((0.001, 0.001, 0.001))
                ),
                "bidirectional_rotation_error_rad": write_array(
                    "bidirectional_rotation_error_rad.npy", np.deg2rad(np.asarray((0.1, 0.1, 0.1)))
                ),
            },
            "frames": [
                {
                    "ordinal": ordinal,
                    "backward_mode": "track",
                    "pose_evidence": {"passes_gate": True},
                }
                for ordinal in range(3)
            ],
        },
    )
    actions = tmp_path / "replay_actions.json"
    _write_json(
        actions,
        {
            "camera_frame_map": [
                {"source_frame": 0, "simulator_step": 119},
                {"source_frame": 10, "simulator_step": 137},
                {"source_frame": 20, "simulator_step": 153},
            ]
        },
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": step,
                    "simulator_scene_diagnostics": {
                        "root_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "white_table": {
                            "position_world_m": [x, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                }
            )
            for step, x in ((119, 0.0), (137, 0.03), (153, 0.06))
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.compare(tracker, actions, trace)
    assert report["decision"] == "measured"
    assert report["samples"] == 3
    assert report["metrics"]["table_translation_rmse_m"] == pytest.approx(0.0)


def test_table_motion_comparison_canonicalizes_table_yaw_symmetry_over_time(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.table_motion_comparison"
    )
    identity = np.eye(4, dtype=np.float64)
    half_turn = identity.copy()
    half_turn[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    alignment = tmp_path / "alignment.json"
    _write_json(
        alignment,
        {
            "source": {"episode_index": 4},
            "stereo_agreement": {
                "passes_internal_gate": True,
                "accepted_translation_p95_m": 0.001,
                "accepted_rotation_p95_deg": 0.1,
            },
            "frames": [
                {
                    "frame_index": 0,
                    "stereo_agreement": {"accepted": True},
                    "eyes": {
                        "head_left": {"accepted": True, "root_from_table": identity.tolist()},
                        "head_right": {"accepted": True, "root_from_table": identity.tolist()},
                    },
                },
                {
                    "frame_index": 10,
                    "stereo_agreement": {"accepted": True},
                    "eyes": {
                        "head_left": {"accepted": True, "root_from_table": half_turn.tolist()},
                        "head_right": {"accepted": True, "root_from_table": half_turn.tolist()},
                    },
                },
                {
                    "frame_index": 20,
                    "stereo_agreement": {"accepted": True},
                    "eyes": {
                        "head_left": {"accepted": True, "root_from_table": identity.tolist()},
                        "head_right": {"accepted": True, "root_from_table": identity.tolist()},
                    },
                },
            ],
        },
    )
    actions = tmp_path / "replay_actions.json"
    _write_json(
        actions,
        {"camera_frame_map": [{"source_frame": frame, "simulator_step": 119 + frame} for frame in (0, 10, 20)]},
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": 119 + frame,
                    "simulator_scene_diagnostics": {
                        "root_pose_world_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "white_table": {
                            "position_world_m": [0.0, 0.0, 0.0],
                            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                }
            )
            for frame in (0, 10, 20)
        )
        + "\n",
        encoding="utf-8",
    )
    report = module.compare(alignment, actions, trace)
    assert report["metrics"]["table_rotation_rmse_deg"] == pytest.approx(0.0)


def test_temporal_tracker_cad_roi_requires_a_complete_prediction_context() -> None:
    # The model-training test environment intentionally does not carry Isaac
    # geometry dependencies such as scipy/trimesh. The executable tracker is
    # exercised in the calibrated ``tv`` environment; keep this unit contract
    # import-free so it still protects the release code's no-partial-ROI rule.
    tracker = (ROOT / "real_to_sim_calibration" / "temporal_cad_tracker.py").read_text(
        encoding="utf-8"
    )
    assert "CAD ROI requires CAD points, predicted table pose, and left-camera pose" in tracker
    assert "cad_roi_max_depth_error_m" in tracker
    assert "cKDTree(pixels).query" in tracker
    assert "left_right_consistency_mask" in tracker
    assert "table_stereo_consistent_fraction" in tracker


def test_source_cad_alignment_never_falls_back_to_a_monocular_reset_pose() -> None:
    source = (ROOT / "real_to_sim_calibration" / "source_cad_alignment.py").read_text(
        encoding="utf-8"
    )
    assert "fewer than three stereo-consistent CAD frame pairs" in source
    assert "robust_fixed_pose(stereo_accepted)" in source
    assert "else accepted" not in source


def test_source_head_mount_consensus_requires_distinct_episodes(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_head_mount_consensus"
    )
    report = {
        "schema_version": "team_ramen_flip_table_source_head_mount_candidate/v1",
        "source_alignment": "episode_1.json",
        "correction": {
            "head_stereo_offset_local_m": [0.02, 0.01, 0.004],
            "head_stereo_rotation_rpy_deg": [0.0, 3.0, 2.5],
        },
    }
    other = {
        **report,
        "source_alignment": "episode_2.json",
        "correction": {
            "head_stereo_offset_local_m": [0.021, 0.009, 0.004],
            "head_stereo_rotation_rpy_deg": [0.0, 3.1, 2.6],
        },
    }
    result = module.consensus([(tmp_path / "one.json", report), (tmp_path / "two.json", other)])
    assert result["accepted_for_fixed_scene_probe"] is True
    assert result["accepted_for_shared_simulator_default"] is False
    with pytest.raises(ValueError, match="distinct"):
        module.consensus([(tmp_path / "one.json", report), (tmp_path / "two.json", report)])


def test_source_head_mount_consensus_accepts_the_explicit_incremental_field(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_head_mount_consensus"
    )
    candidate = {
        "schema_version": "team_ramen_flip_table_source_head_mount_candidate/v1",
        "source_alignment": "episode_177.json",
        "incremental_correction": {
            "head_stereo_offset_local_m": [0.01, 0.0, -0.01],
            "head_stereo_rotation_rpy_deg": [0.0, 1.0, 0.0],
        },
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    normalized = module._load(path)
    assert normalized["correction"] == candidate["incremental_correction"]


def test_fixed_scene_probe_candidate_never_promotes_a_probe_to_a_default() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.fixed_scene_probe_candidate"
    )
    scene = {
        "schema_version": "team_ramen_flip_table_source_scene_candidate/v1",
        "source_alignment": "source.json",
        "candidates": [{"offset_local_m": [0.0, 0.1, 0.0], "yaw_rad": 0.1}],
    }
    consensus = {
        "schema_version": "team_ramen_flip_table_source_head_mount_consensus/v1",
        "reports": ["one.json", "two.json"],
        "accepted_for_fixed_scene_probe": True,
        "shared_head_stereo_offset_local_m": [0.02, 0.01, 0.004],
        "shared_head_stereo_rotation_rpy_deg": [0.0, 3.0, 2.5],
    }
    result = module.compose(scene, consensus)
    assert result["accepted_for_fixed_scene_probe"] is True
    assert result["accepted_for_shared_simulator_default"] is False


def test_heldout_report_only_accepts_typed_episode_bound_comparisons(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.heldout_episode_report"
    )
    comparison = tmp_path / "head.json"
    _write_json(
        comparison,
        {
            "schema_version": "team_ramen_multiframe_head_geometry/v1",
            "source_episode_index": 9,
            "metrics": {"camera_reprojection_median_px": 2.0, "mask_iou": 0.95},
            "metric_sources": {
                "camera_reprojection_median_px": "recorded comparison",
                "mask_iou": "recorded comparison",
            },
        },
    )
    metrics, sources = module._derived_metrics(
        path=comparison,
        expected_schema="team_ramen_multiframe_head_geometry/v1",
        source_episode_index=9,
        allowed={"camera_reprojection_median_px", "mask_iou"},
    )
    assert metrics == {"camera_reprojection_median_px": 2.0, "mask_iou": 0.95}
    assert set(sources) == set(metrics)
    with pytest.raises(ValueError, match="episode"):
        module._derived_metrics(
            path=comparison,
            expected_schema="team_ramen_multiframe_head_geometry/v1",
            source_episode_index=10,
            allowed={"camera_reprojection_median_px"},
        )


def test_scene_candidate_comparison_fails_closed_without_reliable_real_pnp() -> None:
    path = ROOT / "real_to_sim_calibration" / "compare_scene_candidates.py"
    source = path.read_text(encoding="utf-8")
    assert '"decision": "rejected_pending_multiview_fit"' in source
    assert '"accepted_candidate": None' in source
    assert "real_confidence_min" in source


def test_multiframe_head_comparison_requires_multiple_reliable_frames() -> None:
    path = ROOT / "real_to_sim_calibration" / "compare_multiframe_head.py"
    source = path.read_text(encoding="utf-8")
    assert "len(reliable) >= 3" in source
    assert "insufficient_reliable_rgb_geometry" in source


def test_head_stereo_geometry_is_offline_only_and_uses_the_pinned_baseline() -> None:
    path = ROOT / "real_to_sim_calibration" / "stereo_geometry.py"
    source = path.read_text(encoding="utf-8")
    assert "offline calibration diagnostic only" in source
    assert "policy_use" in source
    assert "0.04 <= result.baseline_m <= 0.08" in source
    assert "StereoSGBM_create" in source


def test_source_cad_alignment_uses_stereo_and_cad_edges_without_four_corner_pnp() -> None:
    source = (ROOT / "real_to_sim_calibration" / "source_cad_alignment.py").read_text(
        encoding="utf-8"
    )
    assert "CAD rim and four leg axes" in source
    assert "no four-corner PnP" in source
    assert "SOURCE_CAD_BODY_CENTER_Z_CANDIDATES_M" in source
    assert "cad_body_center_z_candidates_m=SOURCE_CAD_BODY_CENTER_Z_CANDIDATES_M" in source
    assert "source CAD wireframe registration is unavailable" in source
    assert "estimator._estimate_from_cad_wireframe" in source
    assert "requires_simulator_ground_truth\": False" in source
    assert "root_from_right_opencv" in source
    assert "robust_fixed_pose" in source
    assert "TabletopPoseEstimator(" in source
    assert "MINIMUM_EYE_CONFIDENCE" in source
    assert "accepted_paired_frames" in source
    assert '"root_from_opencv_camera"' in source


def test_source_projection_conformance_uses_partial_cad_fit_not_visible_corner_assumptions() -> None:
    source = (ROOT / "real_to_sim_calibration" / "source_projection_conformance.py").read_text(
        encoding="utf-8"
    )
    assert "no source frame must expose four corners" in source
    assert "source stereo/FK/CAD partial-feature fit followed by CAD reprojection" in source
    assert "Rendered RGB colour/texture similarity is deliberately not a geometric acceptance metric." in source
    assert '"uses_simulator_ground_truth": False' in source
    assert "policy features" in source


def test_source_cad_alignment_canonicalizes_the_physical_yaw_symmetry() -> None:
    pytest.importorskip("scipy")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_cad_alignment"
    )
    first = np.eye(4)
    rotated = first @ module.TABLE_YAW_180_SYMMETRY
    canonical = module._canonical_table_pose(rotated, first)
    np.testing.assert_allclose(canonical, first, atol=1.0e-12)


def test_source_scene_candidate_is_reset_only_and_uses_workbench_coordinates() -> None:
    source = (ROOT / "real_to_sim_calibration" / "source_scene_candidate.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "container_overlay" / "robofinals_tasks" / "local_auto_tasks" / "assemble_table_task.py").read_text(
        encoding="utf-8"
    )
    assert "policy_use\": \"forbidden: offline reset calibration only" in source
    assert "offset_workbench" in source
    assert "synchronized_root_pose_reference" in source
    assert "robot_root_pos_local_m" in source
    assert "never a per-frame root teleport" in source
    assert "unapplied_vertical_offset_m" in source
    assert "fixed_to_zero_for_physical_workbench_support" in source
    assert "TABLE_YAW_180_SYMMETRY" in source
    assert "_workbench_support_envelope" in source
    assert "CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION = 0.70" in source
    assert '"accepted_for_v1_probe"' in source
    assert '"workbench": None' in task
    assert '"policy_camera_poses": policy_camera_poses' in task
    assert "calibration_adjusts_camera" in task
    assert "table-only reset candidate" in task
    assert "and not self._policy_camera_mount_defaults" in task
    assert "Fabric-side pose cache" in task
    assert "refreshed assembled table joints" in task
    assert "calibration robot candidate" in task


def test_source_scene_candidate_uses_the_synchronized_warmup_trace_step(tmp_path: Path) -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_scene_candidate"
    )
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join((
            json.dumps({"step": 0, "simulator_scene_diagnostics": {"marker": "reset"}}),
            json.dumps({"step": 119, "simulator_scene_diagnostics": {"marker": "warmup"}}),
        ))
        + "\n",
        encoding="utf-8",
    )
    assert module._trace_diagnostics_at_step(trace, 119) == {"marker": "warmup"}
    with pytest.raises(ValueError, match="synchronized step 120"):
        module._trace_diagnostics_at_step(trace, 120)


def test_source_scene_candidate_preserves_the_recorded_v1_reset_root() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_scene_candidate"
    )
    position, yaw = module._reset_robot_pose(
        {"randomization": {"robot": {"position_local_m": [-0.8, 2.37, 0.78], "yaw_rad": 3.14}}}
    )
    np.testing.assert_allclose(position, [-0.8, 2.37, 0.78])
    assert yaw == pytest.approx(3.14)


def test_source_scene_candidate_prefers_small_yaw_for_180_degree_table_symmetry() -> None:
    pytest.importorskip("scipy")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_scene_candidate"
    )
    base = np.diag((1.0, -1.0, -1.0))
    small_yaw = 0.109
    desired = np.array(
        [
            [-np.cos(small_yaw), -np.sin(small_yaw), 0.0],
            [-np.sin(small_yaw), np.cos(small_yaw), 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    np.testing.assert_allclose(module._yaw_delta(base, desired), small_yaw, atol=1.0e-12)


def test_source_head_mount_candidate_is_trace_based_and_offline_only() -> None:
    source = (ROOT / "real_to_sim_calibration" / "source_head_mount_candidate.py").read_text(
        encoding="utf-8"
    )
    assert '"policy_use": "forbidden: offline reset calibration only"' in source
    assert "root_from_opencv_camera" in source
    assert "policy_camera_poses" in source
    assert "torso_link" in source
    assert "head_stereo_offset_local_m" in source


def test_source_head_mount_offset_matches_the_runtime_rig_pivot() -> None:
    pytest.importorskip("scipy")
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_head_mount_candidate"
    )
    left = np.eye(4)
    left[:3, 3] = [0.10, 0.02, 0.40]
    right = np.eye(4)
    right[:3, 3] = [0.10, -0.04, 0.40]
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    desired = left.copy()
    desired[:3, :3] = rotation
    desired[:3, 3] = [0.16, 0.08, 0.42]
    offset = module._rig_centred_offset(
        torso_from_current_left=left,
        torso_from_current_right=right,
        torso_from_desired_left=desired,
        rotation=rotation,
    )
    center = module._stereo_rig_translation(left, right)
    runtime_left_position = center + offset + rotation @ (left[:3, 3] - center)
    np.testing.assert_allclose(runtime_left_position, desired[:3, 3], atol=1.0e-12)


def test_head_mount_reapplication_composes_instead_of_overwriting() -> None:
    module = importlib.import_module(
        "evaluate.flip_table_simulation.real_to_sim_calibration.source_head_mount_candidate"
    )
    result = module.compose_task_stereo_correction(
        {
            "head_stereo_offset_local_m": [0.02, -0.01, 0.03],
            "head_stereo_rotation_rpy_deg": [0.0, 0.0, 10.0],
        },
        {
            "head_stereo_offset_local_m": [0.01, 0.02, -0.04],
            "head_stereo_rotation_rpy_deg": [0.0, 0.0, 5.0],
        },
    )
    np.testing.assert_allclose(result["head_stereo_offset_local_m"], [0.03, 0.01, -0.01])
    np.testing.assert_allclose(result["head_stereo_rotation_rpy_deg"], [0.0, 0.0, 15.0])
    assert result["head_stereo_correction_composition"]["rotation"] == "R_incremental @ R_candidate"


def test_actuator_identification_reports_rate_independent_time_constants() -> None:
    path = ROOT / "real_to_sim_calibration" / "actuator_identification.py"
    source = path.read_text(encoding="utf-8")
    assert '"policy_use": "forbidden: offline simulator-actuator calibration only"' in source
    assert 'entry["time_constant_s"]' in source
    assert "math.log1p(-alpha)" in source
    assert "ARM_CHANNEL_NAMES" in source
    assert '"group_summaries"' in source
    assert '"arms": _group_summary(channels, ARM_CHANNEL_NAMES)' in source
    assert "--max-delay-s" in source
    assert "--source-frame-start" in source
    assert "contact-dominated rows out of the fit" in source
    assert '"sim_to_real_encoder_match"' in source
    assert "does not alter replay timing" in source


def test_calibrated_arm_profile_is_applied_to_ideal_pd_before_scene_creation() -> None:
    patch = (ROOT / "container_overlay" / "patches" / "patch_g1_global_camera.py").read_text(
        encoding="utf-8"
    )
    assert "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V5" in patch
    assert "FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE" in patch
    assert "FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE" in patch
    assert "FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE" in patch
    assert "FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM" in patch
    assert "FLIP_TABLE_CALIBRATION_DEX1_STIFFNESS_SCALE" in patch
    assert "FLIP_TABLE_CALIBRATION_DEX1_DAMPING_SCALE" in patch
    assert "selected calibrated arm IdealPD profile" in patch
    runner = (ROOT / "run_eval.sh").read_text(encoding="utf-8")
    for name in (
        "FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE",
        "FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE",
        "FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE",
        "FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM",
        "FLIP_TABLE_CALIBRATION_DEX1_STIFFNESS_SCALE",
        "FLIP_TABLE_CALIBRATION_DEX1_DAMPING_SCALE",
    ):
        assert name in runner


def _load_heldout_validation():
    path = ROOT / "real_to_sim_calibration" / "heldout_validation.py"
    spec = importlib.util.spec_from_file_location("flip_table_heldout_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _heldout_metrics() -> dict[str, float]:
    return {
        "camera_reprojection_median_px": 2.0,
        "camera_reprojection_p95_px": 7.0,
        "upper_body_joint_rmse_rad": 0.02,
        "table_translation_rmse_m": 0.01,
        "table_rotation_rmse_deg": 2.0,
        "phase_timing_max_error_s": 0.08,
        "mask_iou": 0.95,
    }


def _heldout_sources() -> dict[str, str]:
    return {name: "recorded comparison artifact" for name in _heldout_metrics()}


def test_heldout_acceptance_requires_all_five_episodes_and_one_frozen_parameter_set(tmp_path: Path) -> None:
    module = _load_heldout_validation()
    indices = (9, 66, 74, 308, 338)
    bundles = {}
    for index in indices:
        bundle_path = tmp_path / f"validation_{index:04d}.json"
        _write_json(bundle_path, {"source_episode_index": index})
        bundles[f"validation_{index:04d}"] = str(bundle_path)
    manifest_path = tmp_path / "calibration_manifest.json"
    _write_json(manifest_path, {"episode_bundles": bundles})
    parameter_path = tmp_path / "shared_parameters.json"
    _write_json(parameter_path, {"fixed": True})
    digest = hashlib.sha256(parameter_path.read_bytes()).hexdigest()
    reports = []
    for index in indices:
        report_path = tmp_path / f"report_{index:04d}.json"
        _write_json(
            report_path,
            {
                "schema_version": module.SCHEMA_VERSION,
                "source_episode_index": index,
                "shared_parameter_sha256": digest,
                "shared_parameters_path": str(parameter_path),
                "metrics": _heldout_metrics(),
                "metric_sources": _heldout_sources(),
            },
        )
        reports.append(report_path)
    result = module.evaluate(manifest_path, tuple(reports))
    assert result["passed"] is True
    assert result["decision"] == "accepted"
    assert result["shared_parameters_frozen"] is True


def test_heldout_acceptance_fails_closed_when_a_required_metric_or_episode_is_missing(tmp_path: Path) -> None:
    module = _load_heldout_validation()
    indices = (9, 66, 74, 308, 338)
    bundles = {}
    for index in indices:
        bundle_path = tmp_path / f"validation_{index:04d}.json"
        _write_json(bundle_path, {"source_episode_index": index})
        bundles[f"validation_{index:04d}"] = str(bundle_path)
    manifest_path = tmp_path / "calibration_manifest.json"
    _write_json(manifest_path, {"episode_bundles": bundles})
    parameter_path = tmp_path / "shared_parameters.json"
    _write_json(parameter_path, {"fixed": True})
    incomplete = _heldout_metrics()
    incomplete.pop("mask_iou")
    report_path = tmp_path / "report_0009.json"
    _write_json(
        report_path,
        {
            "schema_version": module.SCHEMA_VERSION,
            "source_episode_index": 9,
            "shared_parameter_sha256": hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
            "shared_parameters_path": str(parameter_path),
            "metrics": incomplete,
            "metric_sources": _heldout_sources(),
        },
    )
    result = module.evaluate(manifest_path, (report_path,))
    assert result["passed"] is False
    assert result["decision"] == "rejected_or_incomplete"
    first = next(item for item in result["episodes"] if item["source_episode_index"] == 9)
    assert first["gate"]["metrics"]["mask_iou"]["status"] == "missing"
    assert sum(item["status"] == "missing_report" for item in result["episodes"]) == 4
