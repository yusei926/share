from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import model.subtask_policy_training.deployment.real_furniture_groot_n17_worker as worker_module
from inference.desktop.upper_policy.furniture_groot_contract import (
    CAMERA_KEYS,
    TASK_TEXT,
    camera_payload_history,
    compose_model_state,
    extract_executable_action,
    validate_checkpoint_metadata,
)
from inference.desktop.upper_policy.run_flip_table_diffusion import (
    validate_policy_chunk,
)
from inference.desktop.upper_policy.run_flip_table_furniture_groot import (
    InferenceRequestContext,
    TemporalObservationBuffer,
    release_execution_schedule,
)
from model.subtask_policy_training.deployment.real_furniture_groot_n17_worker import (
    _decode_rgb_history,
)
from model.subtask_policy_training.gr00t.dex1_hand_synergy import (
    ASSET_PATH,
    dex1_to_hand,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    REAL_G1_RELATIVE_EEF_ACTION_SLICES,
    REAL_G1_RELATIVE_EEF_STATE_SLICES,
    source_euler_xyz_pose_to_xyz_rot6d,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION,
    DATASET_REPO_ID,
    DATASET_REVISION,
    DEX1_SYNERGY_SHA256,
    EXPECTED_SHA256,
    EXPECTED_TUNING_SCOPE,
    validate_eef_fk_release_audit,
    validate_furniture_training_candidate,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporal_validation_fixture(
    selected: tuple[str, int] = ("-0.1", 10),
) -> dict[str, object]:
    candidates = []
    for temporal_lambda in ("none", "-0.25", "-0.1", "0"):
        for execution_steps in (5, 10, 20):
            is_selected = (temporal_lambda, execution_steps) == selected
            success_count = 5 if is_selected else 4
            candidates.append(
                {
                    "name": (
                        f"lambda_{temporal_lambda}_exec_{execution_steps}"
                    ),
                    "temporal_lambda": temporal_lambda,
                    "execution_steps": execution_steps,
                    "mode": "randomized_validation",
                    "runtime_evaluation_mode": "randomized",
                    "domain_randomization_profile": "validation_v1",
                    "seed": 92001,
                    "policy_inference_seed": 92001,
                    "episode_inference_seeds": [
                        92001 + index for index in range(5)
                    ],
                    "episode_ids": [
                        f"92001:{index}" for index in range(5)
                    ],
                    "test_count": 5,
                    "success_count": success_count,
                    "success_rate": success_count / 5,
                    "trace": {
                        "target_jerk_rms": 0.2,
                        "target_acceleration_rms": 0.1,
                        "tracking_rmse": 0.02,
                    },
                }
            )
    return {
        "schema_version": "team_ramen_groot_n17_temporal_sweep/v1",
        "scripted_controller_tracking": {
            "arm_rmse_rad": 0.02,
            "arm_p95_abs_error_rad": 0.04,
            "actual_arm_range_rad": 0.2,
            "passed": True,
        },
        "candidates": candidates,
        "selected": {
            "name": f"lambda_{selected[0]}_exec_{selected[1]}",
            "temporal_lambda": selected[0],
            "execution_steps": selected[1],
            "success_rate": 1.0,
        },
    }


def _write_release_sidecars(root: Path) -> dict[str, object]:
    progress = root / "progress.jsonl"
    visual = root / "visual_rotation.jsonl"
    contact = root / "orientation_contact_sheet.jpg"
    approval = root / "orientation_contact_sheet.approved"
    progress.write_text("{}\n")
    visual.write_text("{}\n")
    contact.write_bytes(b"fixture-contact-sheet")
    approval.write_text(_sha256(contact) + "\n")
    progress_manifest = root / "progress_manifest.json"
    progress_manifest.write_text(
        json.dumps(
            {
                "schema_version": "flip_table_progress_sidecar_manifest_v1",
                "dataset_repo_id": DATASET_REPO_ID,
                "dataset_revision": DATASET_REVISION,
                "annotation_file": progress.name,
                "annotation_sha256": _sha256(progress),
                "summary": {"episode_count": 174},
            }
        )
    )
    visual_manifest = root / "visual_rotation_manifest.json"
    visual_manifest.write_text(
        json.dumps(
            {
                "schema_version": "flip_table_visual_rotation_manifest_v1",
                "dataset_repo_id": DATASET_REPO_ID,
                "dataset_revision": DATASET_REVISION,
                "episode_count": 174,
                "sidecar_sha256": _sha256(visual),
                "contact_sheet": contact.name,
                "contact_sheet_sha256": _sha256(contact),
                "contact_sheet_human_review_required": True,
                "policy_input": False,
            }
        )
    )
    return {
        "progress_manifest_sha256": _sha256(progress_manifest),
        "visual_rotation_manifest_sha256": _sha256(visual_manifest),
        "artifacts": {
            name: _sha256(root / name)
            for name in (
                "progress.jsonl",
                "visual_rotation.jsonl",
                "orientation_contact_sheet.jpg",
                "orientation_contact_sheet.approved",
            )
        },
        "contact_sheet_review": {
            "required": True,
            "approved_sha256": _sha256(contact),
        },
    }


def _write_checkpoint(root: Path) -> None:
    model = root / "model.safetensors"
    model.write_bytes(b"fixture-model")
    config = {
        "type": "furniture_groot",
        "base_model_path": "nvidia/GR00T-N1.7-3B",
        "base_model_revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        "embodiment_tag": "real_g1_relative_eef_relative_joints",
        "chunk_size": 40,
        "n_action_steps": 10,
        "max_state_dim": 132,
        "max_action_dim": 132,
        "valid_action_dim": 46,
        "use_relative_actions": True,
        "relative_exclude_joints": ["hand", "waist", "base_height", "navigate"],
        "action_decode_transform": None,
        "progress_enabled": True,
        **EXPECTED_TUNING_SCOPE,
        "input_features": {
            "observation.state": {"shape": [49]},
            "observation.images.head_left": {"shape": [3, 480, 640]},
            "observation.images.left_wrist": {"shape": [3, 480, 640]},
            "observation.images.right_wrist": {"shape": [3, 480, 640]},
        },
        "output_features": {"action": {"shape": [53]}},
    }
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config))
    (root / "policy_preprocessor.json").write_text("{}")
    (root / "policy_postprocessor.json").write_text("{}")
    (root / "dex1_g1_synergy.json").write_bytes(ASSET_PATH.read_bytes())
    eef_fk_audit = root / "eef_fk_audit.json"
    eef_fk_payload = {
        "source_repo_id": DATASET_REPO_ID,
        "source_revision": DATASET_REVISION,
        "configured_eef_order": ["left", "right"],
        "eef_pose_format": "xyz_euler_xyz_rad",
        "eef_reference_frame": "robot_root",
        "pass": True,
        "action_fk_residual_pass": True,
        "frame_assignment_pass": True,
        "thresholds": {
            "position_p95_m_max": 0.08,
            "rotation_p95_rad_max": 0.2,
            "swapped_score_ratio_min": 4.0,
        },
        "swapped_to_configured_score_ratio": 6.0,
        "tool_transforms": {
            side: {
                "parent_frame": f"{side}_wrist_yaw_link",
                "translation_m": [0.05, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
            for side in ("left", "right")
        },
        "validation_metrics": {
            "action": {
                side: {
                    "position_error_m": {"p95": 0.07},
                    "rotation_error_rad": {"p95": 0.18},
                }
                for side in ("left", "right")
            }
        },
        "per_episode": [
            {
                "episode_index": episode_index,
                "action_fk_residual_pass": True,
            }
            for episode_index in range(174)
        ],
        "coverage": {
            "episode_count": 174,
            "episode_level_diagnostic_threshold_exceedances": [],
        },
        "mimic_source_episode_gate": {
            "eligible_count": 174,
            "rejected_count": 0,
        },
        "temporal_alignment": {
            "pass": True,
            "selected_offset_frames": 0,
            "material_improvement_threshold": 0.05,
        },
        "training_contract": {
            "eef_teacher": "action.ee_action",
            "joint_teacher": "action.robot_q_desired",
            "policy_action_mask_slots": "0:46",
            "teacher_pair_status": (
                "compatible_with_expected_ik_realization_residual"
            ),
        },
    }
    eef_fk_audit.write_text(json.dumps(eef_fk_payload))
    eef_fk_validation = validate_eef_fk_release_audit(eef_fk_payload)
    dataset = {
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "task_instruction": "flip table",
        "task_index": 0,
        "counts": {"train": 139, "validation": 17, "test": 18},
    }
    candidate_selection = {
        "selection_data": (
            "offline_validation_plus_same_seed_sim_validation"
        ),
        "selected": "auxiliary_progress",
        "candidate_hashes": {
            "baseline": "fixture-baseline-hash",
            "auxiliary_progress": _sha256(model),
        },
    }
    sim_comparison = root / "sim_candidate_selection.json"
    sim_comparison.write_text(
        json.dumps(
            {
                "schema_version": "groot_n17_sim_candidate_comparison_v1",
                "seed": 95001,
                "policy_inference_seed": 95001,
                "episode_ids": [
                    f"95001:{index}" for index in range(5)
                ],
                "offline_validation_episodes": list(range(139, 156)),
                "selection_data": "same_seed_randomized_sim_validation",
                "domain_randomization_profile": "validation_v1",
                "test_split_used": False,
                "candidates": {
                    "baseline": {
                        "model_safetensors_sha256": "fixture-baseline-hash",
                        "test_count": 5,
                        "success_count": 4,
                        "success_rate": 0.8,
                        "seed": 95001,
                        "policy_inference_seed": 95001,
                        "episode_inference_seeds": [
                            95001 + index for index in range(5)
                        ],
                        "episode_ids": [
                            f"95001:{index}" for index in range(5)
                        ],
                        "mode": "randomized_validation",
                        "runtime_evaluation_mode": "randomized",
                        "domain_randomization_profile": "validation_v1",
                        "offline_validation_episodes": list(range(139, 156)),
                        "offline_validation_score": 0.3,
                        "trace": {
                            "target_jerk_rms": 0.4,
                            "target_acceleration_rms": 0.3,
                            "tracking_rmse": 0.02,
                        },
                        "progress_enabled": False,
                    },
                    "auxiliary_progress": {
                        "model_safetensors_sha256": _sha256(model),
                        "test_count": 5,
                        "success_count": 5,
                        "success_rate": 1.0,
                        "seed": 95001,
                        "policy_inference_seed": 95001,
                        "episode_inference_seeds": [
                            95001 + index for index in range(5)
                        ],
                        "episode_ids": [
                            f"95001:{index}" for index in range(5)
                        ],
                        "mode": "randomized_validation",
                        "runtime_evaluation_mode": "randomized",
                        "domain_randomization_profile": "validation_v1",
                        "offline_validation_episodes": list(range(139, 156)),
                        "offline_validation_score": 0.2,
                        "trace": {
                            "target_jerk_rms": 0.2,
                            "target_acceleration_rms": 0.2,
                            "tracking_rmse": 0.01,
                        },
                        "progress_enabled": True,
                    },
                },
                "selected": "auxiliary_progress",
            }
        )
    )
    temporal_validation = _temporal_validation_fixture()
    temporal_validation_bytes = json.dumps(temporal_validation).encode()
    temporal_validation_sha256 = hashlib.sha256(
        temporal_validation_bytes
    ).hexdigest()
    sim_release = root / "sim_release_evaluation.json"
    sim_release.write_text(
        json.dumps(
            {
                "schema_version": (
                    "team_ramen_groot_n17_release_evaluation/v1"
                ),
                "candidate_name": "auxiliary_progress",
                "model_safetensors_sha256": _sha256(model),
                "selected_temporal_setting": {
                    "temporal_lambda": "-0.1",
                    "execution_steps": 10,
                },
                "temporal_validation": temporal_validation,
                "temporal_validation_sha256": temporal_validation_sha256,
                "scripted_controller_tracking": temporal_validation[
                    "scripted_controller_tracking"
                ],
                "fixed_scene": {
                    "test_count": 3,
                    "success_count": 3,
                    "seed": 93001,
                    "policy_inference_seed": 93001,
                    "episode_inference_seeds": [93001, 93002, 93003],
                    "episode_ids": [f"93001:{index}" for index in range(3)],
                    "mode": "nominal",
                    "runtime_evaluation_mode": "nominal",
                    "domain_randomization_profile": "nominal_v1",
                    "temporal_lambda": "-0.1",
                    "execution_steps": 10,
                },
                "unseen_dr": {
                    "test_count": 50,
                    "success_count": 40,
                    "seed": 94001,
                    "policy_inference_seed": 94001,
                    "episode_inference_seeds": [
                        94001 + index for index in range(50)
                    ],
                    "episode_ids": [
                        f"94001:{index}" for index in range(50)
                    ],
                    "mode": "unseen_dr",
                    "runtime_evaluation_mode": "unseen_dr",
                    "domain_randomization_profile": "held_out_v1",
                    "temporal_lambda": "-0.1",
                    "execution_steps": 10,
                },
                "release_goal": {"unseen_dr_passed": True},
                "claim_scope": (
                    "simulator evaluation only; no Sim-to-Real success claim"
                ),
            }
        )
    )
    sim_evaluation = root / "sim_evaluation"
    sim_evaluation.mkdir()
    sim_evaluation_file = sim_evaluation / "README.txt"
    sim_evaluation_file.write_text("fixture\n")
    temporal_selection = (
        sim_evaluation / "release" / "temporal_selection.json"
    )
    temporal_selection.parent.mkdir()
    temporal_selection.write_bytes(temporal_validation_bytes)
    sim_bundle_files = {
        path.relative_to(sim_evaluation).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in (sim_evaluation_file, temporal_selection)
    }
    sim_evaluation_manifest = root / "sim_evaluation_manifest.json"
    sim_evaluation_manifest.write_text(
        json.dumps(
            {
                "schema_version": "groot_n17_sim_evaluation_bundle_v1",
                "root": "sim_evaluation",
                "file_count": len(sim_bundle_files),
                "total_bytes": sum(
                    item["bytes"] for item in sim_bundle_files.values()
                ),
                "files": sim_bundle_files,
            }
        )
    )
    source_bundle = root / "source_snapshot"
    source_bundle.mkdir()
    source_fixture = source_bundle / "fixture.py"
    source_fixture.write_text("# fixture\n")
    source_snapshot = root / "source_snapshot_manifest.json"
    source_snapshot.write_text(
        json.dumps(
            {
                "schema_version": "groot_n17_source_snapshot_v2",
                "scope": "release_time_training_inference_and_evaluation_source",
                "captured_at_utc": "2026-07-30T00:00:00+00:00",
                "git_head": "fixture-head",
                "bundle_root": source_bundle.name,
                "file_count": 1,
                "files": {"fixture.py": _sha256(source_fixture)},
            }
        )
    )
    sidecars = _write_release_sidecars(root)
    (root / "training_run_record.json").write_text(
        json.dumps(
            {
                "source": {
                    "snapshot_scope": (
                        "release_time_training_inference_and_evaluation_source"
                    ),
                    "snapshot_captured_at_utc": "2026-07-30T00:00:00+00:00",
                    "git_head": "fixture-head",
                    "snapshot_manifest": source_snapshot.name,
                    "snapshot_manifest_sha256": _sha256(source_snapshot),
                    "snapshot_file_count": 1,
                },
                "contract": {
                    "repo_id": BASE_MODEL_REPO_ID,
                    "revision": BASE_MODEL_REVISION,
                    "sha256": EXPECTED_SHA256,
                },
                "training_scope": {"config_flags": EXPECTED_TUNING_SCOPE},
                "dex1_adapter": {"sha256": DEX1_SYNERGY_SHA256},
                "eef_fk_audit": {
                    "sha256": _sha256(eef_fk_audit),
                    "source_repo_id": DATASET_REPO_ID,
                    "source_revision": DATASET_REVISION,
                    "episode_count": 174,
                    "action_fk_residual_pass": True,
                    "frame_assignment_pass": True,
                    "selected_offset_frames": 0,
                    "teacher_pair_status": (
                        "compatible_with_expected_ik_realization_residual"
                    ),
                    "fixed_release_gate": eef_fk_validation,
                },
                "sidecars": sidecars,
                "simulation_evaluation": {
                    "candidate_comparison_sha256": _sha256(sim_comparison),
                    "release_evaluation_sha256": _sha256(sim_release),
                    "temporal_validation_sha256": _sha256(
                        temporal_selection
                    ),
                    "bundle_manifest_sha256": _sha256(
                        sim_evaluation_manifest
                    ),
                    "bundle_file_count": len(sim_bundle_files),
                    "bundle_total_bytes": sum(
                        item["bytes"] for item in sim_bundle_files.values()
                    ),
                    "selected_candidate": "auxiliary_progress",
                    "fixed_scene_dr_profile": "nominal_v1",
                    "unseen_dr_profile": "held_out_v1",
                },
            }
        )
    )
    (root / "training_manifest.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "contract": {
                    "logical_state_dim": 49,
                    "logical_action_dim": 53,
                    "packed_state_dim": 132,
                    "packed_action_dim": 132,
                    "valid_action_dim": 46,
                    "action_horizon": 40,
                    "physical_command": (
                        "14 arm joint targets + 2 absolute Dex1-1 commands"
                    ),
                    "policy_cameras": [
                        "head_left",
                        "left_wrist",
                        "right_wrist",
                    ],
                    "head_right_used": False,
                    "progress_in_action": False,
                    "task_instruction": "flip table",
                    "progress_head_shape": [40, 1],
                },
                "checkpoint": {
                    "model_safetensors_sha256": _sha256(model),
                    "config_sha256": _sha256(config_path),
                },
                "candidate_selection": candidate_selection,
                "wandb_url": "https://wandb.ai/test/project/runs/test",
            }
        )
    )


def _set_release_schedule(root: Path, temporal_lambda: str, execution_steps: int) -> None:
    temporal_validation = _temporal_validation_fixture(
        (temporal_lambda, execution_steps)
    )
    temporal_path = (
        root / "sim_evaluation" / "release" / "temporal_selection.json"
    )
    temporal_path.write_text(json.dumps(temporal_validation))

    release_path = root / "sim_release_evaluation.json"
    release = json.loads(release_path.read_text())
    release["selected_temporal_setting"] = {
        "temporal_lambda": temporal_lambda,
        "execution_steps": execution_steps,
    }
    release["temporal_validation"] = temporal_validation
    release["temporal_validation_sha256"] = _sha256(temporal_path)
    release["scripted_controller_tracking"] = temporal_validation[
        "scripted_controller_tracking"
    ]
    for stage in ("fixed_scene", "unseen_dr"):
        release[stage]["temporal_lambda"] = temporal_lambda
        release[stage]["execution_steps"] = execution_steps
    release_path.write_text(json.dumps(release))

    bundle_manifest_path = root / "sim_evaluation_manifest.json"
    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    bundle_entry = bundle_manifest["files"][
        "release/temporal_selection.json"
    ]
    bundle_entry["sha256"] = _sha256(temporal_path)
    bundle_entry["bytes"] = temporal_path.stat().st_size
    bundle_manifest["total_bytes"] = sum(
        item["bytes"] for item in bundle_manifest["files"].values()
    )
    bundle_manifest_path.write_text(json.dumps(bundle_manifest))

    record_path = root / "training_run_record.json"
    record = json.loads(record_path.read_text())
    record["simulation_evaluation"]["release_evaluation_sha256"] = _sha256(
        release_path
    )
    record["simulation_evaluation"]["temporal_validation_sha256"] = _sha256(
        temporal_path
    )
    record["simulation_evaluation"]["bundle_manifest_sha256"] = _sha256(
        bundle_manifest_path
    )
    record["simulation_evaluation"]["bundle_total_bytes"] = bundle_manifest[
        "total_bytes"
    ]
    record_path.write_text(json.dumps(record))


def test_physical_checkpoint_contract_is_exact(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    contract = validate_checkpoint_metadata(tmp_path)
    assert contract["task"] == TASK_TEXT == "flip table"
    assert contract["state_dim"] == 49
    assert contract["logical_action_dim"] == 53
    assert contract["executable_action_dim"] == 16
    assert contract["action_horizon"] == 40
    assert contract["execution_steps"] == 10
    assert contract["temporal_lambda"] == -0.1
    assert contract["temporal_lambda_label"] == "-0.1"
    assert contract["video_delta_indices"] == [-20, 0]
    assert contract["lower_body_command_dimensions"] == 0


def test_physical_checkpoint_uses_selected_none_temporal_schedule(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    _set_release_schedule(tmp_path, "none", 5)
    contract = validate_checkpoint_metadata(tmp_path)
    assert release_execution_schedule(contract) == (5, None, "none")


def test_release_schedule_rejects_label_value_mismatch() -> None:
    with pytest.raises(ValueError, match="differs"):
        release_execution_schedule(
            {
                "execution_steps": 10,
                "temporal_lambda_label": "none",
                "temporal_lambda": -0.1,
            }
        )


def test_h100_candidate_contract_is_exact(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    candidate = validate_furniture_training_candidate(
        tmp_path,
        expected_progress_enabled=True,
    )
    assert candidate["logical_state_dim"] == 49
    assert candidate["logical_action_dim"] == 53
    assert candidate["packed_action_dim"] == 132
    assert candidate["valid_action_dim"] == 46
    assert candidate["action_horizon"] == 40


def test_physical_worker_passes_seed_to_isolated_groot_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_checkpoint(tmp_path)
    received: dict[str, object] = {}

    class FakeRuntime:
        def __init__(self, **kwargs):
            received.update(kwargs)
            self.device = "cpu"

    monkeypatch.setattr(worker_module, "GrootRuntime", FakeRuntime)
    monkeypatch.setattr(worker_module.importlib.metadata, "version", lambda _: "0.6.0")

    worker_module.Runtime(tmp_path, "cpu", 95001)

    assert received["checkpoint"] == tmp_path
    assert received["device"] == "cpu"
    assert received["n_action_steps"] == 40
    assert received["seed"] == 95001


def test_physical_checkpoint_rejects_54d_action(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["output_features"]["action"]["shape"] = [54]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="53-D"):
        validate_checkpoint_metadata(tmp_path)


def test_physical_checkpoint_rejects_modified_source_bundle(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "source_snapshot" / "fixture.py").write_text("# modified\n")
    with pytest.raises(ValueError, match="source snapshot hash changed"):
        validate_checkpoint_metadata(tmp_path)


def test_physical_checkpoint_rejects_modified_progress_sidecar(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    (tmp_path / "progress.jsonl").write_text('{"modified":true}\n')
    with pytest.raises(ValueError, match="sidecars"):
        validate_checkpoint_metadata(tmp_path)


def test_physical_checkpoint_rejects_incomplete_sim_seed_evidence(
    tmp_path: Path,
) -> None:
    _write_checkpoint(tmp_path)
    comparison_path = tmp_path / "sim_candidate_selection.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["candidates"]["baseline"]["episode_inference_seeds"][2] = 123
    comparison_path.write_text(json.dumps(comparison))
    run_record_path = tmp_path / "training_run_record.json"
    run_record = json.loads(run_record_path.read_text())
    run_record["simulation_evaluation"]["candidate_comparison_sha256"] = _sha256(
        comparison_path
    )
    run_record_path.write_text(json.dumps(run_record))

    with pytest.raises(ValueError, match="same-seed simulator evidence"):
        validate_checkpoint_metadata(tmp_path)


def test_live_state_uses_dataset_fk_and_exact_slot_order() -> None:
    body = np.linspace(-0.7, 0.7, 29)
    dex1_fraction = np.asarray([0.25, 0.75])
    eef = np.asarray(
        [
            0.1,
            0.2,
            0.3,
            0.0,
            0.0,
            0.0,
            -0.1,
            0.4,
            0.2,
            0.0,
            0.0,
            np.pi / 2,
        ]
    )
    state = compose_model_state(body, dex1_fraction, eef)
    assert state.shape == (49,)
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["left_wrist_eef_9d"])],
        source_euler_xyz_pose_to_xyz_rot6d(eef[:6].tolist()),
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["right_wrist_eef_9d"])],
        source_euler_xyz_pose_to_xyz_rot6d(eef[6:].tolist()),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["left_arm"])],
        body[15:22],
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["right_arm"])],
        body[22:29],
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["waist"])],
        body[12:15],
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["left_hand"])],
        dex1_to_hand(1.125, side="left", kind="state"),
    )
    np.testing.assert_allclose(
        state[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["right_hand"])],
        dex1_to_hand(3.375, side="right", kind="state"),
    )


def test_h40_action_exposes_only_arms_and_dex1() -> None:
    logical = np.full((40, 53), 999.0)
    left_arm = np.linspace(-0.2, 0.2, 40 * 7).reshape(40, 7)
    right_arm = np.linspace(0.3, -0.3, 40 * 7).reshape(40, 7)
    logical[:, slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_arm"])] = left_arm
    logical[:, slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_arm"])] = right_arm
    for row in range(40):
        logical[
            row,
            slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_hand"]),
        ] = dex1_to_hand(row / 39 * 4.5, side="left", kind="action")
        logical[
            row,
            slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_hand"]),
        ] = dex1_to_hand((39 - row) / 39 * 4.5, side="right", kind="action")
    physical = extract_executable_action(logical)
    np.testing.assert_allclose(physical[:, :7], left_arm)
    np.testing.assert_allclose(physical[:, 7:14], right_arm)
    np.testing.assert_allclose(physical[:, 14], np.linspace(0.0, 4.5, 40))
    np.testing.assert_allclose(physical[:, 15], np.linspace(4.5, 0.0, 40))


def _observation(index: int) -> SimpleNamespace:
    timestamp = 1_000_000_000 + round(index / 30.0 * 1.0e9)
    return SimpleNamespace(
        stale_roles=(),
        camera_capture_monotonic_ns={
            "head_left": timestamp,
            "head_right": timestamp,
            "left_wrist": timestamp,
            "right_wrist": timestamp,
        },
        camera_stream_metadata={
            role: {"jpeg_generation": index + 1}
            for role in ("head_left", "head_right", "left_wrist", "right_wrist")
        },
        camera_jpeg={
            "head_left": b"head",
            "head_right": b"unused",
            "left_wrist": b"left",
            "right_wrist": b"right",
        },
    )


def test_temporal_buffer_selects_exact_minus_20_and_zero() -> None:
    buffer = TemporalObservationBuffer()
    observations = [_observation(index) for index in range(21)]
    for observation in observations:
        assert buffer.add(observation)
    assert buffer.pair() == [observations[0], observations[20]]
    payload = camera_payload_history(buffer.pair())
    assert tuple(payload) == CAMERA_KEYS
    assert all(len(frames) == 2 for frames in payload.values())
    assert all(b"unused" not in frames for frames in payload.values())


def test_async_inference_context_freezes_request_pose_and_h40_window() -> None:
    measured = np.linspace(-0.5, 0.5, 14)
    context = InferenceRequestContext(
        origin_step=20,
        measured_arm_rad=measured,
        submitted_monotonic_s=10.0,
    )
    measured[:] = 99.0
    np.testing.assert_allclose(
        context.measured_arm_rad,
        np.linspace(-0.5, 0.5, 14),
    )
    assert not context.measured_arm_rad.flags.writeable
    assert context.has_remaining_target(59)
    assert not context.has_remaining_target(60)
    assert context.age_seconds(10.75) == pytest.approx(0.75)


def test_worker_decodes_two_rgb_frames_without_resizing() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    decoded = _decode_rgb_history([encoded.tobytes()] * 2, "head_left")
    assert decoded.shape == (2, 480, 640, 3)
    assert decoded.dtype == np.uint8
    assert float(decoded[:, :, :, 0].mean()) > 250.0
    assert float(decoded[:, :, :, 2].mean()) < 5.0


class _Safety:
    arm_position_lower_rad = [-10.0] * 14
    arm_position_upper_rad = [10.0] * 14


class _Config:
    safety = _Safety()


def test_h40_chunk_validation_rejects_wrong_horizon() -> None:
    valid = np.zeros((40, 16), dtype=np.float64)
    valid[:, 14:] = 2.25
    validate_policy_chunk(
        valid,
        measured_arm=np.zeros(14),
        config=_Config(),
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
        expected_horizon=40,
    )
    with pytest.raises(ValueError, match=r"\[40,16\]"):
        validate_policy_chunk(
            valid[:39],
            measured_arm=np.zeros(14),
            config=_Config(),
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
            expected_horizon=40,
        )
