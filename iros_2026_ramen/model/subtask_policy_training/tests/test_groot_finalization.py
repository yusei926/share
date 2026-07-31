from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from model.subtask_policy_training.scripts.finalize_groot_n17_checkpoint import (
    build_directory_manifest,
    build_source_snapshot_manifest,
    discover_wandb_url,
    read_training_task_contract,
    validate_evaluation_checkpoint,
    validate_progress_sidecar_bundle,
)
from model.subtask_policy_training.scripts.verify_policy_hub_roundtrip import (
    validate_offline_evaluation_equivalence,
)
from model.subtask_policy_training.gr00t.n17_contract import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    validate_eef_fk_release_audit,
)


def test_wandb_url_is_derived_from_selected_training_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "wandb" / "run-20260730_051018-3zjgg4wk").mkdir(parents=True)
    monkeypatch.setenv("WANDB_PROJECT", "iros2026-ramen-flip-table")
    monkeypatch.setenv("WANDB_ENTITY", "yusei926")
    assert discover_wandb_url(tmp_path) == (
        "https://wandb.ai/yusei926/iros2026-ramen-flip-table/runs/3zjgg4wk"
    )


def test_source_snapshot_contains_release_entrypoints(tmp_path: Path) -> None:
    bundle = tmp_path / "source_snapshot"
    manifest = build_source_snapshot_manifest(bundle_root=bundle)
    assert manifest["schema_version"] == "groot_n17_source_snapshot_v2"
    assert (
        manifest["scope"]
        == "release_time_training_inference_and_evaluation_source"
    )
    assert manifest["captured_at_utc"]
    assert manifest["bundle_root"] == "source_snapshot"
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["file_count"] > 100
    assert (
        "model/subtask_policy_training/scripts/run_h100_flip_table_groot_n17.sh"
        in manifest["files"]
    )
    assert (
        "evaluate/flip_table_simulation/container_overlay/policy/"
        "flip_table_eval_policy.py"
    ) in manifest["files"]
    for relative, expected_hash in manifest["files"].items():
        path = bundle / relative
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash


def test_sim_evaluation_manifest_hashes_every_file(tmp_path: Path) -> None:
    (tmp_path / "episode_0.mp4").write_bytes(b"video")
    trace = tmp_path / "test_0" / "action_state_trace.jsonl"
    trace.parent.mkdir()
    trace.write_text("{}\n")
    manifest = build_directory_manifest(
        tmp_path,
        schema_version="groot_n17_sim_evaluation_bundle_v1",
    )
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == 8
    assert set(manifest["files"]) == {
        "episode_0.mp4",
        "test_0/action_state_trace.jsonl",
    }


def test_training_task_contract_is_exact(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "tasks.parquet"
    pq.write_table(
        pa.table(
            {
                "task_index": [0],
                "__index_level_0__": ["flip table"],
            }
        ),
        path,
    )
    assert read_training_task_contract(path) == [
        {"task_index": 0, "task": "flip table"}
    ]

    pq.write_table(
        pa.table(
            {
                "task_index": [0],
                "__index_level_0__": ["flip_table"],
            }
        ),
        path,
    )
    with pytest.raises(ValueError, match="task_index 0"):
        read_training_task_contract(path)


def _write_sidecar_bundle(root: Path) -> tuple[Path, Path]:
    milestones = {
        name: {
            "frame": 0 if name == "M0" else 1 if name == "M6" else None,
            "confidence": 0.8 if name in {"M0", "M6"} else 0.0,
            "source": "fixture",
            "valid": name in {"M0", "M6"},
        }
        for name in ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
    }
    progress_path = root / "progress.jsonl"
    progress_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "flip_table_event_progress_v1",
                    "episode_index": episode,
                    "length": 2,
                    "primary_hand": None,
                    "milestones": milestones,
                    "progress": [0.0, 1.0],
                    "progress_mask": [True, True],
                    "review_required": True,
                    "diagnostics": {},
                }
            )
            + "\n"
            for episode in range(174)
        )
    )
    visual_path = root / "visual_rotation.jsonl"
    visual_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": "flip_table_visual_rotation_v1",
                    "episode_index": episode,
                    "length": 2,
                    "rotation_rad": [0.0, 0.1],
                    "confidence": [0.8, 0.8],
                    "detection_fraction": 1.0,
                }
            )
            + "\n"
            for episode in range(174)
        )
    )
    contact_sheet = root / "orientation_contact_sheet.jpg"
    contact_sheet.write_bytes(b"reviewed-contact-sheet")
    contact_hash = hashlib.sha256(contact_sheet.read_bytes()).hexdigest()
    (root / "orientation_contact_sheet.approved").write_text(contact_hash + "\n")

    visual_manifest = root / "visual_rotation_manifest.json"
    visual_manifest.write_text(
        json.dumps(
            {
                "schema_version": "flip_table_visual_rotation_manifest_v1",
                "dataset_repo_id": (
                    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
                ),
                "dataset_revision": (
                    "0dc47877dfb2efbea796a059c81290c649bc773c"
                ),
                "video_key": "observation.images.cam_0",
                "episode_count": 174,
                "sidecar_sha256": hashlib.sha256(
                    visual_path.read_bytes()
                ).hexdigest(),
                "contact_sheet": contact_sheet.name,
                "contact_sheet_sha256": contact_hash,
                "contact_sheet_human_review_required": True,
                "policy_input": False,
            }
        )
    )
    progress_manifest = root / "progress_manifest.json"
    progress_manifest.write_text(
        json.dumps(
            {
                "schema_version": "flip_table_progress_sidecar_manifest_v1",
                "dataset_repo_id": (
                    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
                ),
                "dataset_revision": (
                    "0dc47877dfb2efbea796a059c81290c649bc773c"
                ),
                "annotation_file": progress_path.name,
                "annotation_sha256": hashlib.sha256(
                    progress_path.read_bytes()
                ).hexdigest(),
                "visual_rotation_sidecar": {
                    "path": str(visual_path),
                    "sha256": hashlib.sha256(
                        visual_path.read_bytes()
                    ).hexdigest(),
                },
                "milestones": [
                    "M0",
                    "M1",
                    "M2",
                    "M3",
                    "M4",
                    "M5",
                    "M6",
                ],
                "policy_input_exclusions": [
                    "milestone labels",
                    "progress labels",
                    "future images",
                    "sim ground truth",
                    "object pose",
                    "contact ground truth",
                ],
                "summary": {
                    "episode_count": 174,
                    "review_required_count": 174,
                    "valid_by_milestone": {
                        name: 174 if name in {"M0", "M6"} else 0
                        for name in milestones
                    },
                    "orientation_groups": {
                        "0": 44,
                        "1": 44,
                        "2": 43,
                        "3": 43,
                    },
                    "fixed_phase_segmentation": False,
                },
            }
        )
    )
    return progress_manifest, visual_manifest


def test_progress_sidecar_bundle_requires_all_annotations_and_review(
    tmp_path: Path,
) -> None:
    progress_manifest, visual_manifest = _write_sidecar_bundle(tmp_path)
    artifacts = validate_progress_sidecar_bundle(
        progress_manifest,
        visual_manifest,
    )
    assert set(artifacts) == {
        "progress.jsonl",
        "visual_rotation.jsonl",
        "orientation_contact_sheet.jpg",
        "orientation_contact_sheet.approved",
    }

    (tmp_path / "orientation_contact_sheet.approved").write_text("wrong\n")
    with pytest.raises(ValueError, match="human approval"):
        validate_progress_sidecar_bundle(progress_manifest, visual_manifest)


def test_wandb_url_rejects_ambiguous_training_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "wandb" / "run-20260730_000000-one").mkdir(parents=True)
    (tmp_path / "wandb" / "run-20260730_000001-two").mkdir(parents=True)
    monkeypatch.setenv("WANDB_PROJECT", "iros2026-ramen-flip-table")
    monkeypatch.setenv("WANDB_ENTITY", "yusei926")
    with pytest.raises(ValueError, match="expected one W&B run"):
        discover_wandb_url(tmp_path)


def _evaluation_report(metric: float, *, latency_ms: float = 10.0) -> dict:
    metrics = {
        "physical_arm_rmse_rad": metric,
        "chunk_inference_ms_mean": latency_ms,
        "chunk_inference_ms_p95": latency_ms + 1.0,
    }
    return {
        "schema_version": "groot_n17_offline_chunk_reset_v2",
        "evaluation_type": "offline_chunk_reset_not_closed_loop",
        "model_safetensors_sha256": "checkpoint-sha256",
        "episodes": [156, 157],
        "declared_split": "test",
        "execution_steps": 10,
        "randomness": {
            "base_seed": 42,
            "episode_stride": 1_000_003,
            "uint32_modulus": 2**32,
        },
        "contract": {"state_dim": 49, "logical_action_dim": 53},
        "aggregate": {"physical_arm_rmse_rad": metric},
        "orientation_group_report": {
            "0": {"episodes": [156], "aggregate": {"physical_arm_rmse_rad": metric}}
        },
        "episodes_report": {"156": metrics, "157": metrics},
    }


def test_hf_roundtrip_requires_policy_metrics_to_reproduce() -> None:
    expected = _evaluation_report(0.031, latency_ms=10.0)
    actual = _evaluation_report(0.03100001, latency_ms=99.0)
    validate_offline_evaluation_equivalence(expected, actual)

    changed = _evaluation_report(0.04)
    with pytest.raises(ValueError, match="changed metric"):
        validate_offline_evaluation_equivalence(expected, changed)

    changed_seed = _evaluation_report(0.031)
    changed_seed["randomness"]["base_seed"] = 43
    with pytest.raises(ValueError, match="changed randomness"):
        validate_offline_evaluation_equivalence(expected, changed_seed)

    changed_checkpoint = _evaluation_report(0.031)
    changed_checkpoint["model_safetensors_sha256"] = "different-checkpoint"
    with pytest.raises(ValueError, match="model_safetensors_sha256"):
        validate_offline_evaluation_equivalence(expected, changed_checkpoint)


def test_final_evaluation_must_identify_selected_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"selected")
    evaluation = {
        "model_safetensors_sha256": hashlib.sha256(b"selected").hexdigest()
    }
    validate_evaluation_checkpoint(evaluation, checkpoint_path=checkpoint)

    evaluation["model_safetensors_sha256"] = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError, match="selected model"):
        validate_evaluation_checkpoint(evaluation, checkpoint_path=checkpoint)


def _eef_fk_release_audit() -> dict:
    return {
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
        "swapped_to_configured_score_ratio": 6.25,
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
                    "position_error_m": {"p95": 0.075},
                    "rotation_error_rad": {"p95": 0.18},
                }
                for side in ("left", "right")
            }
        },
        "per_episode": [
            {
                "episode_index": episode,
                "action_fk_residual_pass": episode != 21,
            }
            for episode in range(174)
        ],
        "coverage": {
            "episode_count": 174,
            "episode_level_diagnostic_threshold_exceedances": [21],
        },
        "mimic_source_episode_gate": {
            "eligible_count": 173,
            "rejected_count": 1,
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


def test_eef_fk_release_gate_preserves_fixed_thresholds_and_diagnostics() -> None:
    audit = _eef_fk_release_audit()
    result = validate_eef_fk_release_audit(audit)
    assert result["episode_threshold_exceedances"] == [21]
    assert result["selected_offset_frames"] == 0

    audit["thresholds"]["position_p95_m_max"] = 0.1
    with pytest.raises(ValueError, match="thresholds changed"):
        validate_eef_fk_release_audit(audit)


def test_eef_fk_release_gate_rejects_incomplete_episode_accounting() -> None:
    audit = _eef_fk_release_audit()
    audit["coverage"]["episode_level_diagnostic_threshold_exceedances"] = []
    with pytest.raises(ValueError, match="diagnostics are incomplete"):
        validate_eef_fk_release_audit(audit)
