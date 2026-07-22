from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from model.flip_table_reinforcement_learning.teacher.source_stereo_calibration import fit_rigid_transform
from model.flip_table_reinforcement_learning.scripts.prepare_source_wrist_hand_eye_workspace import (
    build_annotation_template,
    _select_closest_stereo_pair,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_has_only_maintained_modes_and_valid_shell() -> None:
    runner = ROOT / "run_train_in_container.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    source = runner.read_text(encoding="utf-8")

    for mode in ("audit_contract", "audit_partial_reset", "smoke", "evaluate", "train", "evaluate_rlpd_stage", "train_rlpd"):
        assert mode in source
    for removed_mode in ("optimize_contact", "evaluate_eef_teacher", "evaluate_regrasp_teacher", "trajectory_sweep"):
        assert removed_mode not in source


def test_local_runner_forwards_project_scoped_configuration() -> None:
    source = (ROOT / "run_train_local.sh").read_text(encoding="utf-8")

    assert "FLIP_TABLE_*|ROBOFINALS_*|WANDB_*" in source
    assert "done < <(env)" in source


def test_source_calibration_transforms_round_trip() -> None:
    transforms = _load_module(ROOT / "teacher" / "transforms.py", "flip_table_transforms_test")
    pose = np.array((0.1, -0.2, 0.3, 0.2, -0.1, 0.4), dtype=np.float64)

    recovered = transforms.transform_to_pose(transforms.pose_to_transform(pose))

    assert np.allclose(recovered, pose, atol=1.0e-9)
    assert np.allclose(
        transforms.inverse_transform(transforms.pose_to_transform(pose)) @ transforms.pose_to_transform(pose),
        np.eye(4),
        atol=1.0e-9,
    )


def test_source_calibration_recovers_a_rigid_translation() -> None:
    source = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)), dtype=np.float64)
    transform, residuals = fit_rigid_transform(source, source + np.array((0.2, -0.1, 0.3)))

    assert np.allclose(transform[:3, 3], (0.2, -0.1, 0.3), atol=1.0e-9)
    assert np.allclose(residuals, 0.0, atol=1.0e-9)


def test_d405_pairing_never_combines_adjacent_frames() -> None:
    first = [(579_722_000, "first-old"), (617_978_000, "first-new")]
    second = [(583_419_000, "second-old"), (618_024_000, "second-new")]

    selected = _select_closest_stereo_pair(
        first,
        second,
        target_time_ns=600_000_000,
        max_stereo_skew_ns=30_000_000,
    )

    assert selected == (617_978_000, "first-new", 618_024_000, "second-new")


def test_wrist_workspace_provenance_binding_is_all_or_nothing(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        build_annotation_template(
            [],
            wrist_side="left",
            manifest_path=tmp_path / "manifest.json",
            notes=[],
            source_episode_index=0,
        )

    result = build_annotation_template(
        [],
        wrist_side="left",
        manifest_path=tmp_path / "manifest.json",
        notes=[],
        source_episode_index=0,
        calibration_endpoint="initial",
        source_annotation_workspace_manifest_sha256="a" * 64,
    )
    assert result["source_episode_index"] == 0
    assert result["calibration_endpoint"] == "initial"
    assert result["workspace_manifest_sha256"] == "a" * 64


def test_demo_preparer_keeps_19d_real_robot_contract() -> None:
    module = _load_module(ROOT / "scripts" / "prepare_demo_actions.py", "flip_table_demo_preparer_test")
    desired = module._finite_float_list([0.0] * 36, expected=36, label="desired")
    hands = module._finite_float_list([0.0, 4.5], expected=2, label="hands")

    assert len(desired[module.UPPER_BODY_SLICE] + hands) == 19
    assert module.DEFAULT_OUTPUT == ROOT.parents[1] / ".checkpoints" / "flip_table_episode0_actions.json"
