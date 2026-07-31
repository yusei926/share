from __future__ import annotations

import importlib

import numpy as np


MODULE = "evaluate.flip_table_simulation.real_to_sim_calibration.fk_first_camera_calibration"


def test_static_frame_selection_is_distinct_and_prefers_arm_diversity() -> None:
    module = importlib.import_module(MODULE)
    q = np.zeros((12, 36), dtype=np.float64)
    q[:, 22] = (0.0, 0.0, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.6, 1.6, 2.0, 2.0)
    selected = module.static_frame_indices(q, np.arange(12, dtype=np.float64) / 30.0, count=4)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert selected == tuple(sorted(selected))


def test_head_acceptance_allows_strong_automatic_holdout_evidence() -> None:
    module = importlib.import_module(MODULE)
    accepted, reason = module.can_accept_head_calibration(
        median_px=2.0, p95_px=7.0, holdout_points=16, manual_points=0
    )
    assert accepted
    assert "passed" in reason


def test_head_acceptance_rejects_bad_holdout_reprojection() -> None:
    module = importlib.import_module(MODULE)
    accepted, reason = module.can_accept_head_calibration(
        median_px=3.1, p95_px=8.1, holdout_points=16, manual_points=4
    )
    assert not accepted
    assert "reprojection" in reason


def test_scene_pose_delta_accepts_table_yaw_symmetry() -> None:
    module = importlib.import_module(MODULE)
    fitted = np.eye(4, dtype=np.float64)
    heldout = np.eye(4, dtype=np.float64)
    heldout[:3, :3] = np.diag((-1.0, -1.0, 1.0))
    translation, rotation = module._scene_pose_delta(fitted, heldout)
    assert translation == 0.0
    assert rotation == 0.0


def test_workbench_observability_does_not_invent_an_unseen_bench_pose() -> None:
    module = importlib.import_module(MODULE)
    report = module._workbench_observability({"fitted": {"fixed_scene_root_from_table": np.eye(4).tolist()}})
    assert report["status"] == "not_fully_identifiable_from_table_only"
    assert "root_from_workbench yaw" in report["not_identified"]
