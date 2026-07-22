from __future__ import annotations

from dataclasses import replace
import json

import h5py
import numpy as np

from data.flip_table_data_augmentation.provenance import CandidateLedger, CandidateRecord
from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.replicator.appearance import AppearanceController
from data.flip_table_data_augmentation.replicator.camera_image import (
    apply_recorded_camera_geometry,
    ideal_render_intrinsic,
)
from data.flip_table_data_augmentation.replicator.trajectory import (
    _trajectory_group_sha256,
    inspect_accepted_trajectory,
    read_numeric_trace,
    read_state_at,
    sample_indices,
    write_numeric_parquet,
)
from data.flip_table_data_augmentation.source_camera_projection import (
    project_root_points,
    root_from_camera,
    visible_mask,
)
from data.flip_table_data_augmentation.source_stereo_depth import (
    left_right_consistency_mask,
)
from data.flip_table_data_augmentation.source_video import source_frame_indices
from data.flip_table_data_augmentation.source_contract import NUMERIC_FEATURES


def _accepted_fixture(tmp_path):
    hdf5_path = tmp_path / "accepted.hdf5"
    with h5py.File(hdf5_path, "w") as stream:
        data = stream.create_group("data")
        demo = data.create_group("demo_4")
        demo.attrs.update(num_samples=6, seed=46, success=True)
        states = demo.create_group("states")
        states.create_dataset("robot", data=np.arange(18, dtype=np.float32).reshape(6, 3))
        numeric = demo.create_group("dataset_numeric")
        for ordinal, (key, (_dtype, width)) in enumerate(NUMERIC_FEATURES.items()):
            numeric.create_dataset(
                key,
                data=np.full((6, width), ordinal, dtype=np.float32),
            )
        trajectory_hash = _trajectory_group_sha256(demo)
        demo.attrs.update(
            candidate_id="run-attempt-000004",
            trajectory_seed=46,
            config_sha256="a" * 64,
            runtime_digest="b" * 64,
            trajectory_sha256=trajectory_hash,
            source_episode_indices_json=json.dumps([1, 5]),
            generation_payload_json="{}",
        )
    ledger = CandidateLedger(tmp_path / "ledger")
    record = CandidateRecord(
        candidate_id="run-attempt-000004",
        status="claimed",
        source_episode_indices=(),
        trajectory_seed=46,
        config_sha256="a" * 64,
        runtime_digest="b" * 64,
        payload={"attempt_index": 4},
    )
    ledger.claim(record)
    ledger.transition(
        record.candidate_id,
        "generated",
        {"generator_success": True},
        source_episode_indices=(1, 5),
    )
    ledger.transition(
        record.candidate_id,
        "validated",
        {"accepted_hdf5_demo": "demo_4", "trajectory_sha256": trajectory_hash},
    )
    return hdf5_path, ledger, record.candidate_id


def test_rational_sampling_is_deterministic_and_unique() -> None:
    assert sample_indices(50).tolist() == [
        0, 2, 3, 5, 7, 8, 10, 12, 13, 15,
        17, 18, 20, 22, 23, 25, 27, 28, 30, 32,
        33, 35, 37, 38, 40, 42, 43, 45, 47, 48,
    ]


def test_nominal_camera_mount_preserves_isaaclab_xyzw_order() -> None:
    class Task:
        def __init__(self):
            self.poses = []

        def _find_prim_by_suffix(self, _env, suffix, env_id):
            assert env_id == 0
            return suffix

        def _set_stage_prim_local_pose(self, prim, position, quaternion):
            self.poses.append((prim, position.tolist(), quaternion.tolist()))

    controller = object.__new__(AppearanceController)
    controller.env = object()
    controller.config = load_pipeline_config()
    controller.task = Task()
    controller._set_nominal_camera_mounts()
    assert len(controller.task.poses) == 3
    for (_prim, position, quaternion), camera in zip(
        controller.task.poses, controller.config.cameras, strict=True
    ):
        assert np.allclose(position, camera.offset_position_m)
        assert np.allclose(quaternion, camera.offset_quaternion_xyzw)


def test_recorded_camera_geometry_identity_is_pixel_exact() -> None:
    camera = load_pipeline_config().cameras[0]
    ideal = ideal_render_intrinsic(camera)
    identity_camera = replace(
        camera,
        intrinsic_matrix_px=tuple(float(value) for value in ideal.reshape(-1)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    rows, columns = np.indices((camera.height, camera.width), dtype=np.uint16)
    image = np.stack(
        ((columns % 256).astype(np.uint8), (rows % 256).astype(np.uint8), np.zeros_like(rows, dtype=np.uint8)),
        axis=2,
    )

    calibrated = apply_recorded_camera_geometry(image, identity_camera)

    assert np.array_equal(calibrated, image)


def test_measured_camera_geometry_keeps_raw_shape_and_dtype() -> None:
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    for camera in load_pipeline_config().cameras:
        calibrated = apply_recorded_camera_geometry(image, camera)
        assert calibrated.shape == (480, 640, 3)
        assert calibrated.dtype == np.uint8


def test_opengl_camera_projection_uses_cv_forward_axis() -> None:
    camera = load_pipeline_config().cameras[0]
    ideal = ideal_render_intrinsic(camera)
    camera = replace(
        camera,
        offset_position_m=(0.0, 0.0, 0.0),
        offset_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        intrinsic_matrix_px=tuple(float(value) for value in ideal.reshape(-1)),
        distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    transform = root_from_camera(np.eye(4), camera)
    pixels, depth = project_root_points(
        np.asarray(((0.0, 0.0, -1.0), (0.1, -0.1, -1.0))),
        transform,
        camera,
    )

    assert np.allclose(pixels[0], (320.0, 240.0))
    assert pixels[1, 0] > 320.0 and pixels[1, 1] > 240.0
    assert np.allclose(depth, 1.0)
    assert visible_mask(pixels, depth, camera).tolist() == [True, True]


def test_left_right_consistency_uses_corresponding_right_pixel() -> None:
    left = np.full((2, 8), 2.0, dtype=np.float32)
    right = np.full((2, 8), 2.0, dtype=np.float32)
    right[0, 2] = 5.0

    consistent = left_right_consistency_mask(
        left, right, maximum_error_px=0.5
    )

    expected = np.zeros((2, 8), dtype=bool)
    expected[:, 2:] = True
    expected[0, 4] = False
    np.testing.assert_array_equal(consistent, expected)


def test_source_frame_stride_includes_the_exact_episode_endpoint() -> None:
    assert source_frame_indices(10, 3) == (0, 3, 6, 9)
    assert source_frame_indices(11, 3) == (0, 3, 6, 9, 10)


def test_accepted_trajectory_numeric_state_and_parquet(tmp_path) -> None:
    hdf5_path, ledger, candidate_id = _accepted_fixture(tmp_path)
    trajectory = inspect_accepted_trajectory(hdf5_path, candidate_id, ledger)
    indices = sample_indices(trajectory.source_frame_count)
    assert indices.tolist() == [0, 2, 3, 5]
    trace = read_numeric_trace(trajectory, indices)
    assert all(value.shape[0] == 4 for value in trace.values())
    assert read_state_at(trajectory, 2)["robot"].tolist() == [6.0, 7.0, 8.0]
    output = tmp_path / "numeric.parquet"
    digest = write_numeric_parquet(output, trace)
    assert output.is_file() and len(digest) == 64


def test_tampered_trajectory_is_rejected(tmp_path) -> None:
    hdf5_path, ledger, candidate_id = _accepted_fixture(tmp_path)
    with h5py.File(hdf5_path, "r+") as stream:
        stream["data/demo_4/states/robot"][0, 0] = -1
    try:
        inspect_accepted_trajectory(hdf5_path, candidate_id, ledger)
    except ValueError as exc:
        assert "SHA-256 changed" in str(exc)
    else:
        raise AssertionError("tampered accepted trajectory was not rejected")
