from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from inference.desktop.upper_policy.coarse_insert_groot_contract import (
    CAMERA_KEYS,
    MODEL_ACTION_HORIZON,
    compose_model_state,
    extract_executable_action,
)
from inference.desktop.upper_policy.run_coarse_insert_groot import (
    AsyncActionChunkPipeline,
    COARSE_INSERT_RAW_STEP_LIMIT_RAD,
    DEFAULT_URDF,
    DEFAULT_URDF_SHA256,
    _sha256_file,
)
from inference.desktop.upper_policy.subtask_start_pose import (
    COARSE_INSERT_FRAME0,
)


def test_coarse_insert_state_is_exact_49d() -> None:
    body = np.arange(29, dtype=np.float64) / 10.0
    state = compose_model_state(body, [0.25, 0.75], np.zeros(12))
    assert state.shape == (49,)
    np.testing.assert_allclose(state[32:39], body[15:22])
    np.testing.assert_allclose(state[39:46], body[22:29])
    np.testing.assert_allclose(state[46:49], body[12:15])
    assert state[18] == -0.375
    assert state[25] == 1.125
    np.testing.assert_allclose(state[19:25], 0.0)
    np.testing.assert_allclose(state[26:32], 0.0)


def test_coarse_insert_action_discards_waist_base_navigation_and_eef() -> None:
    native = np.zeros((MODEL_ACTION_HORIZON, 53))
    native[:, :18] = 777.0
    native[:, 32:46] = np.arange(14)
    native[:, 46:53] = 999.0
    native[:, 18] = -0.5
    native[:, 25] = 1.0
    result = extract_executable_action(native)
    assert result.shape == (16, 16)
    np.testing.assert_allclose(result[:, :14], np.tile(np.arange(14), (16, 1)))
    np.testing.assert_allclose(result[:, 14:], np.tile([1.5, 3.0], (16, 1)))
    assert not np.any(result == 777.0)
    assert not np.any(result == 999.0)


def test_coarse_insert_camera_contract_has_no_head_right() -> None:
    assert CAMERA_KEYS == (
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )


def test_coarse_insert_uses_training_author_pinned_urdf() -> None:
    assert _sha256_file(DEFAULT_URDF) == DEFAULT_URDF_SHA256


def test_coarse_insert_start_pose_uses_training_frame0_grasp_width() -> None:
    """The second Enter must reproduce the checkpoint's training grasp."""

    assert COARSE_INSERT_FRAME0.training_episode_count == 1697
    assert COARSE_INSERT_FRAME0.exact_training_revision is True
    assert COARSE_INSERT_FRAME0.dex1_opening_fraction == pytest.approx(
        (
            1.4682344198226929 / 4.5,
            2.2439992427825928 / 4.5,
        )
    )
    assert COARSE_INSERT_RAW_STEP_LIMIT_RAD == pytest.approx(0.30)


def test_async_pipeline_promotes_prefetched_chunk_without_hold() -> None:
    initial = np.tile(np.arange(16, dtype=np.float64), (16, 1))
    replacement = initial + 100.0
    pipeline = AsyncActionChunkPipeline(
        initial,
        execution_steps=8,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    try:
        first, index = pipeline.next_action()
        assert index == 0
        np.testing.assert_allclose(first, initial[0])
        pipeline.submit(
            lambda: (replacement, 12.0, {"validated_execution_steps": 8}),
            anchor_generation=(1, 2, 3),
        )
        while pipeline.prediction_pending:
            # Promotion is deliberately legal only at the chunk boundary.
            if pipeline.action_index < 8:
                pipeline.next_action()
            completed = pipeline.promote_if_ready()
            if completed is not None:
                break
            time.sleep(0.001)
        assert completed is not None
        assert completed.anchor_generation == (1, 2, 3)
        next_action, index = pipeline.next_action()
        assert index == 0
        np.testing.assert_allclose(next_action, replacement[0])
        assert pipeline.deadline_miss_ticks == 0
    finally:
        pipeline.close()


def test_async_pipeline_holds_last_target_when_prediction_is_late() -> None:
    initial = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
    replacement = initial + 1000.0
    release = threading.Event()
    pipeline = AsyncActionChunkPipeline(
        initial,
        execution_steps=2,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    try:
        pipeline.next_action()

        def slow_prediction() -> tuple[np.ndarray, float, dict[str, float]]:
            assert release.wait(timeout=2.0)
            return replacement, 100.0, {}

        pipeline.submit(slow_prediction, anchor_generation=(4, 5, 6))
        last, index = pipeline.next_action()
        assert index == 1
        assert pipeline.promote_if_ready() is None
        held, index = pipeline.next_action()
        assert index is None
        np.testing.assert_allclose(held, last)
        assert pipeline.deadline_miss_ticks == 1
        release.set()
        deadline = time.monotonic() + 2.0
        completed = None
        while completed is None and time.monotonic() < deadline:
            completed = pipeline.promote_if_ready()
            time.sleep(0.001)
        assert completed is not None
        resumed, index = pipeline.next_action()
        assert index == 0
        np.testing.assert_allclose(resumed, replacement[0])
    finally:
        release.set()
        pipeline.close()


def test_async_pipeline_does_not_hide_final_prediction_failure() -> None:
    pipeline = AsyncActionChunkPipeline(
        np.zeros((16, 16)),
        execution_steps=8,
        replan_after_steps=1,
        max_prediction_age_s=2.0,
    )
    pipeline.next_action()

    def fail() -> tuple[np.ndarray, float, dict[str, float]]:
        raise RuntimeError("synthetic inference failure")

    pipeline.submit(fail, anchor_generation=(7, 8, 9))
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        pipeline.close()
