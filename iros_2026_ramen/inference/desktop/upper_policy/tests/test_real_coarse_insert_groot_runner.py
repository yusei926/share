from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.upper_policy.coarse_insert_groot_contract import (
    CAMERA_KEYS,
    MODEL_ACTION_HORIZON,
    compose_model_state,
    extract_executable_action,
)
from inference.desktop.upper_policy.run_coarse_insert_groot import (
    COARSE_INSERT_RAW_STEP_LIMIT_RAD,
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


def test_coarse_insert_start_pose_uses_training_frame0_grasp_width() -> None:
    """The second Enter must reproduce the checkpoint's training grasp."""

    assert COARSE_INSERT_FRAME0.training_episode_count == 1907
    assert COARSE_INSERT_FRAME0.dex1_opening_fraction == pytest.approx(
        (
            1.3136359453201294 / 4.5,
            1.8953489065170288 / 4.5,
        )
    )
    assert COARSE_INSERT_RAW_STEP_LIMIT_RAD == pytest.approx(0.30)
