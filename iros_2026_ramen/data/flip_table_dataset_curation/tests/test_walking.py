from __future__ import annotations

import numpy as np

from flip_table_curation.walking import detect_steps


def _detect(left: np.ndarray, right: np.ndarray):
    return detect_steps(
        {"left": left, "right": right},
        fps=30,
        median_window=5,
        floor_tolerance_m=0.015,
        maximum_contact_speed_m_s=0.08,
        minimum_contact_seconds=0.2,
        step_displacement_m=0.04,
    )


def test_stationary_balance_is_not_walking() -> None:
    time = np.arange(180) / 30
    sway = 0.004 * np.sin(time * 2.0)[:, None]
    left = np.c_[sway[:, 0], np.full(180, 0.1), np.zeros(180)]
    right = np.c_[sway[:, 0], np.full(180, -0.1), np.zeros(180)]
    result = _detect(left, right)
    assert not result.walked
    assert result.step_count == 0


def test_single_foot_relocation_is_rejected() -> None:
    left = np.zeros((180, 3))
    left[:, 1] = 0.1
    right = np.zeros((180, 3))
    right[:, 1] = -0.1
    left[60:90, 2] = 0.08
    left[60:90, 0] = np.linspace(0.0, 0.08, 30)
    left[90:, 0] = 0.08
    result = _detect(left, right)
    assert result.walked
    assert result.left_step_count == 1
