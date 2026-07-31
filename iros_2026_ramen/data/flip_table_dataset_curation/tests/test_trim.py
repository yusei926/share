from __future__ import annotations

import numpy as np

from flip_table_curation.trim import detect_trim, resample_trajectory


def test_trim_keeps_configured_pre_and_post_roll() -> None:
    arms = np.zeros((180, 14))
    hands = np.zeros((180, 2))
    arms[60:101, 0] = np.linspace(0.0, 1.0, 41)
    arms[101:, 0] = 1.0
    result = detect_trim(
        arms,
        hands,
        fps=30,
        arm_velocity_threshold=0.08,
        hand_velocity_threshold=0.25,
        persistence_window=9,
        persistence_required=5,
        pre_roll=30,
        post_roll=60,
        minimum_frames=90,
        minimum_terminal_stable_frames=15,
    )
    assert result.valid
    assert result.start == result.first_active - 30
    assert result.end == result.last_active + 1 + 60
    assert result.length >= 130


def test_trim_keeps_source_end_when_official_slice_has_no_post_roll() -> None:
    arms = np.zeros((120, 14))
    hands = np.zeros((120, 2))
    arms[80:, 0] = np.linspace(0.0, 1.0, 40)
    result = detect_trim(
        arms,
        hands,
        fps=30,
        arm_velocity_threshold=0.08,
        hand_velocity_threshold=0.25,
        persistence_window=9,
        persistence_required=5,
        pre_roll=30,
        post_roll=60,
        minimum_frames=90,
        minimum_terminal_stable_frames=15,
    )
    assert result.valid
    assert result.end == len(arms)
    assert not result.post_roll_complete


def test_resampling_is_speed_invariant_for_same_normalized_path() -> None:
    short = np.linspace(0, 1, 30)[:, None]
    long = np.linspace(0, 1, 90)[:, None]
    assert np.allclose(resample_trajectory(short, 64), resample_trajectory(long, 64))
