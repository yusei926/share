from __future__ import annotations

import numpy as np
import pyarrow as pa

from flip_table_curation.build import _ffmpeg_filter, _plan_entry_groups, _replace_columns


def test_reindexing_preserves_policy_values_bitwise() -> None:
    values = np.arange(3 * 36, dtype=np.float32).reshape(3, 36)
    table = pa.table({
        "observation.state.robot_q_current": values.tolist(),
        "action.robot_q_desired": (values + 1).tolist(),
        "timestamp": np.arange(3, dtype=np.float32),
        "frame_index": np.arange(3, dtype=np.int64),
        "episode_index": np.zeros(3, dtype=np.int64),
        "index": np.arange(3, dtype=np.int64),
        "task_index": np.zeros(3, dtype=np.int64),
    })
    result = _replace_columns(table, episode_index=4, global_start=10, fps=30)
    assert result["observation.state.robot_q_current"].to_pylist() == table["observation.state.robot_q_current"].to_pylist()
    assert result["action.robot_q_desired"].to_pylist() == table["action.robot_q_desired"].to_pylist()
    np.testing.assert_allclose(result["timestamp"].to_numpy(), [0.0, 1 / 30, 2 / 30], rtol=0.0, atol=1e-7)
    assert result["episode_index"].to_pylist() == [4, 4, 4]
    assert result["index"].to_pylist() == [10, 11, 12]


def test_video_groups_respect_planning_frame_budget() -> None:
    groups = _plan_entry_groups([(0, 4, 0), (4, 9, 1), (9, 12, 2)], frame_budget=8)
    assert groups == [[(0, 4, 0)], [(4, 9, 1), (9, 12, 2)]]


def test_video_filter_preserves_each_selected_segment_length() -> None:
    expression = _ffmpeg_filter([(10, 14, 0), (30, 35, 1)], fps=30)
    assert "trim=start_frame=10:end_frame=14" in expression
    assert "trim=end_frame=4" in expression
    assert "trim=end_frame=5" in expression
    assert "concat=n=2:v=1:a=0" in expression
    assert "trim=end_frame=9" in expression
    assert "settb=1/90000" in expression
