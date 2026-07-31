from __future__ import annotations

import numpy as np
import pyarrow as pa

from flip_table_curation.analysis import _reject_overlapping_source_video_frames
from flip_table_curation.build import _replace_columns
from flip_table_curation.source import VIDEO_KEYS


def test_reindexing_preserves_all_policy_values_bitwise() -> None:
    values = np.arange(3 * 36, dtype=np.float32).reshape(3, 36)
    table = pa.table(
        {
            "observation.state.robot_q_current": values.tolist(),
            "action.robot_q_desired": (values + 1).tolist(),
            "timestamp": np.arange(3, dtype=np.float32),
            "frame_index": np.arange(3, dtype=np.int64),
            "episode_index": np.zeros(3, dtype=np.int64),
            "index": np.arange(3, dtype=np.int64),
            "task_index": np.zeros(3, dtype=np.int64),
        }
    )
    result = _replace_columns(table, episode_index=4, global_start=10, fps=30)
    assert result["observation.state.robot_q_current"].to_pylist() == table[
        "observation.state.robot_q_current"
    ].to_pylist()
    assert result["action.robot_q_desired"].to_pylist() == table[
        "action.robot_q_desired"
    ].to_pylist()
    assert result["episode_index"].to_pylist() == [4, 4, 4]
    assert result["index"].to_pylist() == [10, 11, 12]


def test_source_video_frame_overlap_rejects_second_episode() -> None:
    rows = []
    for episode, start in ((0, 0.0), (1, 3.0 / 30.0)):
        row = {"episode_index": episode}
        for key in VIDEO_KEYS:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = 0
            row[f"videos/{key}/from_timestamp"] = start
        rows.append(row)
    records = [
        {
            "source_episode_index": episode,
            "trajectory_stability": 1.0,
            "trim_start": 0,
            "trim_length": 4,
            "curation_status": "accepted_auto",
            "rejection_reasons": [],
        }
        for episode in (0, 1)
    ]
    _reject_overlapping_source_video_frames(
        records,
        source_rows=tuple(rows),
        fps=30,
        source_video_frame_counts={
            f"videos/{key}/chunk-000/file-000.mp4": 100
            for key in VIDEO_KEYS
        },
    )
    assert records[0]["curation_status"] == "accepted_auto"
    assert records[1]["curation_status"] == "rejected"
    assert records[1]["rejection_reasons"] == ["source_video_frame_overlap"]
