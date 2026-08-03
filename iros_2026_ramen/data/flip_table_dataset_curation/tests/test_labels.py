from __future__ import annotations

import pandas as pd
import pytest

from flip_table_curation import labels as labels_module
from flip_table_curation.labels import LABEL_COLUMNS, _final_accepted_flip_labels, _validate_labels


def row(**overrides):
    value = {
        "episode_id": 0,
        "frame_start": 5,
        "frame_end": 10,
        "task_index": 2,
        "verdict": "success",
        "failure_category": None,
        "reviewer": "hara",
        "reviewed_at": "2026-08-01T00:00:00+09:00",
        "schema_version": 4,
    }
    value.update(overrides)
    return value


def test_label_validation_accepts_schema_v4() -> None:
    _validate_labels(pd.DataFrame([row()], columns=LABEL_COLUMNS), source_lengths={0: 20}, expected_schema_version=4)


@pytest.mark.parametrize("overrides", [{"frame_end": 20}, {"frame_start": 11, "frame_end": 10}, {"verdict": "unknown"}, {"episode_id": 3}])
def test_label_validation_rejects_invalid_ranges_and_values(overrides) -> None:
    with pytest.raises(ValueError):
        _validate_labels(pd.DataFrame([row(**overrides)], columns=LABEL_COLUMNS), source_lengths={0: 20}, expected_schema_version=4)


def test_label_validation_rejects_duplicate_primary_key() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _validate_labels(pd.DataFrame([row(), row(frame_end=12)], columns=LABEL_COLUMNS), source_lengths={0: 20}, expected_schema_version=4)


def test_selection_uses_only_the_final_successful_flip_per_source_episode() -> None:
    labels = pd.DataFrame([
        row(episode_id=0, frame_start=0, frame_end=3, verdict="success"),
        row(episode_id=0, frame_start=4, frame_end=8, verdict="failure", failure_category="side_flip"),
        row(episode_id=1, frame_start=0, frame_end=3, verdict="failure", failure_category="side_flip"),
        row(episode_id=1, frame_start=4, frame_end=8, verdict="optimal"),
    ], columns=LABEL_COLUMNS)
    flip, accepted = _final_accepted_flip_labels(labels, task_index=2)
    assert len(flip) == 4
    assert accepted[["episode_id", "frame_start", "verdict"]].to_dict("records") == [
        {"episode_id": 1, "frame_start": 4, "verdict": "optimal"}
    ]


def test_selection_reports_rejected_final_verdicts(monkeypatch, tmp_path) -> None:
    class Config:
        digest = "digest"
        source_repo_id = "source"
        source_revision = "revision"
        labels_repo_id = "labels"
        labels_revision = "revision"
        workspace = tmp_path
        raw = {"seed": 42}

        def section(self, name):
            return {"labels": {"schema_version": 4, "task_index": 2}}[name]

    source = pd.DataFrame(
        [
            row(episode_id=0, frame_start=0, frame_end=3, verdict="failure", failure_category="side_flip"),
            row(episode_id=1, frame_start=0, frame_end=3, verdict="success"),
        ],
        columns=LABEL_COLUMNS,
    )
    monkeypatch.setattr(labels_module, "_load_labels", lambda _config: (source, tmp_path / "labels.parquet"))
    monkeypatch.setattr(labels_module, "sha256_file", lambda _path: "hash")

    _selected, report = labels_module.select_segments(Config(), source_lengths={0: 10, 1: 10})

    assert report["rejected_final_flip_verdicts"] == {"failure": 1}


def test_selection_excludes_an_exact_user_requested_segment(monkeypatch, tmp_path) -> None:
    class Config:
        digest = "digest"
        source_repo_id = "source"
        source_revision = "revision"
        labels_repo_id = "labels"
        labels_revision = "revision"
        workspace = tmp_path
        raw = {
            "seed": 42,
            "manual_exclusions": [
                {
                    "source_episode_index": 0,
                    "source_frame_start": 0,
                    "source_frame_end": 3,
                    "reason": "user_requested_low_quality_segment",
                }
            ],
        }

        def section(self, name):
            return {"labels": {"schema_version": 4, "task_index": 2}}[name]

    source = pd.DataFrame(
        [row(episode_id=0, frame_start=0, frame_end=3), row(episode_id=1, frame_start=0, frame_end=3)],
        columns=LABEL_COLUMNS,
    )
    monkeypatch.setattr(labels_module, "_load_labels", lambda _config: (source, tmp_path / "labels.parquet"))
    monkeypatch.setattr(labels_module, "sha256_file", lambda _path: "hash")

    selected, report = labels_module.select_segments(Config(), source_lengths={0: 10, 1: 10})

    assert [row.source_episode_index for row in selected] == [1]
    assert report["user_excluded_segments"][0]["reason"] == "user_requested_low_quality_segment"
