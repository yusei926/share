from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from flip_table_curation import source


def test_video_download_is_limited_to_selected_source_episodes(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    episodes = root / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    info = {
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    rows = []
    for episode_index in (0, 1):
        row = {"episode_index": episode_index}
        for key in source.VIDEO_KEYS:
            row[f"videos/{key}/chunk_index"] = 0
            row[f"videos/{key}/file_index"] = episode_index
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), episodes / "file-000.parquet")
    calls: list[list[str]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(list(kwargs["allow_patterns"]))
        return str(root)

    monkeypatch.setattr(source, "snapshot_download", fake_snapshot_download)

    class Config:
        source_repo_id = "source"
        source_revision = "revision"
        workspace = tmp_path / "workspace"

    snapshot = source.download_source(
        Config(), include_data=False, include_videos=True, video_episode_indices=[1]
    )

    assert len(snapshot.episodes) == 2
    assert calls[0] == ["README.md", "meta/**"]
    assert len(calls) == 2
    assert "videos/observation.images.cam_0/chunk-000/file-001.mp4" in calls[1]
    assert "videos/observation.images.cam_0/chunk-000/file-000.mp4" not in calls[1]


def test_selected_video_download_rejects_unknown_episode(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    episodes = root / "meta" / "episodes" / "chunk-000"
    episodes.mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist([{"episode_index": 0, **{f"videos/{key}/chunk_index": 0 for key in source.VIDEO_KEYS}, **{f"videos/{key}/file_index": 0 for key in source.VIDEO_KEYS}}]), episodes / "file-000.parquet")
    monkeypatch.setattr(source, "snapshot_download", lambda **_kwargs: str(root))

    class Config:
        source_repo_id = "source"
        source_revision = "revision"
        workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="unknown source episodes"):
        source.download_source(
            Config(), include_data=False, include_videos=True, video_episode_indices=[1]
        )
