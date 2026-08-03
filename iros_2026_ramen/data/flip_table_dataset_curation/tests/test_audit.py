from __future__ import annotations

from pathlib import Path

import pytest

from flip_table_curation import audit


def test_audit_writes_failure_report_for_label_conflict(monkeypatch, tmp_path: Path) -> None:
    class Config:
        digest = "digest"
        source_repo_id = "source"
        source_revision = "revision"
        workspace = tmp_path

        def section(self, name):
            assert name == "source"
            return {"fps": 30, "expected_episodes": 1, "expected_frames": 2}

    class Snapshot:
        info = {"fps": 30, "total_frames": 2, "features": {key: {"dtype": "video", "shape": [480, 640, 3]} for key in audit.VIDEO_KEYS}}
        episodes = ({"episode_index": 0, "length": 2},)

    monkeypatch.setattr(audit, "download_source", lambda *args, **kwargs: Snapshot())
    monkeypatch.setattr(audit, "write_selection", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("overlap")))
    with pytest.raises(ValueError, match="overlap"):
        audit.audit_labels(Config())
    assert '"passed": false' in (tmp_path / "audit" / "label_audit.json").read_text()
