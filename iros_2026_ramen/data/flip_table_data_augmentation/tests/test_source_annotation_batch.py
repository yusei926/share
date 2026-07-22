from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.scripts.run_source_annotation_batch import (
    BATCH_SCHEMA_VERSION,
    GENERATED_SOURCE_DIRECTORY_NAMES,
    _batch_identity,
    _episodes,
    _execution_path,
    _failure_details,
    _failure_status,
    _resolve_runtime_mode,
    _resume_command,
    _v1_entrypoint,
)


def test_episode_selection_accepts_ranges_and_rejects_out_of_contract() -> None:
    assert _episodes("3,1-2,2,530") == (1, 2, 3, 530)
    for value in ("", "3-1", "-1", "531", "1,,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            _episodes(value)


def test_batch_identity_binds_source_code_mesh_and_config_not_stop_stage() -> None:
    raw = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    identity = _batch_identity(
        raw,
        (0, 23),
        augmentation_source_sha256="a" * 64,
        mesh_sha256="b" * 64,
    )
    assert identity["schema_version"] == BATCH_SCHEMA_VERSION
    assert identity["config_sha256"] == load_pipeline_config().digest
    assert identity["source_revision"] == raw["source"]["revision"]
    assert identity["episodes"] == [0, 23]
    assert "stop_after" not in identity
    assert "requested_stop_after" not in identity


def test_batch_source_identity_excludes_only_generated_directories() -> None:
    assert GENERATED_SOURCE_DIRECTORY_NAMES == {
        "outputs",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }


def test_resume_flag_is_only_forwarded_when_requested() -> None:
    command = ["runner", "track"]
    assert _resume_command(command, False) == command
    assert _resume_command(command, True) == [*command, "--resume"]
    assert command == ["runner", "track"]


def test_v1_entrypoint_respects_explicit_runtime_mode(tmp_path: Path) -> None:
    assert _v1_entrypoint(tmp_path, "direct").name == "run_v1_direct.sh"
    assert _v1_entrypoint(tmp_path, "docker").name == "run_v1_container.sh"
    with pytest.raises(ValueError):
        _v1_entrypoint(tmp_path, "invalid")


def test_execution_paths_keep_isolated_direct_output_root(tmp_path: Path) -> None:
    outputs = tmp_path / "pilot"
    artifact = outputs / "source" / "episode-000001"
    assert _execution_path(artifact, outputs, runtime_mode="direct") == str(
        artifact.resolve()
    )
    assert (
        _execution_path(artifact, outputs, runtime_mode="docker")
        == "/outputs/source/episode-000001"
    )
    with pytest.raises(ValueError):
        _execution_path(tmp_path / "outside", outputs, runtime_mode="direct")
    with pytest.raises(ValueError):
        _execution_path(artifact, outputs, runtime_mode="auto")


def test_explicit_runtime_mode_resolution_does_not_probe_docker() -> None:
    assert _resolve_runtime_mode("direct") == "direct"
    assert _resolve_runtime_mode("docker") == "docker"
    with pytest.raises(ValueError):
        _resolve_runtime_mode("invalid")


def test_failure_details_preserve_quality_manifest_evidence(tmp_path: Path) -> None:
    stage = tmp_path / "track"
    stage.mkdir()
    manifest = stage / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "accepted": False,
                "rejection_reasons": ["bidirectional_pose_gate"],
                "gate": {"pass": False},
            }
        ),
        encoding="utf-8",
    )
    details = _failure_details("track", stage, tmp_path)
    assert details["stage"] == "track"
    assert details["manifest"] == "track/manifest.json"
    assert details["rejection_reasons"] == ["bidirectional_pose_gate"]
    assert details["gate"] == {"pass": False}
    assert len(details["manifest_sha256"]) == 64


def test_only_explicit_manifest_quality_failure_is_rejected() -> None:
    evidence = {
        "accepted": False,
        "rejection_reasons": ["bidirectional_pose_gate"],
        "gate": {"pass": False},
    }
    assert _failure_status(2, evidence) == "rejected"
    assert _failure_status(1, evidence) == "failed"
    assert _failure_status(2, {"gate": {"pass": False}}) == "failed"
    assert _failure_status(2, {**evidence, "rejection_reasons": []}) == "failed"
