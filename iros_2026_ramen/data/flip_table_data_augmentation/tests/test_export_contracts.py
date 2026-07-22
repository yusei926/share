from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data.flip_table_data_augmentation.config import EXPECTED_CAMERA_KEYS, load_pipeline_config
from data.flip_table_data_augmentation.export.build_dataset import (
    _canonical_source_tasks,
    _canonicalize_video_depth_metadata,
    _lineage_split_groups,
    _local_real_split_indices,
)
from data.flip_table_data_augmentation.export.contracts import (
    RENDER_MANIFEST_SCHEMA_VERSION,
    TASK,
    RenderedEpisode,
    lineage_split,
)
from data.flip_table_data_augmentation.export.file_manifest import (
    verify_file_manifest,
    write_file_manifest,
)
from data.flip_table_data_augmentation.fk_audit import SYNTHETIC_ACTION_FK_SCHEMA_VERSION
from data.flip_table_data_augmentation.export.hf_transaction import (
    _prepare_upload_workspace,
)
from data.flip_table_data_augmentation.export.recompute_stats import image_sample_indices
from data.flip_table_data_augmentation.export.validate_dataset import (
    _episode_splits,
    _expected_image_stat_count,
    _numeric_array,
    _validate_stats,
    _validate_tasks,
    validate_release_thresholds,
)
from data.flip_table_data_augmentation.io_utils import sha256_file


def _render_manifest(root: Path, *, frame_count: int = 2) -> Path:
    numeric = root / "numeric.parquet"
    numeric.write_bytes(b"fixture")
    cameras = {}
    for key in EXPECTED_CAMERA_KEYS:
        directory = root / key.replace(".", "_")
        directory.mkdir(parents=True)
        for index in range(frame_count):
            (directory / f"frame_{index:06d}.png").write_bytes(b"fixture")
        cameras[key] = directory.relative_to(root).as_posix()
    payload = {
        "schema_version": RENDER_MANIFEST_SCHEMA_VERSION,
        "candidate_id": "candidate-000001",
        "trajectory_kind": "mimic",
        "source_kind": "real_demo",
        "appearance_variant": 0,
        "source_episode_indices": [0, 7],
        "source_trajectory_lineage": "source:0+7:segment-hash",
        "frame_count": frame_count,
        "fps": 30,
        "task": TASK,
        "numeric_trace": numeric.name,
        "numeric_trace_sha256": sha256_file(numeric),
        "cameras": cameras,
        "trajectory_sha256": "a" * 64,
        "runtime_manifest_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "randomization": {
            "seed": 41,
            "lighting": {"intensity_lm": 1200},
            "trajectory_sampling": {"source_frame_count": frame_count * 5 // 3},
        },
        "success_report": {
            "accepted": True,
            "strict_v1_contract": True,
            "rejection_reasons": [],
            "action_fk_report": {
                "schema_version": SYNTHETIC_ACTION_FK_SCHEMA_VERSION,
                "frame_count": frame_count * 5 // 3,
                "pass": True,
            },
        },
    }
    path = root / "render_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_rendered_episode_requires_contiguous_three_camera_frames(tmp_path) -> None:
    manifest = _render_manifest(tmp_path)
    episode = RenderedEpisode.load(manifest)
    assert episode.source_episode_indices == (0, 7)
    assert tuple(episode.camera_dirs) == EXPECTED_CAMERA_KEYS

    next(iter(episode.camera_dirs.values())).joinpath("frame_000001.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        RenderedEpisode.load(manifest)


def test_rendered_episode_rejects_claimed_success_without_strict_contract(tmp_path) -> None:
    manifest = _render_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["success_report"]["strict_v1_contract"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="strict-V1"):
        RenderedEpisode.load(manifest)


def test_rendered_episode_rejects_a_second_task_label(tmp_path) -> None:
    manifest = _render_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["task"] = "flip the already assembled table over"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="rendered episode task"):
        RenderedEpisode.load(manifest)


def test_lineage_split_is_stable_and_does_not_depend_on_variant() -> None:
    weights = {"train": 0.9, "validation": 0.05, "test": 0.05}
    first = lineage_split("source-lineage-17", weights)
    assert first in weights
    assert lineage_split("source-lineage-17", weights) == first


def test_real_source_subset_uses_local_indices_without_losing_original_lineage() -> None:
    config = load_pipeline_config()
    selected = (0, 3, 9, 10, 11, 24, 31, 32, 36, 44, 62)
    real_groups, _, ordered = _lineage_split_groups(selected, (), config)
    local_groups = _local_real_split_indices(selected, real_groups)

    assert sorted(index for values in local_groups.values() for index in values) == list(
        range(len(selected))
    )
    assert sorted(value for kind, value in ordered if kind == "real") == list(selected)


def test_lerobot_split_ranges_must_cover_every_episode_contiguously() -> None:
    info = {
        "splits": {"train": "0:8", "validation": "8:9", "test": "9:10"},
    }
    assert _episode_splits(info, 10) == (
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "train",
        "validation",
        "test",
    )

    info["splits"]["validation"] = "7:9"
    with pytest.raises(ValueError, match="non-contiguous"):
        _episode_splits(info, 10)


def test_smoke_dataset_may_omit_empty_splits_but_final_dataset_may_not() -> None:
    info = {"splits": {"train": "0:2", "test": "2:3"}}
    assert _episode_splits(info, 3, require_all=False) == (
        "train",
        "train",
        "test",
    )
    with pytest.raises(ValueError, match="final info.splits"):
        _episode_splits(info, 3, require_all=True)

    wrong_order = {"splits": {"test": "0:1", "train": "1:2"}}
    with pytest.raises(ValueError, match="canonical order"):
        _episode_splits(wrong_order, 2, require_all=False)


def test_final_release_thresholds_cannot_be_weakened() -> None:
    config = load_pipeline_config()
    validate_release_thresholds(
        config,
        require_full_source=False,
        minimum_synthetic_trajectories=1,
        minimum_appearance_variants=1,
    )
    validate_release_thresholds(
        config,
        require_full_source=True,
        minimum_synthetic_trajectories=2000,
        minimum_appearance_variants=2,
    )
    with pytest.raises(ValueError, match="cannot be weakened"):
        validate_release_thresholds(
            config,
            require_full_source=True,
            minimum_synthetic_trajectories=1999,
            minimum_appearance_variants=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        validate_release_thresholds(
            config,
            require_full_source=False,
            minimum_synthetic_trajectories=0,
            minimum_appearance_variants=1,
        )


def test_source_task_column_is_normalized_without_editing_source(tmp_path) -> None:
    import pandas as pd

    metadata = tmp_path / "meta"
    metadata.mkdir()
    source = pd.DataFrame({"task_index": [0], "__index_level_0__": ["flip table"]})
    source.to_parquet(metadata / "tasks.parquet", index=False)
    canonical = _canonical_source_tasks(tmp_path)
    assert canonical.columns.tolist() == ["task_index"]
    assert canonical.index.tolist() == ["flip table"]
    assert canonical.index.name == "task"
    assert source.equals(pd.read_parquet(metadata / "tasks.parquet"))


def test_source_rgb_depth_flags_are_normalized_in_memory() -> None:
    features = {
        key: {
            "dtype": "video",
            "info": {"video.is_depth_map": False},
        }
        for key in EXPECTED_CAMERA_KEYS
    }
    _canonicalize_video_depth_metadata(features)
    for feature in features.values():
        assert feature["info"] == {"is_depth_map": False}


def test_final_task_table_and_episode_tasks_are_canonical(tmp_path) -> None:
    import pandas as pd

    tasks = pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index(["flip table"], name="task"),
    )
    path = tmp_path / "tasks.parquet"
    tasks.to_parquet(path)
    _validate_tasks(path, [{"tasks": ["flip table"]}], 1)

    with pytest.raises(ValueError, match="every episode"):
        _validate_tasks(path, [{"tasks": ["other"]}], 1)


def test_lerobot_image_sampling_count_is_deterministic() -> None:
    assert image_sample_indices(1).tolist() == [0]
    assert image_sample_indices(101)[0] == 0
    assert image_sample_indices(101)[-1] == 100
    lengths = [1, 101, 867]
    assert _expected_image_stat_count(lengths) == sum(
        len(image_sample_indices(length)) for length in lengths
    )


def test_numeric_arrow_storage_requires_float32_and_exact_width() -> None:
    import pyarrow as pa

    column = pa.chunked_array([pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32()))])
    assert _numeric_array(column, key="feature", width=2).shape == (2, 2)

    wrong_dtype = pa.chunked_array([pa.array([[1.0, 2.0]], type=pa.list_(pa.float64()))])
    with pytest.raises(ValueError, match="float32"):
        _numeric_array(wrong_dtype, key="feature", width=2)
    wrong_width = pa.chunked_array([pa.array([[1.0], [2.0, 3.0]], type=pa.list_(pa.float32()))])
    with pytest.raises(ValueError, match="width"):
        _numeric_array(wrong_width, key="feature", width=2)


def test_numeric_stats_are_checked_against_full_recomputation() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float64)
    feature_stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        "q01": [1.0, 2.0],
        "q10": [1.0, 2.0],
        "q50": [3.0, 6.0],
        "q90": [5.0, 10.0],
        "q99": [5.0, 10.0],
    }
    image_stats = {
        name: np.full((3, 1, 1), 0.5).tolist()
        for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
    }
    image_stats["count"] = [7]
    stats = {"feature": feature_stats, **{key: image_stats for key in EXPECTED_CAMERA_KEYS}}
    _validate_stats(stats, numeric_values={"feature": values}, total_frames=3, image_sample_count=7)

    stats["feature"]["mean"][0] += 0.1
    with pytest.raises(ValueError, match="full Parquet recomputation"):
        _validate_stats(stats, numeric_values={"feature": values}, total_frames=3, image_sample_count=7)


def test_content_manifest_detects_added_and_modified_files(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    value = tmp_path / "meta" / "info.json"
    value.write_text("{}\n", encoding="utf-8")
    write_file_manifest(tmp_path)
    assert verify_file_manifest(tmp_path)["file_count"] == 1

    value.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        verify_file_manifest(tmp_path)


def test_hf_upload_workspace_is_resumable_and_does_not_mutate_dataset(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "README.md").write_text("dataset\n", encoding="utf-8")
    manifest = write_file_manifest(dataset)
    local_files = tuple(
        sorted(path.relative_to(dataset).as_posix() for path in dataset.rglob("*") if path.is_file())
    )
    verification = tmp_path / "verification"
    arguments = {
        "dataset_root": dataset,
        "repo_id": "Team-RAMEN/test-private-dataset",
        "config_sha256": "a" * 64,
        "manifest_sha256": sha256_file(manifest),
        "local_files": local_files,
    }
    upload = _prepare_upload_workspace(verification, **arguments)
    assert (upload / "README.md").read_text(encoding="utf-8") == "dataset\n"
    assert verify_file_manifest(dataset)["file_count"] == 1

    cache_file = upload / ".cache" / ".huggingface" / "resume.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{}\n", encoding="utf-8")
    assert _prepare_upload_workspace(verification, **arguments) == upload
    assert cache_file.is_file()
    assert verify_file_manifest(dataset)["file_count"] == 1

    with pytest.raises(ValueError, match="different dataset upload"):
        _prepare_upload_workspace(
            verification,
            **{**arguments, "manifest_sha256": "b" * 64},
        )
