from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_lerobot_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_lerobot_dataset", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_info(total_episodes: int, total_frames: int) -> dict:
    video = {
        "dtype": "video",
        "shape": [480, 640, 3],
        "info": {"video.fps": 30},
    }
    return {
        "codebase_version": "v3.0",
        "fps": 30,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": {
            "observation.images.cam_0": video,
            "observation.images.cam_2": video,
            "observation.images.cam_3": video,
            "observation.state.ee_state": {"dtype": "float32", "shape": [12]},
            "observation.state.robot_q_current": {"dtype": "float32", "shape": [36]},
            "observation.state.hand_state": {"dtype": "float32", "shape": [2]},
            "action.ee_action": {"dtype": "float32", "shape": [12]},
            "action.robot_q_desired": {"dtype": "float32", "shape": [36]},
            "action.hand_cmd": {"dtype": "float32", "shape": [2]},
        },
    }


def test_default_reports_live_inside_the_feature_directory() -> None:
    module = load_module()

    assert module.DEFAULT_AUDIT_OUTPUT == ROOT / "outputs" / "audits" / "flip_table_dataset_audit.json"
    assert module.DEFAULT_SPLIT_OUTPUT == ROOT / "outputs" / "audits" / "flip_table_episode_split.json"


def make_episodes(count: int, length: int = 10) -> list[dict]:
    episodes = []
    for index in range(count):
        episodes.append(
            {
                "episode_index": index,
                "tasks": ["flip table"],
                "length": length,
                "dataset_from_index": index * length,
                "dataset_to_index": (index + 1) * length,
                "source_episode_name": f"source-{index:03d}",
                "source_start_sec": 1.0,
                "source_end_sec": 2.0,
            }
        )
    return episodes


def test_episode_metadata_accepts_contiguous_provenance() -> None:
    module = load_module()
    episodes = make_episodes(4)
    errors, warnings, summary = module.audit_episode_metadata(make_info(4, 40), episodes)

    assert errors == []
    assert warnings == ["no explicit per-episode success label; demonstration success needs visual audit"]
    assert summary["unique_source_episodes"] == 4
    assert summary["episode_length_frames"]["p50"] == 10.0


def test_episode_metadata_rejects_noncontiguous_ranges() -> None:
    module = load_module()
    episodes = make_episodes(2)
    episodes[1]["dataset_from_index"] = 11
    errors, _, _ = module.audit_episode_metadata(make_info(2, 20), episodes)

    assert any("dataset ranges" in error for error in errors)


def test_episode_metadata_requires_relative_eef_source_fields() -> None:
    module = load_module()
    info = make_info(1, 10)
    del info["features"]["action.ee_action"]

    errors, _, _ = module.audit_episode_metadata(info, make_episodes(1))

    assert "missing required numeric feature action.ee_action" in errors


def test_grouped_split_is_disjoint_and_reproducible() -> None:
    module = load_module()
    episodes = make_episodes(20)
    split_a = module.assign_grouped_splits(episodes, 0.1, 0.1, seed=42)
    split_b = module.assign_grouped_splits(episodes, 0.1, 0.1, seed=42)

    assert split_a == split_b
    train = set(split_a["splits"]["train"]["episode_indices"])
    validation = set(split_a["splits"]["validation"]["episode_indices"])
    test = set(split_a["splits"]["test"]["episode_indices"])
    assert len(train) == 16
    assert len(validation) == 2
    assert len(test) == 2
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == set(range(20))


def test_grouped_split_keeps_duplicate_source_slices_together() -> None:
    module = load_module()
    episodes = make_episodes(10)
    episodes[1]["source_episode_name"] = episodes[0]["source_episode_name"]
    split = module.assign_grouped_splits(episodes, 0.2, 0.2, seed=7)

    containing_splits = [
        name
        for name, values in split["splits"].items()
        if 0 in values["episode_indices"] or 1 in values["episode_indices"]
    ]
    assert len(containing_splits) == 1
    assert {0, 1}.issubset(split["splits"][containing_splits[0]]["episode_indices"])


def test_distribution_summary_ignores_nonfinite_values() -> None:
    module = load_module()
    summary = module.distribution_summary(module.np.asarray([1.0, 2.0, module.np.nan]))

    assert summary["count"] == 2
    assert summary["min"] == 1.0
    assert summary["max"] == 2.0


def test_fixed_size_numeric_columns_are_supported() -> None:
    module = load_module()
    array = pa.array([[1.0, 2.0], [3.0, 4.0]], type=pa.list_(pa.float32(), 2))

    values = module.list_array_to_numpy(array, expected_width=2)

    assert values.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_small_grouped_split_never_silently_omits_requested_holdouts() -> None:
    module = load_module()
    split = module.assign_grouped_splits(make_episodes(3), 0.1, 0.1, seed=42)

    assert len(split["splits"]["train"]["source_episode_names"]) == 1
    assert len(split["splits"]["validation"]["source_episode_names"]) == 1
    assert len(split["splits"]["test"]["source_episode_names"]) == 1
