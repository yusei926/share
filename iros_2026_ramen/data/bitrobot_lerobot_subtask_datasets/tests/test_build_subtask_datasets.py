from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts" / "build_subtask_datasets.py"
    spec = importlib.util.spec_from_file_location("build_subtask_datasets", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_subtask_config_contract() -> None:
    module = load_builder()
    config = module.load_json(ROOT / "configs" / "subtasks.json")
    specs = module.load_subtask_specs(config)

    assert config["source_repo_id"] == "BitRobot/G1_WBT_Dex1_Building-Children-Table"
    assert config["raw_repo_id"] == "BitRobot/2026-humanoid-ikea-assembly-challenge"
    assert config["target_repo_template"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_{subtask}_1"
    assert [spec.name for spec in specs] == [
        "move_to_work_pose",
        "pick_leg",
        "coarse_insert",
        "final_insert_contact",
        "tighten",
        "rotate_table_base",
        "recover_or_regrasp",
        "flip_table",
    ]


def test_plan_segments_direct_and_split_labels(monkeypatch) -> None:
    module = load_builder()
    specs = module.load_subtask_specs(module.load_json(ROOT / "configs" / "subtasks.json"))
    raw_info = {
        "episode_name": "episode_test",
        "start_timestamp_ns": 1_000_000_000,
        "end_timestamp_ns": 11_000_000_000,
        "subtasks": [
            {"task": "pick table leg", "timestamp_ns": 2_000_000_000},
            {"task": "insert table leg to table base", "timestamp_ns": 5_000_000_000},
            {"task": "flip table", "timestamp_ns": 9_000_000_000},
        ],
    }

    def fake_hf_download(*, repo_id: str, repo_type: str, filename: str) -> str:
        assert filename == "episode_test/episode_0001/info.json"
        return "/tmp/fake-info.json"

    def fake_load_json(path: Path):
        assert path == Path("/tmp/fake-info.json")
        return raw_info

    monkeypatch.setattr(module, "hf_hub_download", fake_hf_download)
    monkeypatch.setattr(module, "load_json", fake_load_json)
    segments = module.plan_segments(
        raw_repo_id="raw",
        raw_info_files=["episode_test/episode_0001/info.json"],
        episode_rows=[{"episode_index": 0}],
        specs=specs,
        min_duration_sec=0.1,
    )

    assert [(seg.start_sec, seg.end_sec) for seg in segments["pick_leg"]] == [(1.0, 4.0)]
    assert [(round(seg.start_sec, 3), round(seg.end_sec, 3)) for seg in segments["coarse_insert"]] == [(4.0, 6.6)]
    assert [(round(seg.start_sec, 3), round(seg.end_sec, 3)) for seg in segments["final_insert_contact"]] == [(6.6, 8.0)]
    assert [(seg.start_sec, seg.end_sec) for seg in segments["flip_table"]] == [(8.0, 10.0)]


def test_data_path_from_episode() -> None:
    module = load_builder()
    assert (
        module.data_path_from_episode({"data/chunk_index": 2, "data/file_index": 7})
        == "data/chunk-002/file-007.parquet"
    )


def test_storage_defaults_and_chunk_rollover() -> None:
    module = load_builder()

    assert module.DEFAULT_DATA_FILE_SIZE_MB == 100
    assert module.DEFAULT_VIDEO_FILE_SIZE_MB == 500
    assert module.next_file_location(module.FileLocation(0, 998), 1000) == module.FileLocation(0, 999)
    assert module.next_file_location(module.FileLocation(0, 999), 1000) == module.FileLocation(1, 0)


def test_write_info_keeps_lerobot_v3_storage_fields(tmp_path: Path) -> None:
    module = load_builder()
    features = {
        "observation.state.robot_q_current": {"dtype": "float32", "shape": [32], "names": None},
        "action.robot_q_desired": {"dtype": "float32", "shape": [32], "names": None},
    }
    module.write_info(
        output_dir=tmp_path,
        source_info={
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1",
            "fps": 30,
            "features": {},
            "dataset_name": "should_not_be_copied",
            "team_ramen": {"subtask": "old"},
        },
        features=features,
        total_episodes=2,
        total_frames=120,
        chunks_size=1000,
        data_files_size_mb=100,
        video_files_size_mb=500,
    )

    info = module.load_json(tmp_path / "meta" / "info.json")
    assert info["codebase_version"] == "v3.0"
    assert info["robot_type"] == "unitree_g1"
    assert info["data_files_size_in_mb"] == 100
    assert info["video_files_size_in_mb"] == 500
    assert info["features"] == features
    assert "dataset_name" not in info
    assert "team_ramen" not in info
