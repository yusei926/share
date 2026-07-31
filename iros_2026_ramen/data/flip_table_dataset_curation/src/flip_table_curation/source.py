from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from .config import CurationConfig


RGB_KEYS = tuple(f"observation.images.cam_{index}" for index in range(4))
IR_KEYS = tuple(f"observation.images.cam_{index}_ir" for index in range(4))
VIDEO_KEYS = RGB_KEYS + IR_KEYS
NUMERIC_WIDTHS = {
    "observation.state.ee_state": 12,
    "observation.state.hand_state": 2,
    "observation.state.robot_q_current": 36,
    "action.ee_action": 12,
    "action.hand_cmd": 2,
    "action.robot_q_desired": 36,
}
INDEX_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    info: dict[str, Any]
    episodes: tuple[dict[str, Any], ...]

    @property
    def fps(self) -> int:
        return int(self.info["fps"])

    def video_path(self, row: dict[str, Any], key: str) -> Path:
        chunk = int(row[f"videos/{key}/chunk_index"])
        file_index = int(row[f"videos/{key}/file_index"])
        relative = str(self.info["video_path"]).format(
            video_key=key, chunk_index=chunk, file_index=file_index
        )
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def video_offset(self, row: dict[str, Any], key: str) -> float:
        return float(row[f"videos/{key}/from_timestamp"])


def download_source(
    config: CurationConfig,
    *,
    include_videos: bool,
    rgb_only: bool = False,
) -> SourceSnapshot:
    patterns = ["README.md", "meta/**", "data/**"]
    if include_videos:
        keys = RGB_KEYS if rgb_only else VIDEO_KEYS
        patterns.extend(f"videos/{key}/**" for key in keys)
    root = Path(
        snapshot_download(
            repo_id=config.source_repo_id,
            repo_type="dataset",
            revision=config.source_revision,
            allow_patterns=patterns,
            cache_dir=config.workspace / "hf_cache",
        )
    )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    return SourceSnapshot(root=root, info=info, episodes=tuple(rows))


def read_numeric_table(snapshot: SourceSnapshot) -> pa.Table:
    paths = sorted((snapshot.root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError("source numeric parquet is missing")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def fixed_list_numpy(column: Any, width: int) -> np.ndarray:
    values = np.asarray(column.to_pylist(), dtype=np.float64)
    if values.shape != (len(column), width):
        raise ValueError(f"column must have shape ({len(column)}, {width}), got {values.shape}")
    return values


def episode_slice(table: pa.Table, row: dict[str, Any]) -> pa.Table:
    start = int(row["dataset_from_index"])
    end = int(row["dataset_to_index"])
    length = int(row["length"])
    if end - start != length:
        raise ValueError(f"episode {row['episode_index']} metadata range is invalid")
    result = table.slice(start, length)
    episodes = np.asarray(result["episode_index"].to_numpy(), dtype=np.int64)
    if len(result) != length or not np.all(episodes == int(row["episode_index"])):
        raise ValueError(f"episode {row['episode_index']} numeric rows are not contiguous")
    return result


def source_inventory(snapshot: SourceSnapshot) -> dict[str, Any]:
    features = snapshot.info.get("features", {})
    missing_videos = sorted(set(VIDEO_KEYS) - set(features))
    missing_numeric = sorted(set(NUMERIC_WIDTHS) - set(features))
    return {
        "root": str(snapshot.root),
        "fps": snapshot.info.get("fps"),
        "episodes": len(snapshot.episodes),
        "frames": sum(int(row["length"]) for row in snapshot.episodes),
        "missing_video_features": missing_videos,
        "missing_numeric_features": missing_numeric,
        "unique_source_episode_names": len(
            {str(row.get("source_episode_name", "")) for row in snapshot.episodes}
        ),
    }

