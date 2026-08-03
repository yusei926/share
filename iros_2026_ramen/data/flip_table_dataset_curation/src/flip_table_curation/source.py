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
        relative = str(self.info["video_path"]).format(
            video_key=key,
            chunk_index=int(row[f"videos/{key}/chunk_index"]),
            file_index=int(row[f"videos/{key}/file_index"]),
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
    include_data: bool,
    include_videos: bool,
    video_episode_indices: Iterable[int] | None = None,
) -> SourceSnapshot:
    metadata_patterns = ["README.md", "meta/**"]
    root = Path(
        snapshot_download(
            repo_id=config.source_repo_id,
            repo_type="dataset",
            revision=config.source_revision,
            allow_patterns=metadata_patterns,
            cache_dir=config.workspace / "hf_cache",
        )
    )
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))

    patterns = list(metadata_patterns)
    if include_data:
        patterns.append("data/**")
    if include_videos:
        if video_episode_indices is None:
            patterns.extend(f"videos/{key}/**" for key in VIDEO_KEYS)
        else:
            requested = {int(value) for value in video_episode_indices}
            by_episode = {int(row["episode_index"]): row for row in rows}
            unknown = sorted(requested - set(by_episode))
            if unknown:
                raise ValueError(f"requested videos for unknown source episodes: {unknown[:10]}")
            for key in VIDEO_KEYS:
                for episode_index in sorted(requested):
                    row = by_episode[episode_index]
                    patterns.append(
                        str(info["video_path"]).format(
                            video_key=key,
                            chunk_index=int(row[f"videos/{key}/chunk_index"]),
                            file_index=int(row[f"videos/{key}/file_index"]),
                        )
                    )
    if patterns != metadata_patterns:
        root = Path(
            snapshot_download(
                repo_id=config.source_repo_id,
                repo_type="dataset",
                revision=config.source_revision,
                allow_patterns=patterns,
                cache_dir=config.workspace / "hf_cache",
            )
        )
    return SourceSnapshot(root=root, info=info, episodes=tuple(rows))


def read_numeric_table(snapshot: SourceSnapshot) -> pa.Table:
    paths = sorted((snapshot.root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError("source numeric parquet is missing")
    return pa.concat_tables([pq.read_table(path) for path in paths])


def episode_slice(table: pa.Table, row: dict[str, Any]) -> pa.Table:
    start = int(row["dataset_from_index"])
    end = int(row["dataset_to_index"])
    length = int(row["length"])
    if end - start != length:
        raise ValueError(f"episode {row['episode_index']} metadata range is invalid")
    result = table.slice(start, length)
    episode_indices = np.asarray(result["episode_index"].to_numpy(), dtype=np.int64)
    if len(result) != length or not np.all(episode_indices == int(row["episode_index"])):
        raise ValueError(f"episode {row['episode_index']} numeric rows are not contiguous")
    return result


def source_episode_name(episode_index: int) -> str:
    return f"source_episode_{episode_index:04d}"
