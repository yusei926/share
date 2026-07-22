"""Pinned source download and LeRobot v3 source-contract validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig


NUMERIC_FEATURES = {
    "observation.state.ee_state": ("float32", 12),
    "observation.state.hand_state": ("float32", 2),
    "observation.state.robot_q_current": ("float32", 36),
    "action.ee_action": ("float32", 12),
    "action.hand_cmd": ("float32", 2),
    "action.robot_q_desired": ("float32", 36),
}
INDEX_FEATURES = ("timestamp", "frame_index", "episode_index", "index", "task_index")


@dataclass(frozen=True)
class SourceAudit:
    source_root: Path
    source_revision: str
    episodes: int
    frames: int
    data_files: int
    metadata_files: int
    declared_arrow_casts: tuple[str, ...]
    exact_float32_roundtrip_features: tuple[str, ...]


def validate_source_info(info: dict[str, Any], config: PipelineConfig) -> None:
    errors: list[str] = []
    expected_scalars = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "fps": config.source.fps,
        "total_episodes": config.source.episodes,
        "total_frames": config.source.frames,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
    }
    for key, expected in expected_scalars.items():
        if info.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {info.get(key)!r}")

    features = info.get("features")
    if not isinstance(features, dict):
        errors.append("features must be an object")
        features = {}
    for key, (dtype, width) in NUMERIC_FEATURES.items():
        value = features.get(key)
        if not isinstance(value, dict) or value.get("dtype") != dtype or value.get("shape") != [width]:
            errors.append(f"{key}: expected {dtype}[{width}], got {value!r}")
    for camera in config.cameras:
        value = features.get(camera.source_key)
        expected_shape = [camera.height, camera.width, 3]
        if not isinstance(value, dict) or value.get("dtype") != "video" or value.get("shape") != expected_shape:
            errors.append(f"{camera.source_key}: expected video{expected_shape}, got {value!r}")
            continue
        video_info = value.get("info", {})
        expected_video = {
            "video.height": camera.height,
            "video.width": camera.width,
            "video.fps": camera.fps,
            "video.channels": 3,
            "video.is_depth_map": False,
            "has_audio": False,
        }
        for key, expected in expected_video.items():
            if video_info.get(key) != expected:
                errors.append(
                    f"{camera.source_key}.{key}: expected {expected!r}, got {video_info.get(key)!r}"
                )
    if errors:
        raise ValueError("source info contract failed:\n- " + "\n- ".join(errors))


def snapshot_download_pinned(
    config: PipelineConfig,
    *,
    local_dir: Path | None = None,
    include_videos: bool = False,
) -> Path:
    """Download only the configured immutable source revision."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download the source dataset") from exc
    patterns = [".gitattributes", "README.md", "meta/**", "data/**"]
    if include_videos:
        patterns.append("videos/**")
    path = snapshot_download(
        repo_id=config.source.repo_id,
        repo_type="dataset",
        revision=config.source.revision,
        local_dir=None if local_dir is None else str(local_dir),
        allow_patterns=patterns,
    )
    resolved = Path(path).resolve()
    validate_source_info(json.loads((resolved / "meta" / "info.json").read_text(encoding="utf-8")), config)
    return resolved


def download_pinned_source_files(
    config: PipelineConfig,
    source_root: Path,
    relative_paths: Iterable[str | Path],
) -> tuple[Path, ...]:
    """Materialize selected files in the immutable source snapshot."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download source files") from exc
    root = Path(source_root).resolve()
    results = []
    for value in relative_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"source path must be a safe relative path: {value}")
        expected = root / relative
        if expected.is_file():
            results.append(expected)
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=config.source.repo_id,
                repo_type="dataset",
                revision=config.source.revision,
                filename=relative.as_posix(),
            )
        ).resolve()
        if not expected.is_file() or expected.resolve() != downloaded:
            raise FileNotFoundError(
                f"pinned source file was not materialized in the active snapshot: {relative}"
            )
        results.append(expected)
    return tuple(results)


def _iter_parquets(root: Path, relative: str) -> list[Path]:
    return sorted((root / relative).glob("chunk-*/*.parquet"))


def _is_floating_list(arrow_type: Any) -> bool:
    import pyarrow as pa

    return (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ) and pa.types.is_floating(arrow_type.value_type)


def _contiguous(values: Iterable[int], *, start: int = 0) -> bool:
    return all(value == expected for expected, value in enumerate(values, start=start))


def audit_source_dataset(source_root: Path, config: PipelineConfig) -> SourceAudit:
    """Perform the metadata and tabular checks needed before FK conversion.

    The source Parquet files currently store several declared float32 features
    as Arrow double lists. This audit records those deterministic export casts;
    it never changes the source snapshot.
    """

    try:
        import pyarrow as pa
        import pyarrow.dataset as pads
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for full source auditing") from exc

    root = Path(source_root).resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    validate_source_info(info, config)
    data_files = _iter_parquets(root, "data")
    episode_files = _iter_parquets(root, "meta/episodes")
    if not data_files or not episode_files or not (root / "meta" / "tasks.parquet").is_file():
        raise ValueError("source snapshot is missing data, episode metadata, or tasks")

    dataset = pads.dataset([str(path) for path in data_files], format="parquet")
    casts: list[str] = []
    for key, (declared_dtype, _width) in NUMERIC_FEATURES.items():
        field = dataset.schema.field(key)
        if not _is_floating_list(field.type):
            raise ValueError(f"{key} must be an Arrow list of floats, got {field.type}")
        if declared_dtype == "float32" and not pa.types.is_float32(field.type.value_type):
            casts.append(f"{key}:{field.type.value_type}->float32")
    timestamp = dataset.schema.field("timestamp").type
    if not pa.types.is_floating(timestamp):
        raise ValueError(f"timestamp must be floating, got {timestamp}")
    if not pa.types.is_float32(timestamp):
        casts.append(f"timestamp:{timestamp}->float32")
    for key in INDEX_FEATURES[1:]:
        if not pa.types.is_int64(dataset.schema.field(key).type):
            raise ValueError(f"{key} must be int64")
    if dataset.count_rows() != config.source.frames:
        raise ValueError(f"source has {dataset.count_rows()} rows, expected {config.source.frames}")

    exact_roundtrips: list[str] = []
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        for batch in dataset.scanner(columns=[key], batch_size=65_536).to_batches():
            column = batch.column(0)
            if column.null_count:
                raise ValueError(f"source {key} contains null rows")
            if pa.types.is_fixed_size_list(column.type):
                valid_width = column.type.list_size == width
            else:
                offsets = np.asarray(column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
                valid_width = bool(np.all(np.diff(offsets) == width))
            if not valid_width:
                raise ValueError(f"source {key} contains a row whose width differs from {width}")
            values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"source {key} contains NaN or Inf")
            if not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
                raise ValueError(f"source {key} cannot be losslessly materialized as declared float32")
        exact_roundtrips.append(key)
    for batch in dataset.scanner(columns=["timestamp"], batch_size=65_536).to_batches():
        values = np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(values).all() or not np.array_equal(
            values, values.astype(np.float32).astype(np.float64)
        ):
            raise ValueError("source timestamp cannot be losslessly materialized as declared float32")
    exact_roundtrips.append("timestamp")

    episode_table = pq.read_table([str(path) for path in episode_files])
    if episode_table.num_rows != config.source.episodes:
        raise ValueError(
            f"source has {episode_table.num_rows} episode rows, expected {config.source.episodes}"
        )
    episode_indices = [int(value) for value in episode_table["episode_index"].to_pylist()]
    if not _contiguous(sorted(episode_indices)):
        raise ValueError("episode_index metadata is not contiguous from zero")
    lengths = [int(value) for value in episode_table["length"].to_pylist()]
    if any(length <= 0 for length in lengths) or sum(lengths) != config.source.frames:
        raise ValueError("episode lengths are non-positive or do not sum to total_frames")

    return SourceAudit(
        source_root=root,
        source_revision=config.source.revision,
        episodes=len(episode_indices),
        frames=sum(lengths),
        data_files=len(data_files),
        metadata_files=len(episode_files) + 2,
        declared_arrow_casts=tuple(sorted(casts)),
        exact_float32_roundtrip_features=tuple(sorted(exact_roundtrips)),
    )
