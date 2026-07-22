"""Full LeRobot v3, lineage, shard, and video integrity validation."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .contracts import LINEAGE_SCHEMA_VERSION, NUMERIC_KEYS, TASK
from ..fk_audit import SYNTHETIC_ACTION_FK_SCHEMA_VERSION
from .file_manifest import verify_file_manifest
from .recompute_stats import STATS_REPORT_SCHEMA_VERSION
from ..config import EXPECTED_CAMERA_KEYS, PipelineConfig
from ..io_utils import sha256_file
from ..source_contract import INDEX_FEATURES, NUMERIC_FEATURES


SPLIT_NAMES = ("train", "validation", "test")
STAT_VALUE_KEYS = ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99")
SHARD_SIZE_TOLERANCE = 1.1


def validate_release_thresholds(
    config: PipelineConfig,
    *,
    require_full_source: bool,
    minimum_synthetic_trajectories: int,
    minimum_appearance_variants: int,
) -> None:
    """Reject invalid minima and prevent weakening the final release gates."""

    values = {
        "minimum_synthetic_trajectories": minimum_synthetic_trajectories,
        "minimum_appearance_variants": minimum_appearance_variants,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not require_full_source:
        return
    configured = {
        "minimum_synthetic_trajectories": int(
            config.raw["generation"]["successful_trajectories_min"]
        ),
        "minimum_appearance_variants": int(
            config.raw["generation"]["appearance_variants_per_trajectory_min"]
        ),
    }
    weakened = {
        name: {"requested": values[name], "configured": configured[name]}
        for name in values
        if values[name] < configured[name]
    }
    if weakened:
        raise ValueError(f"final release thresholds cannot be weakened: {weakened}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parquet_files(root: Path, relative: str) -> tuple[Path, ...]:
    files = tuple(sorted((root / relative).glob("chunk-*/*.parquet")))
    if not files:
        raise ValueError(f"no Parquet files found under {relative}")
    return files


def _read_parquet(paths: tuple[Path, ...], columns: list[str] | None = None):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for complete dataset validation") from exc
    tables = [pq.read_table(path, columns=columns) for path in paths]
    return pa.concat_tables(tables, promote_options="default")


def _expected_image_stat_count(lengths: list[int]) -> int:
    """Match LeRobot 0.6.0's deterministic per-episode image sampling count."""

    total = 0
    for length in lengths:
        minimum = min(length, 100)
        total += max(minimum, min(int(length**0.75), 10_000))
    return total


def _feature_contract(features: dict[str, Any]) -> None:
    expected_keys = set(NUMERIC_FEATURES).union(INDEX_FEATURES, EXPECTED_CAMERA_KEYS)
    if set(features) != expected_keys:
        missing = sorted(expected_keys - set(features))
        extra = sorted(set(features) - expected_keys)
        raise ValueError(
            f"output feature set differs from the policy contract: missing={missing}, extra={extra}"
        )
    video_keys = tuple(key for key, feature in features.items() if feature.get("dtype") == "video")
    if video_keys != EXPECTED_CAMERA_KEYS:
        raise ValueError(f"policy RGB cameras must retain ordered keys {EXPECTED_CAMERA_KEYS}")
    for key, (dtype, width) in NUMERIC_FEATURES.items():
        if features[key].get("dtype") != dtype or features[key].get("shape") != [width]:
            raise ValueError(f"{key} must be declared as {dtype}[{width}]")
    expected_indices = {
        "timestamp": ("float32", [1]),
        "frame_index": ("int64", [1]),
        "episode_index": ("int64", [1]),
        "index": ("int64", [1]),
        "task_index": ("int64", [1]),
    }
    for key, (dtype, shape) in expected_indices.items():
        if features[key].get("dtype") != dtype or features[key].get("shape") != shape:
            raise ValueError(f"{key} must be declared as {dtype}{shape}")
    expected_video_info = {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "is_depth_map": False,
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": False,
    }
    for key in EXPECTED_CAMERA_KEYS:
        feature = features[key]
        if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
            raise ValueError(f"{key} must be declared as a 640x480 RGB video")
        info = feature.get("info")
        if not isinstance(info, dict):
            raise ValueError(f"{key}.info must be an object")
        for field, expected in expected_video_info.items():
            if info.get(field) != expected:
                raise ValueError(f"{key}.{field} must be {expected!r}, got {info.get(field)!r}")


def _numeric_array(column: Any, *, key: str, width: int) -> np.ndarray:
    """Return a dense float64 view while enforcing Arrow float32 list width."""

    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for complete dataset validation") from exc
    value = column.combine_chunks()
    arrow_type = value.type
    if not (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ) or not pa.types.is_float32(arrow_type.value_type):
        raise ValueError(f"{key} Arrow storage must be a list of float32, got {arrow_type}")
    if value.null_count:
        raise ValueError(f"{key} contains null rows")
    if pa.types.is_fixed_size_list(arrow_type):
        if arrow_type.list_size != width:
            raise ValueError(f"{key} Arrow list width is {arrow_type.list_size}, expected {width}")
    else:
        offsets = np.asarray(value.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        if not np.all(np.diff(offsets) == width):
            raise ValueError(f"{key} contains a row whose width differs from {width}")
    flat = np.asarray(value.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    return flat.reshape(len(value), width)


def _stat_array(stats: dict[str, Any], feature: str, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(stats.get(name), dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"stats.{feature}.{name} must be finite with shape {shape}, got {value.shape}")
    return value


def _validate_stats(
    stats: dict[str, Any],
    *,
    numeric_values: dict[str, np.ndarray],
    total_frames: int,
    image_sample_count: int,
) -> None:
    expected_features = set(numeric_values).union(EXPECTED_CAMERA_KEYS)
    if set(stats) != expected_features:
        raise ValueError("meta/stats.json must contain exactly the policy numeric and RGB features")
    for key, values in numeric_values.items():
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            raise ValueError(f"stats for {key} must be an object")
        width = values.shape[1]
        expected = {
            "min": np.min(values, axis=0),
            "max": np.max(values, axis=0),
            "mean": np.mean(values, axis=0),
            "std": np.std(values, axis=0),
        }
        for name, recalculated in expected.items():
            observed = _stat_array(feature_stats, key, name, (width,))
            if not np.allclose(observed, recalculated, rtol=5.0e-5, atol=1.0e-6):
                raise ValueError(f"stats.{key}.{name} differs from a full Parquet recomputation")
        count = _stat_array(feature_stats, key, "count", (1,))
        if int(count[0]) != total_frames:
            raise ValueError(f"stats.{key}.count does not cover every dataset frame")
        for name in ("q01", "q10", "q50", "q90", "q99"):
            _stat_array(feature_stats, key, name, (width,))

    for key in EXPECTED_CAMERA_KEYS:
        feature_stats = stats.get(key)
        if not isinstance(feature_stats, dict):
            raise ValueError(f"stats for {key} must be an object")
        for name in STAT_VALUE_KEYS:
            value = _stat_array(feature_stats, key, name, (3, 1, 1))
            lower = -1.0e-6 if name != "std" else 0.0
            if np.any(value < lower) or np.any(value > 1.0 + 1.0e-6):
                raise ValueError(f"stats.{key}.{name} is outside normalized RGB range")
        count = _stat_array(feature_stats, key, "count", (1,))
        if int(count[0]) != image_sample_count:
            raise ValueError(
                f"stats.{key}.count does not include the deterministic sample from every episode"
            )


def _validate_tasks(path: Path, episode_rows: list[dict[str, Any]], total_tasks: int) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for complete dataset validation") from exc
    frame = pd.read_parquet(path)
    if total_tasks != 1 or list(frame.columns) != ["task_index"]:
        raise ValueError("dataset must declare exactly one task_index column")
    if frame["task_index"].tolist() != [0] or frame.index.tolist() != [TASK]:
        raise ValueError(f"meta/tasks.parquet must map task 0 to {TASK!r}")
    if any(row.get("tasks") != [TASK] for row in episode_rows):
        raise ValueError(f"every episode must declare only the {TASK!r} task")


def _referenced_shards(
    rows: list[dict[str, Any]], *, prefix: str, root: Path, suffix: str
) -> dict[Path, list[dict[str, Any]]]:
    groups: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        chunk = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        if chunk < 0 or file_index < 0:
            raise ValueError(f"{prefix} contains a negative shard index")
        groups[root / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.{suffix}"].append(row)
    return groups


def _validate_shard_set(
    referenced: dict[Path, list[dict[str, Any]]], actual: tuple[Path, ...], *, label: str
) -> None:
    referenced_paths = set(referenced)
    actual_paths = set(actual)
    if referenced_paths != actual_paths:
        missing = sorted(str(path) for path in referenced_paths - actual_paths)
        orphaned = sorted(str(path) for path in actual_paths - referenced_paths)
        raise ValueError(
            f"{label} shard references differ from files: "
            f"missing={missing[:5]}, orphaned={orphaned[:5]}"
        )


def _parquet_uncompressed_bytes(path: Path) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for complete dataset validation") from exc
    metadata = pq.read_metadata(path)
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.row_group(row_group).num_columns)
    )


def _validate_data_shards(
    rows: list[dict[str, Any]], *, root: Path, paths: tuple[Path, ...]
) -> None:
    referenced = _referenced_shards(rows, prefix="data", root=root, suffix="parquet")
    _validate_shard_set(referenced, paths, label="data")
    for path, shard_rows in referenced.items():
        table = _read_parquet((path,), columns=["episode_index"])
        expected_episodes = [int(row["episode_index"]) for row in shard_rows]
        observed = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        if table.num_rows != sum(int(row["length"]) for row in shard_rows):
            raise ValueError(f"{path} row count differs from its episode metadata")
        if list(dict.fromkeys(observed.tolist())) != expected_episodes:
            raise ValueError(f"{path} episode ordering differs from its episode metadata")


def _validate_video_slices(
    rows: list[dict[str, Any]],
    *,
    key: str,
    root: Path,
    probes_by_path: dict[Path, dict[str, Any]],
) -> None:
    prefix = f"videos/{key}"
    referenced = _referenced_shards(rows, prefix=prefix, root=root, suffix="mp4")
    _validate_shard_set(referenced, tuple(probes_by_path), label=key)
    for path, shard_rows in referenced.items():
        expected_from_frame = 0
        for row in shard_rows:
            from_timestamp = float(row[f"{prefix}/from_timestamp"])
            to_timestamp = float(row[f"{prefix}/to_timestamp"])
            if not math.isfinite(from_timestamp) or not math.isfinite(to_timestamp):
                raise ValueError(f"{key} contains a non-finite video timestamp")
            from_frame = round(from_timestamp * 30.0)
            to_frame = round(to_timestamp * 30.0)
            if from_frame != expected_from_frame or to_frame - from_frame != int(row["length"]):
                raise ValueError(f"{key} episode slices are non-contiguous or disagree with episode length")
            expected_from_frame = to_frame
        if expected_from_frame != int(probes_by_path[path]["frames"]):
            raise ValueError(f"{key} metadata slices do not cover every decoded shard frame")


def _probe_video(path: Path, *, full_decode: bool) -> dict[str, Any]:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,avg_frame_rate,nb_read_frames,pix_fmt",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise ValueError(f"ffprobe failed for {path}: {probe.stderr.strip()}")
    payload = json.loads(probe.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise ValueError(f"{path} must contain exactly one video stream and no audio")
    stream = streams[0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise ValueError(f"{path} must use H.264 yuv420p")
    if (int(stream.get("width", 0)), int(stream.get("height", 0))) != (640, 480):
        raise ValueError(f"{path} must be 640x480")
    numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/", 1)
    fps = float(numerator) / float(denominator)
    if not math.isclose(fps, 30.0, abs_tol=1.0e-3):
        raise ValueError(f"{path} must be 30 fps, got {fps}")
    frames = int(stream.get("nb_read_frames", 0))
    if frames <= 0:
        raise ValueError(f"{path} contains no decodable frames")
    if full_decode:
        decode = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            check=False,
            capture_output=True,
            text=True,
        )
        if decode.returncode != 0 or decode.stderr.strip():
            raise ValueError(f"full video decode failed for {path}: {decode.stderr.strip()}")
    return {"frames": frames, "fps": fps, "pixel_format": stream.get("pix_fmt")}


def _load_lineage(root: Path) -> tuple[dict[str, Any], ...]:
    path = root / "meta" / "augmentation" / "episodes.jsonl"
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid lineage JSON on line {line_number}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != LINEAGE_SCHEMA_VERSION:
            raise ValueError(f"invalid lineage schema on line {line_number}")
        records.append(value)
    if not records:
        raise ValueError("lineage sidecar cannot be empty")
    return tuple(records)


def _episode_splits(
    info: dict[str, Any], total_episodes: int, *, require_all: bool = True
) -> tuple[str, ...]:
    raw_splits = info.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("info.splits must be a non-empty object")
    split_names = tuple(raw_splits)
    if any(name not in SPLIT_NAMES for name in split_names):
        raise ValueError(f"info.splits contains a name outside {SPLIT_NAMES}")
    expected_names = tuple(name for name in SPLIT_NAMES if name in raw_splits)
    if split_names != expected_names:
        raise ValueError(f"info.splits keys must follow the canonical order {SPLIT_NAMES}")
    if require_all and split_names != SPLIT_NAMES:
        raise ValueError(f"final info.splits must contain ordered keys {SPLIT_NAMES}")
    episode_splits: list[str] = []
    expected_start = 0
    for name in split_names:
        value = raw_splits[name]
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError(f"info.splits.{name} must use the LeRobot start:end format")
        start_text, end_text = value.split(":")
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as exc:
            raise ValueError(f"info.splits.{name} contains a non-integer range") from exc
        if start != expected_start or end <= start or end > total_episodes:
            raise ValueError(f"info.splits.{name} is empty, overlapping, or non-contiguous")
        episode_splits.extend([name] * (end - start))
        expected_start = end
    if expected_start != total_episodes:
        raise ValueError("info.splits ranges do not cover every episode")
    return tuple(episode_splits)


def validate_dataset(
    root: str | Path,
    config: PipelineConfig,
    *,
    full_video_decode: bool,
    require_full_source: bool,
    minimum_synthetic_trajectories: int = 1,
    minimum_appearance_variants: int = 1,
) -> dict[str, Any]:
    """Validate every dataset contract and return a serializable report."""

    validate_release_thresholds(
        config,
        require_full_source=require_full_source,
        minimum_synthetic_trajectories=minimum_synthetic_trajectories,
        minimum_appearance_variants=minimum_appearance_variants,
    )
    dataset_root = Path(root).expanduser().resolve()
    attributes = (dataset_root / ".gitattributes").read_text(encoding="utf-8")
    if "*.mp4 filter=lfs" not in attributes or "*.parquet filter=lfs" not in attributes:
        raise ValueError(".gitattributes must track MP4 and Parquet files with Git LFS")
    info = _json_object(dataset_root / "meta" / "info.json")
    if info.get("codebase_version") != "v3.0":
        raise ValueError("output dataset must be LeRobotDataset v3")
    if info.get("robot_type") != "unitree_g1":
        raise ValueError("output dataset robot_type must be unitree_g1")
    if info.get("fps") != 30:
        raise ValueError("output dataset must be 30 fps")
    if info.get("data_files_size_in_mb") != config.target.data_shard_size_mb:
        raise ValueError("output data shard target differs from pipeline config")
    if info.get("video_files_size_in_mb") != config.target.video_shard_size_mb:
        raise ValueError("output video shard target differs from pipeline config")
    if info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet":
        raise ValueError("output data_path is not the LeRobot v3 canonical path")
    if info.get("video_path") != (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    ):
        raise ValueError("output video_path is not the LeRobot v3 canonical path")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("output info.features must be an object")
    _feature_contract(features)

    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    if total_episodes <= 0 or total_frames <= 0:
        raise ValueError("output episode and frame counts must be positive")
    episode_splits = _episode_splits(
        info, total_episodes, require_all=require_full_source
    )
    episodes = _read_parquet(_parquet_files(dataset_root, "meta/episodes"))
    episode_rows = episodes.to_pylist()
    if len(episode_rows) != total_episodes:
        raise ValueError("episode metadata count differs from info.total_episodes")
    if [int(row["episode_index"]) for row in episode_rows] != list(range(total_episodes)):
        raise ValueError("episode metadata indices must be contiguous and ordered")
    lengths = [int(row["length"]) for row in episode_rows]
    if sum(lengths) != total_frames or any(length <= 0 for length in lengths):
        raise ValueError("episode lengths do not sum to info.total_frames")
    expected_from = 0
    for row, length in zip(episode_rows, lengths, strict=True):
        if int(row["dataset_from_index"]) != expected_from:
            raise ValueError("episode dataset_from_index is not contiguous")
        expected_from += length
        if int(row["dataset_to_index"]) != expected_from:
            raise ValueError("episode dataset_to_index is inconsistent with length")
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    _validate_tasks(tasks_path, episode_rows, int(info.get("total_tasks", -1)))

    data_files = _parquet_files(dataset_root, "data")
    _validate_data_shards(episode_rows, root=dataset_root / "data", paths=data_files)
    data_shard_uncompressed_bytes = {
        path: _parquet_uncompressed_bytes(path) for path in data_files
    }
    data_limit = int(config.target.data_shard_size_mb * 1024**2 * SHARD_SIZE_TOLERANCE)
    oversized_data = [
        path for path, size in data_shard_uncompressed_bytes.items() if size > data_limit
    ]
    if oversized_data:
        raise ValueError(f"data shard exceeds the configured target by more than 10%: {oversized_data[0]}")
    data = _read_parquet(
        data_files,
        columns=[*INDEX_FEATURES, *NUMERIC_KEYS],
    )
    if data.num_rows != total_frames:
        raise ValueError("data Parquet row count differs from info.total_frames")
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for complete dataset validation") from exc
    if not pa.types.is_float32(data.schema.field("timestamp").type) or data["timestamp"].null_count:
        raise ValueError("timestamp Arrow storage must be float32")
    for key in INDEX_FEATURES[1:]:
        if not pa.types.is_int64(data.schema.field(key).type) or data[key].null_count:
            raise ValueError(f"{key} Arrow storage must be non-null int64")
    episode_index = np.asarray(data["episode_index"].to_numpy(), dtype=np.int64)
    frame_index = np.asarray(data["frame_index"].to_numpy(), dtype=np.int64)
    timestamp = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
    index = np.asarray(data["index"].to_numpy(), dtype=np.int64)
    task_index = np.asarray(data["task_index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(index, np.arange(total_frames, dtype=np.int64)):
        raise ValueError("data index must be globally contiguous from zero")
    if not np.array_equal(task_index, np.zeros(total_frames, dtype=np.int64)):
        raise ValueError("every data row must use the single flip-table task index")
    cursor = 0
    for expected_episode, length in enumerate(lengths):
        segment = slice(cursor, cursor + length)
        if not np.array_equal(episode_index[segment], np.full(length, expected_episode)):
            raise ValueError("data episode_index values are not contiguous")
        expected_frames = np.arange(length)
        if not np.array_equal(frame_index[segment], expected_frames):
            raise ValueError("frame_index must restart at zero and remain contiguous per episode")
        if not np.allclose(timestamp[segment], expected_frames / 30.0, atol=1.0e-4):
            raise ValueError("timestamps are not synchronized to frame_index at 30 fps")
        cursor += length
    numeric_values: dict[str, np.ndarray] = {}
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        values = _numeric_array(data[key], key=key, width=width)
        if not np.isfinite(values).all():
            raise ValueError(f"{key} contains NaN or Inf")
        numeric_values[key] = values

    lineage = _load_lineage(dataset_root)
    if len(lineage) != total_episodes:
        raise ValueError("lineage sidecar count differs from total episodes")
    if [record.get("episode_index") for record in lineage] != list(range(total_episodes)):
        raise ValueError("lineage episode indices must be contiguous and ordered")
    split_by_lineage: dict[str, set[str]] = defaultdict(set)
    real_source_indices: list[int] = []
    candidate_contracts: dict[str, tuple[str, tuple[int, ...], str]] = {}
    candidate_variants_seen: set[tuple[str, int]] = set()
    for output_episode_index, record in enumerate(lineage):
        split = record.get("split")
        line = record.get("source_trajectory_lineage")
        if split not in {"train", "validation", "test"} or not isinstance(line, str) or not line:
            raise ValueError("lineage records require a valid split and non-empty lineage")
        if split != episode_splits[output_episode_index]:
            raise ValueError("lineage split differs from meta/info.json episode ranges")
        split_by_lineage[line].add(split)
        if record.get("source_repo_id") != config.source.repo_id:
            raise ValueError("lineage source repository differs from the pinned source")
        if record.get("source_revision") != config.source.revision:
            raise ValueError("lineage source revision differs from the pinned source")
        source_indices = record.get("source_episode_indices")
        if (
            not isinstance(source_indices, list)
            or not source_indices
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source_indices)
            or source_indices != sorted(set(source_indices))
            or any(value < 0 or value >= config.source.episodes for value in source_indices)
        ):
            raise ValueError("lineage source episode indices are invalid")
        if record.get("kind") == "real":
            if len(source_indices) != 1:
                raise ValueError("a real output episode must map to exactly one source episode")
            real_source_indices.append(source_indices[0])
            if record.get("selected_features_copied_without_numeric_transform") is not True:
                raise ValueError("real lineage must declare lossless numeric feature copying")
        elif record.get("kind") == "synthetic":
            report = record.get("success_report")
            if not isinstance(report, dict) or report.get("accepted") is not True:
                raise ValueError("synthetic lineage lacks accepted strict success evidence")
            if report.get("strict_v1_contract") is not True:
                raise ValueError("synthetic lineage did not use the strict V1 contract")
            if report.get("rejection_reasons") not in ([], ()):
                raise ValueError("accepted synthetic lineage contains rejection reasons")
            action_fk_report = report.get("action_fk_report")
            if (
                not isinstance(action_fk_report, dict)
                or action_fk_report.get("schema_version")
                != SYNTHETIC_ACTION_FK_SCHEMA_VERSION
                or action_fk_report.get("pass") is not True
            ):
                raise ValueError("synthetic lineage lacks a passing action FK audit")
            candidate_id = record.get("candidate_id")
            variant = record.get("appearance_variant")
            trajectory_sha256 = record.get("trajectory_sha256")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError("synthetic lineage requires a candidate_id")
            if isinstance(variant, bool) or not isinstance(variant, int) or variant < 0:
                raise ValueError("synthetic lineage requires a non-negative appearance variant")
            if not _is_sha256(trajectory_sha256):
                raise ValueError("synthetic lineage requires a physical trajectory hash")
            identity = (candidate_id, variant)
            if identity in candidate_variants_seen:
                raise ValueError("synthetic candidate/appearance identity is duplicated")
            candidate_variants_seen.add(identity)
            contract = (line, tuple(source_indices), trajectory_sha256)
            existing = candidate_contracts.setdefault(candidate_id, contract)
            if existing != contract:
                raise ValueError("appearance variants changed candidate lineage or trajectory")
            if record.get("config_sha256") != config.digest:
                raise ValueError("synthetic lineage config hash differs from active config")
        else:
            raise ValueError("lineage kind must be real or synthetic")
    leaked = sorted(line for line, splits in split_by_lineage.items() if len(splits) != 1)
    if leaked:
        raise ValueError(f"source trajectory lineages leak across splits: {leaked[:10]}")

    real = [record for record in lineage if record.get("kind") == "real"]
    synthetic = [record for record in lineage if record.get("kind") == "synthetic"]
    if len(real) + len(synthetic) != total_episodes:
        raise ValueError("lineage kind must be real or synthetic")
    if require_full_source and len(real) != config.source.episodes:
        raise ValueError("final dataset does not contain every pinned real source episode")
    if len(real_source_indices) != len(set(real_source_indices)):
        raise ValueError("one source real episode appears more than once")
    if require_full_source and sorted(real_source_indices) != list(range(config.source.episodes)):
        raise ValueError("final dataset does not contain the complete pinned real episode index set")
    candidate_variants = Counter(record["candidate_id"] for record in synthetic)
    if len(candidate_variants) < minimum_synthetic_trajectories:
        raise ValueError("dataset has too few distinct successful synthetic trajectories")
    deficient = [key for key, count in candidate_variants.items() if count < minimum_appearance_variants]
    if deficient:
        raise ValueError("one or more synthetic trajectories has too few appearance variants")
    variants_by_candidate: dict[str, set[int]] = defaultdict(set)
    candidates_by_trajectory: dict[str, set[str]] = defaultdict(set)
    for record in synthetic:
        variants_by_candidate[record["candidate_id"]].add(record["appearance_variant"])
        candidates_by_trajectory[record["trajectory_sha256"]].add(record["candidate_id"])
    noncontiguous_variants = [
        candidate_id
        for candidate_id, variants in variants_by_candidate.items()
        if variants != set(range(max(variants) + 1))
    ]
    if noncontiguous_variants:
        raise ValueError("appearance variants must be contiguous from zero for every candidate")
    duplicated_trajectories = [
        trajectory_sha256
        for trajectory_sha256, candidate_ids in candidates_by_trajectory.items()
        if len(candidate_ids) != 1
    ]
    if duplicated_trajectories:
        raise ValueError("multiple candidates duplicate the same physical trajectory")

    video_report: dict[str, Any] = {}
    for key in EXPECTED_CAMERA_KEYS:
        paths = tuple(sorted((dataset_root / "videos" / key).glob("chunk-*/*.mp4")))
        if not paths:
            raise ValueError(f"no video shards found for {key}")
        probes_by_path = {
            path: _probe_video(path, full_decode=full_video_decode) for path in paths
        }
        probes = list(probes_by_path.values())
        _validate_video_slices(
            episode_rows,
            key=key,
            root=dataset_root / "videos" / key,
            probes_by_path=probes_by_path,
        )
        decoded_frames = sum(int(item["frames"]) for item in probes)
        if decoded_frames != total_frames:
            raise ValueError(f"{key} decoded frame count {decoded_frames} differs from {total_frames}")
        video_limit = int(config.target.video_shard_size_mb * 1024**2 * SHARD_SIZE_TOLERANCE)
        oversized_video = [path for path in paths if path.stat().st_size > video_limit]
        if oversized_video:
            raise ValueError(
                f"{key} shard exceeds the configured target by more than 10%: {oversized_video[0]}"
            )
        video_report[key] = {
            "files": len(paths),
            "decoded_frames": decoded_frames,
            "largest_shard_bytes": max(path.stat().st_size for path in paths),
        }

    stats = _json_object(dataset_root / "meta" / "stats.json")
    stats_numeric_values = {
        **numeric_values,
        "timestamp": timestamp.reshape(-1, 1),
        "frame_index": frame_index.astype(np.float64).reshape(-1, 1),
        "episode_index": episode_index.astype(np.float64).reshape(-1, 1),
        "index": index.astype(np.float64).reshape(-1, 1),
        "task_index": task_index.astype(np.float64).reshape(-1, 1),
    }
    image_sample_count = _expected_image_stat_count(lengths)
    _validate_stats(
        stats,
        numeric_values=stats_numeric_values,
        total_frames=total_frames,
        image_sample_count=image_sample_count,
    )
    stats_report = _json_object(
        dataset_root / "meta" / "augmentation" / "stats-recalculation.json"
    )
    expected_stats_report = {
        "schema_version": STATS_REPORT_SCHEMA_VERSION,
        "pipeline_config_sha256": config.digest,
        "lerobot_version": "0.6.0",
        "numeric_rows": total_frames,
        "episodes": total_episodes,
        "image_sample_count_per_camera": image_sample_count,
        "stats_sha256": sha256_file(dataset_root / "meta" / "stats.json"),
    }
    for key, expected in expected_stats_report.items():
        if stats_report.get(key) != expected:
            raise ValueError(f"stats recalculation report field {key} is inconsistent")
    camera_stats_report = stats_report.get("cameras")
    if not isinstance(camera_stats_report, dict) or set(camera_stats_report) != set(EXPECTED_CAMERA_KEYS):
        raise ValueError("stats recalculation report camera set is inconsistent")
    for key in EXPECTED_CAMERA_KEYS:
        if camera_stats_report[key].get("sampled_frames") != image_sample_count:
            raise ValueError(f"stats recalculation report sample count is inconsistent for {key}")
    file_manifest = verify_file_manifest(dataset_root)
    summary = _json_object(dataset_root / "meta" / "augmentation" / "summary.json")
    if summary.get("real_episodes") != len(real) or summary.get("synthetic_episodes") != len(synthetic):
        raise ValueError("augmentation summary counts differ from lineage sidecars")
    if summary.get("pipeline_config_sha256") != config.digest:
        raise ValueError("augmentation summary config hash differs from active config")
    if summary.get("successful_physical_trajectories") != len(candidate_variants):
        raise ValueError("augmentation summary physical trajectory count is inconsistent")
    if summary.get("appearance_variants") != len(synthetic):
        raise ValueError("augmentation summary appearance variant count is inconsistent")
    if summary.get("split_counts") != dict(Counter(episode_splits)):
        raise ValueError("augmentation summary split counts differ from meta/info.json")
    return {
        "valid": True,
        "root": str(dataset_root),
        "episodes": total_episodes,
        "frames": total_frames,
        "real_episodes": len(real),
        "synthetic_episodes": len(synthetic),
        "successful_physical_trajectories": len(candidate_variants),
        "split_counts": dict(Counter(record["split"] for record in lineage)),
        "data_files": len(data_files),
        "largest_data_shard_uncompressed_bytes": max(data_shard_uncompressed_bytes.values()),
        "video": video_report,
        "file_count": file_manifest["file_count"],
        "total_bytes": file_manifest["total_bytes"],
        "full_video_decode": full_video_decode,
    }
