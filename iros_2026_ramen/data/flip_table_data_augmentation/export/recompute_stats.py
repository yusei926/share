"""Deterministically recompute LeRobot 0.6.0 statistics for the final dataset."""

from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EXPECTED_CAMERA_KEYS, PipelineConfig
from ..io_utils import atomic_write_json, sha256_file
from ..source_contract import INDEX_FEATURES, NUMERIC_FEATURES


STATS_REPORT_SCHEMA_VERSION = "team_ramen_flip_table_stats_recalculation/v1"
LEROBOT_VERSION = "0.6.0"


def image_sample_indices(length: int) -> np.ndarray:
    """Return the exact deterministic indices used by LeRobot 0.6.0."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("episode length must be a positive integer")
    minimum = min(length, 100)
    count = max(minimum, min(int(length**0.75), 10_000))
    return np.round(np.linspace(0, length - 1, count)).astype(np.int64)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to recompute dataset statistics") from exc
    paths = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise ValueError("dataset contains no episode metadata")
    table = pa.concat_tables([pq.read_table(path) for path in paths], promote_options="default")
    rows = table.to_pylist()
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError("episode metadata must be contiguous before statistics are recomputed")
    return rows


def _arrow_list_array(column: Any, *, key: str, width: int) -> np.ndarray:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to recompute dataset statistics") from exc
    value = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    arrow_type = value.type
    if not (
        pa.types.is_list(arrow_type)
        or pa.types.is_large_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        raise ValueError(f"{key} must use Arrow list storage")
    if value.null_count:
        raise ValueError(f"{key} contains null rows")
    if pa.types.is_fixed_size_list(arrow_type):
        if arrow_type.list_size != width:
            raise ValueError(f"{key} list width differs from {width}")
    else:
        offsets = np.asarray(value.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
        if not np.all(np.diff(offsets) == width):
            raise ValueError(f"{key} contains a row whose width differs from {width}")
    values = np.asarray(value.values.to_numpy(zero_copy_only=False), dtype=np.float64)
    return values.reshape(len(value), width)


def _numeric_stats(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    try:
        import pyarrow.parquet as pq
        from lerobot.datasets.compute_stats import RunningQuantileStats
    except ImportError as exc:
        raise RuntimeError("LeRobot 0.6.0 and pyarrow are required to recompute statistics") from exc
    keys = (*NUMERIC_FEATURES, *INDEX_FEATURES)
    runners = {key: RunningQuantileStats() for key in keys}
    total_rows = 0
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65_536, columns=list(keys)):
            total_rows += batch.num_rows
            for key, (_dtype, width) in NUMERIC_FEATURES.items():
                values = _arrow_list_array(
                    batch.column(batch.schema.get_field_index(key)), key=key, width=width
                )
                if not np.isfinite(values).all():
                    raise ValueError(f"{key} contains NaN or Inf")
                runners[key].update(values)
            for key in INDEX_FEATURES:
                values = np.asarray(
                    batch.column(batch.schema.get_field_index(key)).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                ).reshape(-1, 1)
                if not np.isfinite(values).all():
                    raise ValueError(f"{key} contains NaN or Inf")
                runners[key].update(values)
    if total_rows < 2:
        raise ValueError("dataset needs at least two rows for statistics")
    result = {key: runner.get_statistics() for key, runner in runners.items()}
    if any(int(feature["count"][0]) != total_rows for feature in result.values()):
        raise RuntimeError("numeric statistics did not consume every row")
    return result, total_rows


def _camera_sample_plan(
    root: Path, rows: list[dict[str, Any]], key: str
) -> tuple[dict[Path, set[int]], int]:
    plans: dict[Path, set[int]] = {}
    expected_count = 0
    prefix = f"videos/{key}"
    for row in rows:
        length = int(row["length"])
        local_indices = image_sample_indices(length)
        chunk = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        path = root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        from_frame = round(float(row[f"{prefix}/from_timestamp"]) * 30.0)
        selected = plans.setdefault(path, set())
        before = len(selected)
        selected.update((from_frame + local_indices).tolist())
        if len(selected) - before != len(local_indices):
            raise ValueError(f"{key} sampled frame ranges overlap")
        expected_count += len(local_indices)
    return plans, expected_count


def _camera_stats(
    root: Path, rows: list[dict[str, Any]], key: str
) -> tuple[dict[str, np.ndarray], int, int]:
    try:
        import av
        from lerobot.datasets.compute_stats import RunningQuantileStats
    except ImportError as exc:
        raise RuntimeError("PyAV and LeRobot 0.6.0 are required to recompute RGB statistics") from exc
    plans, expected_count = _camera_sample_plan(root, rows, key)
    runner = RunningQuantileStats()
    sampled_count = 0
    decoded_count = 0
    for path, selected in sorted(plans.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        pending: list[np.ndarray] = []
        last_frame_index = -1
        with av.open(str(path), mode="r") as container:
            if len(container.streams.video) != 1 or container.streams.audio:
                raise ValueError(f"{path} must contain one video stream and no audio")
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                last_frame_index = frame_index
                decoded_count += 1
                if frame_index not in selected:
                    continue
                image = frame.to_ndarray(format="rgb24")
                if image.shape != (480, 640, 3):
                    raise ValueError(f"{path} decoded an RGB frame with shape {image.shape}")
                pending.append(image[::4, ::4, :].reshape(-1, 3))
                sampled_count += 1
                if len(pending) == 128:
                    runner.update(np.concatenate(pending, axis=0))
                    pending.clear()
            if pending:
                runner.update(np.concatenate(pending, axis=0))
        if selected and max(selected) >= last_frame_index + 1:
            raise ValueError(f"{path} ended before every selected RGB frame")
    if sampled_count != expected_count:
        raise RuntimeError(f"{key} sampled {sampled_count} frames, expected {expected_count}")
    raw = runner.get_statistics()
    result = {
        name: (
            np.asarray([sampled_count], dtype=np.int64)
            if name == "count"
            else np.asarray(value / 255.0, dtype=np.float64).reshape(3, 1, 1)
        )
        for name, value in raw.items()
    }
    return result, sampled_count, decoded_count


def recompute_dataset_stats(
    dataset_root: str | Path, config: PipelineConfig
) -> dict[str, Any]:
    """Overwrite inherited stats with deterministic final-dataset statistics."""

    if version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"statistics must be computed with lerobot=={LEROBOT_VERSION}")
    root = Path(dataset_root).expanduser().resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    rows = _episode_rows(root)
    numeric, total_rows = _numeric_stats(root)
    if total_rows != int(info.get("total_frames", -1)):
        raise ValueError("numeric statistics row count differs from info.total_frames")
    camera_reports: dict[str, Any] = {}
    stats: dict[str, dict[str, np.ndarray]] = dict(numeric)
    expected_image_count: int | None = None
    for key in EXPECTED_CAMERA_KEYS:
        feature_stats, sampled, decoded = _camera_stats(root, rows, key)
        if expected_image_count is None:
            expected_image_count = sampled
        elif sampled != expected_image_count:
            raise RuntimeError("policy cameras did not use the same deterministic frame sample")
        stats[key] = feature_stats
        camera_reports[key] = {"sampled_frames": sampled, "decoded_frames": decoded}

    stats_path = root / "meta" / "stats.json"
    atomic_write_json(stats_path, _json_ready(stats))
    report = {
        "schema_version": STATS_REPORT_SCHEMA_VERSION,
        "pipeline_config_sha256": config.digest,
        "lerobot_version": LEROBOT_VERSION,
        "algorithm": "full_numeric_and_lerobot_060_deterministic_rgb_sampling",
        "numeric_rows": total_rows,
        "episodes": len(rows),
        "image_sample_count_per_camera": expected_image_count,
        "cameras": camera_reports,
        "stats_sha256": sha256_file(stats_path),
    }
    atomic_write_json(root / "meta" / "augmentation" / "stats-recalculation.json", report)
    return report
