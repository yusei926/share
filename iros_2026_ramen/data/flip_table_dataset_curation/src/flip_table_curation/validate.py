from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .build import MANIFEST_NAME, MANIFEST_SCHEMA, OWNED_MARKER, dataset_root
from .config import CurationConfig
from .source import NUMERIC_WIDTHS, VIDEO_KEYS, download_source, episode_slice, read_numeric_table
from .util import atomic_write_json, sha256_file


def _probe(path: Path) -> dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command))["streams"][0]


def validate_dataset_root(root: Path, config: CurationConfig, *, require_source_comparison: bool = True, minimum_episodes: int | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    errors: list[str] = []
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unsupported manifest schema")
    if manifest.get("config_sha256") != config.digest or manifest.get("code_sha256") != config.code_digest:
        errors.append("manifest config/code hash mismatch")
    expected_files = manifest.get("file_sha256", {})
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in {MANIFEST_NAME, OWNED_MARKER}}
    if actual_files != set(expected_files):
        errors.append("dataset file inventory differs from manifest")
    for relative, expected_hash in expected_files.items():
        if not (root / relative).is_file() or sha256_file(root / relative) != expected_hash:
            errors.append(f"file hash mismatch: {relative}")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episode_rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    minimum = minimum_episodes if minimum_episodes is not None else int(config.section("target")["minimum_episodes"])
    if len(episode_rows) < minimum or len(episode_rows) != int(info["total_episodes"]):
        errors.append("episode count mismatch")
    if [int(row["episode_index"]) for row in episode_rows] != list(range(len(episode_rows))):
        errors.append("episode indices are not contiguous")
    if any(row.get("curation_verdict") not in {"optimal", "success"} for row in episode_rows):
        errors.append("non-accepted curation verdict is present")
    split_sources: dict[str, set[str]] = defaultdict(set)
    for row in episode_rows:
        split_sources[str(row["curation_split"])].add(str(row["source_episode_name"]))
        if int(row["source_frame_end"]) - int(row["source_frame_start"]) != int(row["length"]):
            errors.append(f"invalid source interval: episode={row['episode_index']}")
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_sources[first] & split_sources[second]:
            errors.append(f"source lineage leakage: {first}/{second}")
    tables = [pq.read_table(path) for path in sorted((root / "data").glob("chunk-*/*.parquet"))]
    numeric = pa.concat_tables(tables) if tables else pa.table({})
    if len(numeric) != int(info["total_frames"]):
        errors.append("numeric frame count mismatch")
    if len(numeric):
        if not np.array_equal(np.asarray(numeric["index"].to_numpy()), np.arange(len(numeric))):
            errors.append("global numeric index is not contiguous")
        for key, width in NUMERIC_WIDTHS.items():
            values = np.asarray(numeric[key].to_pylist())
            if values.shape != (len(numeric), width) or not np.isfinite(values).all():
                errors.append(f"invalid numeric feature: {key}")
    expected_video_frames: dict[tuple[str, int], int] = defaultdict(int)
    for row in episode_rows:
        for key in VIDEO_KEYS:
            file_index = int(row[f"videos/{key}/file_index"])
            expected_video_frames[(key, file_index)] += int(row["length"])
            duration = float(row[f"videos/{key}/to_timestamp"]) - float(row[f"videos/{key}/from_timestamp"])
            if abs(duration * int(info["fps"]) - int(row["length"])) > 0.01:
                errors.append(f"video duration mismatch: episode={row['episode_index']} key={key}")
    max_bytes = int(config.section("video")["max_file_size_mb"]) * 1024 * 1024
    def check_video(item: tuple[str, int, int]) -> tuple[str, str | None]:
        key, file_index, expected_frames = item
        path = root / "videos" / key / "chunk-000" / f"file-{file_index:03d}.mp4"
        try:
            stream = _probe(path)
            numerator, denominator = (int(value) for value in stream["r_frame_rate"].split("/"))
            if int(stream["width"]) != 640 or int(stream["height"]) != 480 or abs(numerator / denominator - int(info["fps"])) > 1e-6:
                return f"{key}/{file_index}", "invalid video stream metadata"
            if int(stream["nb_read_frames"]) != expected_frames:
                return f"{key}/{file_index}", f"frame mismatch: {key}/file-{file_index:03d}: {stream['nb_read_frames']} != {expected_frames}"
            if path.stat().st_size > max_bytes:
                return f"{key}/{file_index}", "video file exceeds configured size cap"
            return f"{key}/{file_index}", None
        except Exception as error:
            return f"{key}/{file_index}", f"video validation failed: {error}"
    video_errors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for _, error in executor.map(check_video, [(key, index, count) for (key, index), count in expected_video_frames.items()]):
            if error:
                video_errors.append(error)
    errors.extend(video_errors)
    if require_source_comparison:
        snapshot = download_source(config, include_data=True, include_videos=False)
        source_table = read_numeric_table(snapshot)
        source_rows = {int(row["episode_index"]): row for row in snapshot.episodes}
        for row in episode_rows:
            source = episode_slice(source_table, source_rows[int(row["source_episode_index"])]).slice(int(row["source_frame_start"]), int(row["length"]))
            output = pq.read_table(root / "data" / "chunk-000" / f"file-{int(row['data/file_index']):03d}.parquet")
            for key in NUMERIC_WIDTHS:
                if source[key].to_pylist() != output[key].to_pylist():
                    errors.append(f"source numeric values changed: episode={row['episode_index']} key={key}")
                    break
    report = {"schema_version": "team_ramen_manual_flip_table_validation/v1", "dataset_root": str(root), "episodes": len(episode_rows), "frames": int(info["total_frames"]), "video_files": len(expected_video_frames), "errors": errors, "passed": not errors}
    if errors:
        raise RuntimeError(f"dataset validation failed: {errors[:5]}")
    return report


def validate_local(config: CurationConfig, *, minimum_episodes: int | None = None) -> dict[str, Any]:
    report = validate_dataset_root(dataset_root(config), config, minimum_episodes=minimum_episodes)
    atomic_write_json(config.workspace / "validation" / "validation.json", report)
    return report
