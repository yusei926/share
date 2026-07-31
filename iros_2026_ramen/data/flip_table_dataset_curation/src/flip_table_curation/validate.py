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
from .source import INDEX_KEYS, NUMERIC_WIDTHS, VIDEO_KEYS, download_source, episode_slice, read_numeric_table
from .util import atomic_write_json, sha256_file


def _probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.check_output(command))["streams"][0]


def validate_dataset_root(
    root: Path,
    config: CurationConfig,
    *,
    require_source_comparison: bool = True,
    minimum_episodes: int | None = None,
) -> dict[str, Any]:
    root = Path(root).resolve()
    errors: list[str] = []
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unsupported manifest schema")
    if manifest.get("config_sha256") != config.digest:
        errors.append("config hash mismatch")
    if manifest.get("code_sha256") != config.code_digest:
        errors.append("code hash mismatch")
    expected_files = manifest.get("file_sha256", {})
    for relative, expected in expected_files.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"file hash mismatch: {relative}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {MANIFEST_NAME, OWNED_MARKER}
    }
    if actual_files != set(expected_files):
        errors.append("dataset file inventory differs from manifest")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episode_rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        episode_rows.extend(pq.read_table(path).to_pylist())
    episode_rows.sort(key=lambda row: int(row["episode_index"]))
    minimum = (
        int(config.section("target")["minimum_episodes"])
        if minimum_episodes is None
        else minimum_episodes
    )
    if len(episode_rows) < minimum:
        errors.append(f"episode count {len(episode_rows)} is below {minimum}")
    if len(episode_rows) != int(info["total_episodes"]):
        errors.append("info episode count mismatch")
    if [int(row["episode_index"]) for row in episode_rows] != list(
        range(len(episode_rows))
    ):
        errors.append("episode indices are not contiguous")
    if any(row.get("curation_status") != "accepted_auto" for row in episode_rows):
        errors.append("non-accepted episode is present")
    if any(int(row.get("curation_step_count", -1)) != 0 for row in episode_rows):
        errors.append("walking episode is present")
    split_sources: dict[str, set[str]] = defaultdict(set)
    for row in episode_rows:
        split_sources[str(row["curation_split"])].add(str(row["source_episode_name"]))
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_sources[first] & split_sources[second]:
            errors.append(f"source leakage between {first} and {second}")

    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    tables = [pq.read_table(path) for path in data_files]
    table = pa.concat_tables(tables) if tables else pa.table({})
    if len(table) != int(info["total_frames"]):
        errors.append("numeric frame count mismatch")
    if len(table):
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        if not np.array_equal(indices, np.arange(len(table), dtype=np.int64)):
            errors.append("global index is not contiguous")
        for key, width in NUMERIC_WIDTHS.items():
            values = np.asarray(table[key].to_pylist())
            if values.shape != (len(table), width) or not np.isfinite(values).all():
                errors.append(f"invalid numeric feature {key}")

    if require_source_comparison:
        source = download_source(config, include_videos=False)
        source_table = read_numeric_table(source)
        source_rows = {int(row["episode_index"]): row for row in source.episodes}
        source_intervals: dict[
            tuple[str, int], list[tuple[int, int, int]]
        ] = defaultdict(list)
        output_by_episode = {
            int(row["episode_index"]): pq.read_table(
                root
                / "data"
                / f"chunk-{int(row['data/chunk_index']):03d}"
                / f"file-{int(row['data/file_index']):03d}.parquet"
            )
            for row in episode_rows
        }
        for row in episode_rows:
            source_episode = int(row["source_episode_index"])
            start = int(row["source_frame_start"])
            length = int(row["length"])
            source_slice = episode_slice(source_table, source_rows[source_episode]).slice(
                start, length
            )
            output = output_by_episode[int(row["episode_index"])]
            source_row = source_rows[source_episode]
            for key in VIDEO_KEYS:
                file_index = int(source_row[f"videos/{key}/file_index"])
                video_start = int(
                    round(
                        float(source_row[f"videos/{key}/from_timestamp"])
                        * int(info["fps"])
                    )
                ) + start
                source_intervals[(key, file_index)].append(
                    (
                        video_start,
                        video_start + length,
                        int(row["episode_index"]),
                    )
                )
            for key in NUMERIC_WIDTHS:
                if source_slice[key].to_pylist() != output[key].to_pylist():
                    errors.append(
                        f"source numeric values changed: episode={row['episode_index']} key={key}"
                    )
                    break
        for (key, file_index), intervals in source_intervals.items():
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:], strict=False):
                if current[0] < previous[1]:
                    errors.append(
                        "source video frame reused: "
                        f"{key}/file-{file_index:03d} "
                        f"episodes={previous[2]},{current[2]}"
                    )

    expected_video_frames: dict[tuple[str, int], int] = defaultdict(int)
    for row in episode_rows:
        for key in VIDEO_KEYS:
            file_index = int(row[f"videos/{key}/file_index"])
            expected_video_frames[(key, file_index)] += int(row["length"])
            duration = (
                float(row[f"videos/{key}/to_timestamp"])
                - float(row[f"videos/{key}/from_timestamp"])
            )
            if abs(duration * int(info["fps"]) - int(row["length"])) > 0.01:
                errors.append(
                    f"episode video duration mismatch: ep={row['episode_index']} key={key}"
                )
    video_probe: dict[str, Any] = {}
    video_jobs = [
        (
            key,
            file_index,
            expected,
            root / "videos" / key / "chunk-000" / f"file-{file_index:03d}.mp4",
        )
        for (key, file_index), expected in sorted(expected_video_frames.items())
    ]
    with ThreadPoolExecutor(max_workers=min(8, len(video_jobs))) as executor:
        probes = list(executor.map(lambda job: _probe(job[3]), video_jobs))
    for (key, file_index, expected, path), probe in zip(
        video_jobs, probes, strict=True
    ):
        frames = int(probe["nb_read_frames"])
        video_probe[f"{key}/file-{file_index:03d}"] = probe
        if frames != expected:
            errors.append(f"video frames {frames}!={expected}: {path}")
        if probe["width"] != 640 or probe["height"] != 480:
            errors.append(f"video dimensions are invalid: {path}")
        numerator, denominator = (int(value) for value in probe["r_frame_rate"].split("/"))
        if abs(numerator / denominator - int(info["fps"])) > 1e-6:
            errors.append(f"video fps is invalid: {path}")
    report = {
        "schema_version": "team_ramen_flip_table_curated_validation/v1",
        "root": str(root),
        "episodes": len(episode_rows),
        "frames": len(table),
        "split_counts": {key: len(value) for key, value in split_sources.items()},
        "video_files": len(video_probe),
        "errors": errors,
        "passed": not errors,
    }
    if errors:
        raise RuntimeError(
            f"dataset validation failed with {len(errors)} errors: "
            + " | ".join(errors[:20])
        )
    return report


def validate_local(config: CurationConfig, *, minimum_episodes: int | None = None) -> dict:
    report = validate_dataset_root(
        dataset_root(config),
        config,
        minimum_episodes=minimum_episodes,
    )
    output = config.workspace / "validation" / "validation.json"
    atomic_write_json(output, report)
    print(f"[validate] passed report={output}")
    return report
