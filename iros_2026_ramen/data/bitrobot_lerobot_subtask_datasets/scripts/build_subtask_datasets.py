from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "subtasks.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "datasets"
NS_PER_SEC = 1_000_000_000.0
VIDEO_DT_TOLERANCE_SEC = 1.0 / 30.0 + 1e-6
BYTES_PER_MB = 1024 * 1024
DEFAULT_CHUNKS_SIZE = 1_000
DEFAULT_DATA_FILE_SIZE_MB = 100
DEFAULT_VIDEO_FILE_SIZE_MB = 500


@dataclass(frozen=True)
class SubtaskSpec:
    name: str
    task: str
    labels: tuple[str, ...] = ()
    split_source_label: str | None = None
    split_start_fraction: float = 0.0
    split_end_fraction: float = 1.0


@dataclass(frozen=True)
class Segment:
    subtask: str
    task: str
    source_episode_index: int
    source_episode_name: str
    start_sec: float
    end_sec: float
    source_task: str

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass(frozen=True)
class FileLocation:
    chunk_index: int
    file_index: int


class StatsAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum: list[float] | None = None
        self.sumsq: list[float] | None = None
        self.min: list[float] | None = None
        self.max: list[float] | None = None

    def update(self, value: Any) -> None:
        values = _flatten_numeric(value)
        if not values:
            return
        if self.sum is None:
            self.sum = [0.0] * len(values)
            self.sumsq = [0.0] * len(values)
            self.min = [math.inf] * len(values)
            self.max = [-math.inf] * len(values)
        if len(values) != len(self.sum):
            raise ValueError(f"stats dimension changed from {len(self.sum)} to {len(values)}")
        self.count += 1
        assert self.sumsq is not None
        assert self.min is not None
        assert self.max is not None
        for index, item in enumerate(values):
            self.sum[index] += item
            self.sumsq[index] += item * item
            self.min[index] = min(self.min[index], item)
            self.max[index] = max(self.max[index], item)

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0 or self.sum is None or self.sumsq is None or self.min is None or self.max is None:
            return {"min": [], "max": [], "mean": [], "std": []}
        mean = [item / self.count for item in self.sum]
        variance = [
            max(self.sumsq[index] / self.count - mean[index] * mean[index], 0.0)
            for index in range(len(mean))
        ]
        return {
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": [math.sqrt(item) for item in variance],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subtasks", nargs="*", default=None)
    parser.add_argument("--source-repo-id", default=None)
    parser.add_argument("--raw-repo-id", default=None)
    parser.add_argument("--target-repo-template", default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--data-files-size-mb", type=int, default=DEFAULT_DATA_FILE_SIZE_MB)
    parser.add_argument("--video-files-size-mb", type=int, default=DEFAULT_VIDEO_FILE_SIZE_MB)
    parser.add_argument("--episodes-per-meta-file", type=int, default=1_000)
    parser.add_argument("--video-keys", nargs="*", default=None)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--video-mode", choices=("copy", "reencode"), default="copy")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    source_repo_id = args.source_repo_id or config["source_repo_id"]
    raw_repo_id = args.raw_repo_id or config["raw_repo_id"]
    target_repo_template = args.target_repo_template or config["target_repo_template"]
    specs = load_subtask_specs(config)
    selected_names = args.subtasks or [spec.name for spec in specs]
    selected = [spec for spec in specs if spec.name in set(selected_names)]
    missing = sorted(set(selected_names) - {spec.name for spec in specs})
    if missing:
        raise ValueError(f"unknown subtask(s): {', '.join(missing)}")

    api = HfApi()
    source_info = load_source_info(source_repo_id)
    source_stats = load_source_stats(source_repo_id)
    raw_info_files = list_raw_info_files(api, raw_repo_id)
    episode_rows = load_episode_rows(source_repo_id)
    if len(raw_info_files) != int(source_info["total_episodes"]) or len(episode_rows) != int(source_info["total_episodes"]):
        raise RuntimeError(
            "source/raw episode count mismatch: "
            f"raw_info={len(raw_info_files)} source_info={source_info['total_episodes']} episode_rows={len(episode_rows)}"
        )
    if args.max_episodes:
        raw_info_files = raw_info_files[: args.max_episodes]
        episode_rows = episode_rows[: args.max_episodes]

    segments_by_subtask = plan_segments(
        raw_repo_id=raw_repo_id,
        raw_info_files=raw_info_files,
        episode_rows=episode_rows,
        specs=selected,
        min_duration_sec=float(config.get("min_duration_sec", 0.5)),
    )
    for spec in selected:
        segments = segments_by_subtask.get(spec.name, [])
        target_repo_id = target_repo_template.format(subtask=spec.name)
        output_dir = args.output_root / target_repo_id.replace("/", "__")
        print(f"{spec.name}: {len(segments)} segments -> {target_repo_id}")
        if args.dry_run:
            continue
        build_dataset(
            api=api,
            source_repo_id=source_repo_id,
            raw_repo_id=raw_repo_id,
            source_info=source_info,
            source_stats=source_stats,
            target_repo_id=target_repo_id,
            output_dir=output_dir,
            spec=spec,
            segments=segments,
            episode_rows=episode_rows,
            selected_video_keys=args.video_keys,
            skip_videos=args.skip_videos,
            video_mode=args.video_mode,
            data_files_size_mb=args.data_files_size_mb,
            video_files_size_mb=args.video_files_size_mb,
            episodes_per_meta_file=args.episodes_per_meta_file,
            force=args.force,
        )
        if args.upload:
            api.create_repo(repo_id=target_repo_id, repo_type="dataset", private=args.private, exist_ok=True)
            delete_patterns = ["data/**", "videos/**", "meta/**", "README.md"] if args.force else None
            api.upload_folder(
                repo_id=target_repo_id,
                repo_type="dataset",
                folder_path=output_dir,
                commit_message=f"Upload {spec.name} BitRobot LeRobot v3 subtask dataset",
                delete_patterns=delete_patterns,
            )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_subtask_specs(config: dict[str, Any]) -> list[SubtaskSpec]:
    specs = []
    for item in config["subtasks"]:
        specs.append(
            SubtaskSpec(
                name=str(item["name"]),
                task=str(item["task"]),
                labels=tuple(str(label) for label in item.get("labels", [])),
                split_source_label=item.get("split_source_label"),
                split_start_fraction=float(item.get("split_start_fraction", 0.0)),
                split_end_fraction=float(item.get("split_end_fraction", 1.0)),
            )
        )
    return specs


def load_source_info(source_repo_id: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename="meta/info.json")
    return load_json(Path(path))


def load_source_stats(source_repo_id: str) -> dict[str, Any]:
    path = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename="meta/stats.json")
    return load_json(Path(path))


def list_raw_info_files(api: HfApi, raw_repo_id: str) -> list[str]:
    return sorted(
        item
        for item in api.list_repo_files(repo_id=raw_repo_id, repo_type="dataset")
        if item.endswith("/info.json")
    )


def load_episode_rows(source_repo_id: str) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    api = HfApi()
    rows: list[dict[str, Any]] = []
    for filename in sorted(
        item
        for item in api.list_repo_files(repo_id=source_repo_id, repo_type="dataset")
        if item.startswith("meta/episodes/") and item.endswith(".parquet")
    ):
        path = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename=filename)
        rows.extend(pq.read_table(path).to_pylist())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def plan_segments(
    *,
    raw_repo_id: str,
    raw_info_files: list[str],
    episode_rows: list[dict[str, Any]],
    specs: list[SubtaskSpec],
    min_duration_sec: float,
) -> dict[str, list[Segment]]:
    direct: dict[str, list[SubtaskSpec]] = defaultdict(list)
    split: dict[str, list[SubtaskSpec]] = defaultdict(list)
    for spec in specs:
        for label in spec.labels:
            direct[label].append(spec)
        if spec.split_source_label:
            split[spec.split_source_label].append(spec)

    segments: dict[str, list[Segment]] = defaultdict(list)
    for raw_index, raw_info_file in enumerate(raw_info_files):
        raw_info = load_json(Path(hf_hub_download(repo_id=raw_repo_id, repo_type="dataset", filename=raw_info_file)))
        episode_row = episode_rows[raw_index]
        episode_index = int(episode_row["episode_index"])
        if episode_index != raw_index:
            raise RuntimeError(f"unexpected episode order: row episode_index={episode_index}, raw_index={raw_index}")
        episode_name = str(raw_info["episode_name"])
        episode_start_ns = int(raw_info["start_timestamp_ns"])
        episode_end_ns = int(raw_info["end_timestamp_ns"])
        annotations = sorted(raw_info.get("subtasks", []), key=lambda item: int(item["timestamp_ns"]))
        for index, item in enumerate(annotations):
            source_task = str(item["task"])
            start_ns = int(item["timestamp_ns"])
            end_ns = int(annotations[index + 1]["timestamp_ns"]) if index + 1 < len(annotations) else episode_end_ns
            start_sec = max(0.0, (start_ns - episode_start_ns) / NS_PER_SEC)
            end_sec = max(start_sec, (end_ns - episode_start_ns) / NS_PER_SEC)
            duration = end_sec - start_sec
            for spec in direct.get(source_task, []):
                add_segment(segments, spec, episode_index, episode_name, start_sec, end_sec, source_task, min_duration_sec)
            for spec in split.get(source_task, []):
                split_start = start_sec + duration * spec.split_start_fraction
                split_end = start_sec + duration * spec.split_end_fraction
                add_segment(segments, spec, episode_index, episode_name, split_start, split_end, source_task, min_duration_sec)
    return segments


def add_segment(
    segments: dict[str, list[Segment]],
    spec: SubtaskSpec,
    episode_index: int,
    episode_name: str,
    start_sec: float,
    end_sec: float,
    source_task: str,
    min_duration_sec: float,
) -> None:
    if end_sec - start_sec < min_duration_sec:
        return
    segments[spec.name].append(
        Segment(
            subtask=spec.name,
            task=spec.task,
            source_episode_index=episode_index,
            source_episode_name=episode_name,
            start_sec=start_sec,
            end_sec=end_sec,
            source_task=source_task,
        )
    )


def build_dataset(
    *,
    api: HfApi,
    source_repo_id: str,
    raw_repo_id: str,
    source_info: dict[str, Any],
    source_stats: dict[str, Any],
    target_repo_id: str,
    output_dir: Path,
    spec: SubtaskSpec,
    segments: list[Segment],
    episode_rows: list[dict[str, Any]],
    selected_video_keys: list[str] | None,
    skip_videos: bool,
    video_mode: str,
    data_files_size_mb: int,
    video_files_size_mb: int,
    episodes_per_meta_file: int,
    force: bool,
) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    features = dict(source_info["features"])
    all_video_keys = [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    video_keys = selected_video_keys if selected_video_keys is not None else all_video_keys
    unknown_video_keys = sorted(set(video_keys) - set(all_video_keys))
    if unknown_video_keys:
        raise ValueError(f"unknown video key(s): {', '.join(unknown_video_keys)}")
    if skip_videos:
        video_keys = []
        features = {
            key: value
            for key, value in features.items()
            if not (isinstance(value, dict) and value.get("dtype") == "video")
        }
    else:
        features = {
            key: value
            for key, value in features.items()
            if not (isinstance(value, dict) and value.get("dtype") == "video" and key not in set(video_keys))
        }

    pa, pq = require_pyarrow()
    source_file_rows: dict[str, list[Segment]] = defaultdict(list)
    for segment in segments:
        source_episode = episode_rows[segment.source_episode_index]
        source_data_path = data_path_from_episode(source_episode)
        source_file_rows[source_data_path].append(segment)

    task_rows = [{"task_index": 0, "__index_level_0__": spec.task}]
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(task_rows), output_dir / "meta" / "tasks.parquet")

    stats = {key: StatsAccumulator() for key, feature in features.items() if feature.get("dtype") != "video"}
    episode_meta: list[dict[str, Any]] = []
    data_buffer: list[dict[str, Any]] = []
    data_location = FileLocation(chunk_index=0, file_index=0)
    data_buffer_bytes = 0
    data_target_bytes = data_files_size_mb * BYTES_PER_MB
    global_index = 0
    data_from_index = 0
    fps = float(source_info.get("fps", 30.0))
    chunks_size = int(source_info.get("chunks_size", DEFAULT_CHUNKS_SIZE))

    for source_data_path, grouped_segments in sorted(source_file_rows.items()):
        table_path = hf_hub_download(repo_id=source_repo_id, repo_type="dataset", filename=source_data_path)
        table = pq.read_table(table_path)
        rows = table.to_pylist()
        rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_episode[int(row["episode_index"])].append(row)
        for segment in grouped_segments:
            source_rows = rows_by_episode.get(segment.source_episode_index, [])
            selected_rows = [
                row
                for row in source_rows
                if segment.start_sec <= float(row["timestamp"]) < segment.end_sec
            ]
            if not selected_rows:
                continue
            output_episode_index = len(episode_meta)
            source_frame_start_sec = float(selected_rows[0]["timestamp"])
            source_frame_end_sec = min(segment.end_sec, float(selected_rows[-1]["timestamp"]) + 1.0 / fps)
            segment_rows: list[dict[str, Any]] = []
            for frame_index, row in enumerate(selected_rows):
                out = dict(row)
                out["timestamp"] = float(row["timestamp"]) - source_frame_start_sec
                out["frame_index"] = frame_index
                out["episode_index"] = output_episode_index
                out["index"] = global_index + frame_index
                out["task_index"] = 0
                segment_rows.append(out)

            segment_bytes = estimate_rows_size_bytes(pa, segment_rows)
            if data_buffer and data_buffer_bytes + segment_bytes > data_target_bytes:
                write_data_file(pa, pq, output_dir, data_location, data_buffer)
                data_location = next_file_location(data_location, chunks_size)
                data_buffer = []
                data_buffer_bytes = 0
            segment_data_location = data_location
            for out in segment_rows:
                for key, accumulator in stats.items():
                    if key in out:
                        accumulator.update(out[key])
            data_buffer.extend(segment_rows)
            data_buffer_bytes += segment_bytes
            global_index += len(segment_rows)
            data_to_index = data_from_index + len(selected_rows)
            episode_meta.append(
                {
                    "episode_index": output_episode_index,
                    "tasks": [spec.task],
                    "length": len(selected_rows),
                    "data/chunk_index": segment_data_location.chunk_index,
                    "data/file_index": segment_data_location.file_index,
                    "dataset_from_index": data_from_index,
                    "dataset_to_index": data_to_index,
                    "source_episode_index": segment.source_episode_index,
                    "source_episode_name": segment.source_episode_name,
                    "source_task": segment.source_task,
                    "source_start_sec": segment.start_sec,
                    "source_end_sec": segment.end_sec,
                    "source_frame_start_sec": source_frame_start_sec,
                    "source_frame_end_sec": source_frame_end_sec,
                }
            )
            data_from_index = data_to_index
    if data_buffer:
        write_data_file(pa, pq, output_dir, data_location, data_buffer)
    print(f"  data: {len(episode_meta)} episodes, {global_index} frames", flush=True)

    if not skip_videos:
        write_segment_videos(
            source_repo_id=source_repo_id,
            output_dir=output_dir,
            episode_meta=episode_meta,
            episode_rows=episode_rows,
            video_keys=video_keys,
            video_mode=video_mode,
            video_files_size_mb=video_files_size_mb,
            chunks_size=chunks_size,
            fps=fps,
        )

    write_episode_meta(pa, pq, output_dir, episode_meta, episodes_per_meta_file)
    stats_json = {key: accumulator.to_dict() for key, accumulator in stats.items()}
    for key in video_keys:
        stats_json[key] = source_stats.get(key, imagenet_video_stats())
    write_json(output_dir / "meta" / "stats.json", stats_json)
    write_info(
        output_dir=output_dir,
        source_info=source_info,
        features=features,
        total_episodes=len(episode_meta),
        total_frames=global_index,
        chunks_size=chunks_size,
        data_files_size_mb=data_files_size_mb,
        video_files_size_mb=video_files_size_mb,
    )
    write_readme(
        output_dir,
        target_repo_id,
        source_repo_id,
        raw_repo_id,
        spec,
        len(episode_meta),
        global_index,
        video_keys,
        skip_videos,
    )


def data_path_from_episode(episode_row: dict[str, Any]) -> str:
    return (
        f"data/chunk-{int(episode_row['data/chunk_index']):03d}/"
        f"file-{int(episode_row['data/file_index']):03d}.parquet"
    )


def estimate_rows_size_bytes(pa: Any, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return int(pa.Table.from_pylist(rows).nbytes)


def next_file_location(location: FileLocation, chunks_size: int) -> FileLocation:
    if location.file_index >= chunks_size - 1:
        return FileLocation(chunk_index=location.chunk_index + 1, file_index=0)
    return FileLocation(chunk_index=location.chunk_index, file_index=location.file_index + 1)


def write_data_file(pa: Any, pq: Any, output_dir: Path, location: FileLocation, rows: list[dict[str, Any]]) -> None:
    path = (
        output_dir
        / "data"
        / f"chunk-{location.chunk_index:03d}"
        / f"file-{location.file_index:03d}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def write_episode_meta(pa: Any, pq: Any, output_dir: Path, episodes: list[dict[str, Any]], episodes_per_file: int) -> None:
    if not episodes:
        return
    for start in range(0, len(episodes), episodes_per_file):
        chunk = episodes[start : start + episodes_per_file]
        file_index = start // episodes_per_file
        path = output_dir / "meta" / "episodes" / "chunk-000" / f"file-{file_index:03d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(chunk), path)


def write_segment_videos(
    *,
    source_repo_id: str,
    output_dir: Path,
    episode_meta: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
    video_mode: str,
    video_files_size_mb: int,
    chunks_size: int,
    fps: float,
) -> int:
    total_video_files = 0
    target_bytes = video_files_size_mb * BYTES_PER_MB
    temp_root = output_dir / ".tmp_video_clips"
    if temp_root.exists():
        shutil.rmtree(temp_root)

    for video_key in video_keys:
        safe_key = video_key.replace("/", "__")
        temp_dir = temp_root / safe_key
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        location = FileLocation(chunk_index=0, file_index=0)
        batch_paths: list[Path] = []
        batch_size = 0
        batch_duration = 0.0
        batch_count = 0
        print(
            f"  videos/{video_key}: clipping {len(episode_meta)} episodes into <= {video_files_size_mb} MB files",
            flush=True,
        )

        def flush_batch() -> None:
            nonlocal batch_paths, batch_size, batch_duration, batch_count, location, total_video_files
            if not batch_paths:
                return
            out_path = (
                output_dir
                / "videos"
                / video_key
                / f"chunk-{location.chunk_index:03d}"
                / f"file-{location.file_index:03d}.mp4"
            )
            concat_video_files(batch_paths, out_path, fps)
            size_mb = out_path.stat().st_size / BYTES_PER_MB
            print(
                f"    wrote {out_path.relative_to(output_dir)}: {batch_count} clips, "
                f"{batch_duration:.1f}s, {size_mb:.1f} MB",
                flush=True,
            )
            for path in batch_paths:
                path.unlink(missing_ok=True)
            total_video_files += 1
            location = next_file_location(location, chunks_size)
            batch_paths = []
            batch_size = 0
            batch_duration = 0.0
            batch_count = 0

        try:
            for clip_index, meta in enumerate(episode_meta, start=1):
                output_episode_index = int(meta["episode_index"])
                source_episode = episode_rows[int(meta["source_episode_index"])]
                clip_path = temp_dir / f"clip-{output_episode_index:06d}.mp4"
                source_chunk = int(source_episode[f"videos/{video_key}/chunk_index"])
                source_file = int(source_episode[f"videos/{video_key}/file_index"])
                source_from = float(source_episode[f"videos/{video_key}/from_timestamp"])
                source_to = float(source_episode[f"videos/{video_key}/to_timestamp"])
                frame_start = float(meta["source_frame_start_sec"])
                frame_end = float(meta["source_frame_end_sec"])
                clip_start = source_from + frame_start
                clip_end = min(source_from + frame_end, source_to)
                if clip_end <= clip_start + VIDEO_DT_TOLERANCE_SEC:
                    raise RuntimeError(
                        f"invalid clip window for {video_key} episode {output_episode_index}: "
                        f"{clip_start:.6f}..{clip_end:.6f}"
                    )
                source_video = hf_hub_download(
                    repo_id=source_repo_id,
                    repo_type="dataset",
                    filename=f"videos/{video_key}/chunk-{source_chunk:03d}/file-{source_file:03d}.mp4",
                )
                clip_video(Path(source_video), clip_path, clip_start, clip_end, video_mode)
                clip_size = clip_path.stat().st_size
                if batch_paths and batch_size + clip_size > target_bytes:
                    flush_batch()

                duration = get_video_duration_sec(clip_path)
                if duration <= VIDEO_DT_TOLERANCE_SEC:
                    duration = clip_end - clip_start
                meta[f"videos/{video_key}/chunk_index"] = location.chunk_index
                meta[f"videos/{video_key}/file_index"] = location.file_index
                meta[f"videos/{video_key}/from_timestamp"] = batch_duration
                meta[f"videos/{video_key}/to_timestamp"] = batch_duration + duration

                batch_paths.append(clip_path)
                batch_size += clip_size
                batch_duration += duration
                batch_count += 1
                if clip_index == len(episode_meta) or clip_index % 100 == 0:
                    print(
                        f"    {video_key}: clipped {clip_index}/{len(episode_meta)}",
                        flush=True,
                    )
            flush_batch()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    shutil.rmtree(temp_root, ignore_errors=True)
    return total_video_files


def clip_video(input_path: Path, output_path: Path, start_sec: float, end_sec: float, mode: str) -> None:
    if output_path.exists():
        return
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-to",
        f"{end_sec:.6f}",
        "-i",
        str(input_path),
    ]
    if mode == "copy":
        cmd.extend(["-c", "copy", "-avoid_negative_ts", "make_zero"])
    else:
        cmd.extend(["-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20"])
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True)


def get_video_duration_sec(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return float(result.stdout.strip())


def concat_video_files(input_paths: list[Path], output_path: Path, fps: float) -> None:
    if not input_paths:
        raise ValueError("no videos to concatenate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(input_paths) == 1:
        shutil.move(str(input_paths[0]), output_path)
        return

    concat_path = output_path.with_suffix(".ffconcat")
    concat_path.write_text(
        "ffconcat version 1.0\n"
        + "".join(f"file '{path.resolve()}'\n" for path in input_paths),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        concat_video_files_reencode(input_paths, output_path, fps)
    finally:
        concat_path.unlink(missing_ok=True)


def concat_video_files_reencode(input_paths: list[Path], output_path: Path, fps: float) -> None:
    max_inputs = 100
    if len(input_paths) <= max_inputs:
        concat_video_files_reencode_once(input_paths, output_path, fps)
        return

    temp_parts: list[Path] = []
    try:
        for part_index, start in enumerate(range(0, len(input_paths), max_inputs)):
            part_path = output_path.with_name(f".{output_path.stem}.part-{part_index:03d}.mp4")
            concat_video_files_reencode_once(input_paths[start : start + max_inputs], part_path, fps)
            temp_parts.append(part_path)
        concat_video_files(temp_parts, output_path, fps)
    finally:
        for part_path in temp_parts:
            part_path.unlink(missing_ok=True)


def concat_video_files_reencode_once(input_paths: list[Path], output_path: Path, fps: float) -> None:
    inputs: list[str] = []
    for path in input_paths:
        inputs.extend(["-i", str(path)])
    filter_complex = "".join(f"[{index}:v]" for index in range(len(input_paths)))
    filter_complex += f"concat=n={len(input_paths)}:v=1:a=0[outv]"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-r",
        f"{fps:g}",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def write_info(
    *,
    output_dir: Path,
    source_info: dict[str, Any],
    features: dict[str, Any],
    total_episodes: int,
    total_frames: int,
    chunks_size: int,
    data_files_size_mb: int,
    video_files_size_mb: int,
) -> None:
    known_info_keys = {
        "codebase_version",
        "fps",
        "features",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "chunks_size",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "data_path",
        "video_path",
        "robot_type",
        "splits",
        "tools",
    }
    info = {key: value for key, value in source_info.items() if key in known_info_keys}
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = 1
    info["chunks_size"] = chunks_size
    info["data_files_size_in_mb"] = data_files_size_mb
    info["video_files_size_in_mb"] = video_files_size_mb
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"] = features
    info["splits"] = {"train": f"0:{total_episodes}"}
    write_json(output_dir / "meta" / "info.json", info)


def write_readme(
    output_dir: Path,
    target_repo_id: str,
    source_repo_id: str,
    raw_repo_id: str,
    spec: SubtaskSpec,
    total_episodes: int,
    total_frames: int,
    video_keys: list[str],
    skip_videos: bool,
) -> None:
    text = f"""# {target_repo_id.split('/', 1)[-1]}

Team RAMEN subtask dataset sliced from `{source_repo_id}`.

- Subtask: `{spec.name}`
- Task: `{spec.task}`
- Episodes: `{total_episodes}`
- Frames: `{total_frames}`
- Videos included: `{not skip_videos}`
- Video keys: `{", ".join(video_keys) if video_keys else "none"}`
- Subtask timestamps: `{raw_repo_id}`

The dataset keeps the official BitRobot LeRobotDataset v3 feature names. No
camera rename, state/action concatenation, or model-specific modality mapping is
applied.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required") from exc
    return pa, pq


def _flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_numeric(item))
        return out
    return []


def imagenet_video_stats() -> dict[str, list[float]]:
    return {
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }


if __name__ == "__main__":
    main()
