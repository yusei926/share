"""Assemble real and rendered synthetic episodes with LeRobot 0.6.0 APIs."""

from __future__ import annotations

from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image

from .contracts import LINEAGE_SCHEMA_VERSION, NUMERIC_KEYS, TASK, RenderedEpisode, lineage_split
from .file_manifest import write_file_manifest
from .recompute_stats import recompute_dataset_stats
from ..config import EXPECTED_CAMERA_KEYS, PipelineConfig
from ..io_utils import atomic_write_json, atomic_write_text, read_json_object, sha256_file
from ..source_contract import validate_source_info


LEROBOT_VERSION = "0.6.0"
AUTOMATIC_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
SPLIT_NAMES = ("train", "validation", "test")
OrderedEpisode = tuple[Literal["real", "synthetic"], int | RenderedEpisode]


def _require_lerobot_060(config: PipelineConfig):
    installed = version("lerobot")
    if installed != config.dataset_runtime.lerobot_version or installed != LEROBOT_VERSION:
        raise RuntimeError(
            f"dataset assembly requires lerobot=={config.dataset_runtime.lerobot_version}, found {installed}"
        )
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.dataset_tools import modify_features, split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset, aggregate_datasets, modify_features, split_dataset


def _canonical_source_tasks(source_root: Path):
    """Normalize the pinned source task table without editing the source."""

    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for LeRobot dataset assembly") from exc
    source = pd.read_parquet(source_root / "meta" / "tasks.parquet")
    if "task_index" not in source or source["task_index"].tolist() != [0]:
        raise ValueError("source tasks must contain exactly task_index 0")
    if source.index.tolist() == [TASK]:
        task_names = source.index.tolist()
    else:
        name_columns = [column for column in source.columns if column != "task_index"]
        if len(name_columns) != 1:
            raise ValueError("source tasks must contain exactly one task-name column")
        task_names = source[name_columns[0]].tolist()
    if task_names != [TASK]:
        raise ValueError(f"source task 0 must be {TASK!r}")
    return pd.DataFrame(
        {"task_index": [0]},
        index=pd.Index([TASK], name="task"),
    )


def _canonicalize_video_depth_metadata(features: dict[str, Any]) -> None:
    """Normalize the pinned source RGB metadata for LeRobot 0.6.0."""

    for key in EXPECTED_CAMERA_KEYS:
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"source feature {key} is not a video")
        info = feature.get("info")
        if not isinstance(info, dict):
            raise ValueError(f"source feature {key} lacks video metadata")
        source_flag = info.pop("video.is_depth_map", None)
        current = info.get("is_depth_map", source_flag)
        if current is not False or source_flag not in (None, False):
            raise ValueError(f"source feature {key} must be RGB, not a depth map")
        info["is_depth_map"] = False


def _synthetic_features(source_info: dict[str, Any]) -> dict[str, Any]:
    features = source_info.get("features")
    if not isinstance(features, dict):
        raise ValueError("source info features must be an object")
    selected = {
        key: value
        for key, value in features.items()
        if key in NUMERIC_KEYS or key in EXPECTED_CAMERA_KEYS
    }
    expected = set(NUMERIC_KEYS).union(EXPECTED_CAMERA_KEYS)
    if set(selected) != expected:
        raise ValueError(f"source feature schema is missing {sorted(expected - set(selected))}")
    return {key: value for key, value in selected.items() if key not in AUTOMATIC_FEATURES}


def _load_numeric_trace(episode: RenderedEpisode) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to export rendered numeric traces") from exc
    table = pq.read_table(episode.numeric_trace, columns=list(NUMERIC_KEYS))
    if table.num_rows != episode.frame_count:
        raise ValueError(
            f"{episode.candidate_id} numeric trace has {table.num_rows} rows, expected {episode.frame_count}"
        )
    arrays: dict[str, np.ndarray] = {}
    for key in NUMERIC_KEYS:
        array = np.asarray(table[key].to_pylist(), dtype=np.float32)
        expected_width = {
            "observation.state.ee_state": 12,
            "observation.state.hand_state": 2,
            "observation.state.robot_q_current": 36,
            "action.ee_action": 12,
            "action.hand_cmd": 2,
            "action.robot_q_desired": 36,
        }[key]
        if array.shape != (episode.frame_count, expected_width):
            raise ValueError(
                f"{episode.candidate_id} {key} has shape {array.shape}, "
                f"expected {(episode.frame_count, expected_width)}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{episode.candidate_id} {key} contains NaN or Inf")
        arrays[key] = array
    return arrays


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (480, 640, 3):
        raise ValueError(f"rendered camera frame {path} has shape {rgb.shape}, expected (480, 640, 3)")
    return rgb


def _validate_render_set(
    episodes: Iterable[RenderedEpisode], *, min_appearance_variants: int
) -> tuple[RenderedEpisode, ...]:
    if (
        isinstance(min_appearance_variants, bool)
        or not isinstance(min_appearance_variants, int)
        or min_appearance_variants <= 0
    ):
        raise ValueError("min_appearance_variants must be a positive integer")
    values = tuple(episodes)
    if not values:
        raise ValueError("at least one rendered synthetic episode is required")
    identities = [(episode.candidate_id, episode.appearance_variant) for episode in values]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate/appearance identities must be unique")
    trajectory_hashes: dict[str, str] = {}
    lineage_by_candidate: dict[str, tuple[str, tuple[int, ...]]] = {}
    kinds_by_candidate: dict[str, tuple[str, str]] = {}
    variant_indices: dict[str, set[int]] = {}
    for episode in values:
        existing = trajectory_hashes.setdefault(episode.candidate_id, episode.trajectory_sha256)
        if existing != episode.trajectory_sha256:
            raise ValueError("one candidate_id cannot refer to multiple physical trajectories")
        lineage = (episode.source_trajectory_lineage, episode.source_episode_indices)
        existing_lineage = lineage_by_candidate.setdefault(episode.candidate_id, lineage)
        if existing_lineage != lineage:
            raise ValueError("appearance variants of one candidate must share immutable source lineage")
        kinds = (episode.trajectory_kind, episode.source_kind)
        existing_kinds = kinds_by_candidate.setdefault(episode.candidate_id, kinds)
        if existing_kinds != kinds:
            raise ValueError("appearance variants of one candidate must share trajectory/source kinds")
        variant_indices.setdefault(episode.candidate_id, set()).add(episode.appearance_variant)
    deficient = sorted(
        candidate
        for candidate, variants in variant_indices.items()
        if kinds_by_candidate[candidate][0] == "mimic"
        and len(variants) < min_appearance_variants
    )
    if deficient:
        raise ValueError(
            f"candidates need at least {min_appearance_variants} appearance variants: {deficient[:10]}"
        )
    invalid_direct = sorted(
        candidate
        for candidate, variants in variant_indices.items()
        if kinds_by_candidate[candidate][0] == "direct_sim_teleop" and variants != {0}
    )
    if invalid_direct:
        raise ValueError(
            "direct sim teleop candidates must contain exactly appearance variant zero: "
            f"{invalid_direct[:10]}"
        )
    noncontiguous = sorted(
        candidate
        for candidate, variants in variant_indices.items()
        if variants != set(range(max(variants) + 1))
    )
    if noncontiguous:
        raise ValueError(f"appearance variants must be contiguous from zero: {noncontiguous[:10]}")
    candidates_by_hash: dict[str, str] = {}
    for candidate_id, trajectory_sha256 in trajectory_hashes.items():
        previous = candidates_by_hash.setdefault(trajectory_sha256, candidate_id)
        if previous != candidate_id:
            raise ValueError(
                f"candidates {previous!r} and {candidate_id!r} duplicate one physical trajectory"
            )
    return tuple(sorted(values, key=lambda item: (item.candidate_id, item.appearance_variant)))


def _write_lineage_sidecars(
    target_root: Path,
    *,
    ordered_episodes: tuple[OrderedEpisode, ...],
    config: PipelineConfig,
) -> dict[str, Any]:
    metadata_dir = target_root / "meta" / "augmentation"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    split_weights = {name: float(config.raw["splits"][name]) for name in SPLIT_NAMES}
    records: list[dict[str, Any]] = []
    for output_index, (kind, value) in enumerate(ordered_episodes):
        if kind == "real":
            source_index = int(value)
            lineage = f"real:{config.source.repo_id}@{config.source.revision}:{source_index:06d}"
            records.append(
                {
                    "schema_version": LINEAGE_SCHEMA_VERSION,
                    "episode_index": output_index,
                    "kind": "real",
                    "split": lineage_split(lineage, split_weights),
                    "source_trajectory_lineage": lineage,
                    "source_repo_id": config.source.repo_id,
                    "source_revision": config.source.revision,
                    "source_episode_indices": [source_index],
                    "selected_features_copied_without_numeric_transform": True,
                }
            )
            continue
        if kind != "synthetic" or not isinstance(value, RenderedEpisode):
            raise ValueError(f"unsupported ordered episode entry: {(kind, value)!r}")
        episode = value
        records.append(
            {
                "schema_version": LINEAGE_SCHEMA_VERSION,
                "episode_index": output_index,
                "kind": "synthetic",
                "trajectory_kind": episode.trajectory_kind,
                "source_kind": episode.source_kind,
                "split": lineage_split(episode.source_trajectory_lineage, split_weights),
                "source_trajectory_lineage": episode.source_trajectory_lineage,
                "source_repo_id": config.source.repo_id,
                "source_revision": config.source.revision,
                "source_episode_indices": list(episode.source_episode_indices),
                "candidate_id": episode.candidate_id,
                "appearance_variant": episode.appearance_variant,
                "trajectory_sha256": episode.trajectory_sha256,
                "numeric_trace_sha256": sha256_file(episode.numeric_trace),
                "render_manifest_sha256": sha256_file(episode.manifest_path),
                "runtime_manifest_sha256": episode.runtime_manifest_sha256,
                "config_sha256": episode.config_sha256,
                "randomization": episode.randomization,
                "success_report": episode.success_report,
            }
        )
    lineage_path = metadata_dir / "episodes.jsonl"
    atomic_write_text(
        lineage_path,
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
    )
    summary = {
        "schema_version": "team_ramen_flip_table_augmentation_summary/v1",
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "pipeline_config_sha256": config.digest,
        "real_episodes": sum(record["kind"] == "real" for record in records),
        "synthetic_episodes": sum(record["kind"] == "synthetic" for record in records),
        "successful_physical_trajectories": len(
            {record["candidate_id"] for record in records if record["kind"] == "synthetic"}
        ),
        "appearance_variants": sum(record["kind"] == "synthetic" for record in records),
        "split_counts": dict(Counter(record["split"] for record in records)),
    }
    atomic_write_json(metadata_dir / "summary.json", summary)
    return summary


def _lineage_split_groups(
    source_episode_indices: tuple[int, ...],
    rendered: tuple[RenderedEpisode, ...],
    config: PipelineConfig,
) -> tuple[
    dict[str, list[int]],
    dict[str, list[RenderedEpisode]],
    tuple[OrderedEpisode, ...],
]:
    weights = {name: float(config.raw["splits"][name]) for name in SPLIT_NAMES}
    real = {name: [] for name in SPLIT_NAMES}
    synthetic = {name: [] for name in SPLIT_NAMES}
    for source_index in source_episode_indices:
        lineage = f"real:{config.source.repo_id}@{config.source.revision}:{source_index:06d}"
        real[lineage_split(lineage, weights)].append(source_index)
    for episode in rendered:
        synthetic[lineage_split(episode.source_trajectory_lineage, weights)].append(episode)
    ordered: list[OrderedEpisode] = []
    for name in SPLIT_NAMES:
        ordered.extend(("real", index) for index in real[name])
        ordered.extend(("synthetic", episode) for episode in synthetic[name])
    return real, synthetic, tuple(ordered)


def _local_real_split_indices(
    selected_source: tuple[int, ...],
    real_groups: dict[str, list[int]],
) -> dict[str, list[int]]:
    source_to_local_index = {
        source_episode_index: local_episode_index
        for local_episode_index, source_episode_index in enumerate(selected_source)
    }
    flattened = [index for name in SPLIT_NAMES for index in real_groups[name]]
    if sorted(flattened) != list(selected_source):
        raise RuntimeError("real split groups do not cover the selected source exactly once")
    return {
        name: [source_to_local_index[index] for index in real_groups[name]]
        for name in SPLIT_NAMES
        if real_groups[name]
    }


def _set_contiguous_splits(
    output: Path,
    ordered_episodes: tuple[OrderedEpisode, ...],
    config: PipelineConfig,
) -> None:
    counts = Counter()
    weights = {
        name: float(config.raw["splits"][name])
        for name in SPLIT_NAMES
    }
    seen_order = []
    for kind, value in ordered_episodes:
        if kind == "real":
            lineage = f"real:{config.source.repo_id}@{config.source.revision}:{int(value):06d}"
        else:
            lineage = value.source_trajectory_lineage
        split = lineage_split(lineage, weights)
        counts[split] += 1
        if not seen_order or seen_order[-1] != split:
            seen_order.append(split)
    expected_order = [name for name in SPLIT_NAMES if counts[name]]
    if seen_order != expected_order:
        raise RuntimeError("assembled episodes are not contiguous by lineage split")
    ranges = {}
    start = 0
    for name in expected_order:
        end = start + counts[name]
        ranges[name] = f"{start}:{end}"
        start = end
    info_path = output / "meta" / "info.json"
    info = read_json_object(info_path)
    if int(info.get("total_episodes", -1)) != len(ordered_episodes):
        raise RuntimeError("aggregate episode count changed before split metadata update")
    info["splits"] = ranges
    atomic_write_json(info_path, info)


def _write_dataset_card(target_root: Path, summary: dict[str, Any], config: PipelineConfig) -> None:
    content = f"""---
license: other
task_categories:
- robotics
tags:
- lerobot
- humanoid
- sim-to-real
viewer: false
---

# IROS2026 RAMEN flip-table augmented dataset

This private LeRobotDataset v3 combines {summary['real_episodes']} unchanged real
flip-table episodes with {summary['synthetic_episodes']} physically accepted synthetic
episodes. The real data derives from `{config.source.repo_id}` at immutable revision
`{config.source.revision}` (upstream raw data: CC BY 4.0). Synthetic imagery and
trajectories were produced with the organizer RoboFinals IKEA V1 environment; use is
also subject to the organizer asset and competition terms.

## Features

- RGB: `observation.images.cam_0` (head-left), `cam_2` (left wrist), and `cam_3`
  (right wrist), all 640x480 at 30 fps.
- State: 12-D root-frame bilateral EEF pose, 2-D Dex1 state, and 36-D G1 state.
- Action: 12-D root-frame bilateral EEF target, 2-D Dex1 command, and 36-D G1 target.

Synthetic candidates are generated camera-free with Isaac Lab Mimic, rejected unless
the strict V1 physics and safety contract passes, then replayed for Replicator RGB
rendering. Sim-only object pose, contact, segmentation, and success signals are not
policy features. `meta/augmentation/episodes.jsonl` records immutable source lineage,
seeded randomization, runtime/config hashes, and success evidence for every episode.
All appearance variants of one physical source lineage remain in one split.

## Limitations

The synthetic data has not by itself demonstrated real-robot transfer. Camera,
material, dynamics, contact, and latency randomization reduce but do not eliminate the
sim-to-real gap. Validate on G1 + Dex1-1 with conservative joint, velocity, acceleration,
force, and emergency-stop limits before deployment.
"""
    atomic_write_text(target_root / "README.md", content)


def _write_gitattributes(target_root: Path) -> None:
    atomic_write_text(
        target_root / ".gitattributes",
        "*.mp4 filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
    )


def assemble_dataset(
    *,
    source_root: str | Path,
    render_manifests: Iterable[str | Path],
    output_root: str | Path,
    work_root: str | Path,
    config: PipelineConfig,
    source_episode_indices: Iterable[int] | None = None,
    min_appearance_variants: int | None = None,
) -> dict[str, Any]:
    """Build a final v3 dataset without modifying the pinned source snapshot."""

    LeRobotDataset, aggregate_datasets, modify_features, split_dataset = _require_lerobot_060(config)
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    work = Path(work_root).expanduser().resolve()
    if output.exists() or work.exists():
        raise FileExistsError("output_root and work_root must not already exist")
    info = read_json_object(source / "meta" / "info.json")
    validate_source_info(info, config)
    selected_source = (
        tuple(range(config.source.episodes))
        if source_episode_indices is None
        else tuple(sorted(set(int(index) for index in source_episode_indices)))
    )
    if not selected_source or any(index < 0 or index >= config.source.episodes for index in selected_source):
        raise ValueError("source_episode_indices contains an invalid source episode")
    configured_minimum_variants = int(
        config.raw["generation"]["appearance_variants_per_trajectory_min"]
    )
    requested_minimum_variants = (
        configured_minimum_variants
        if min_appearance_variants is None
        else min_appearance_variants
    )
    if requested_minimum_variants < configured_minimum_variants:
        raise ValueError(
            "min_appearance_variants cannot weaken the configured generation gate"
        )
    rendered = _validate_render_set(
        (RenderedEpisode.load(path) for path in render_manifests),
        min_appearance_variants=requested_minimum_variants,
    )
    if any(episode.config_sha256 != config.digest for episode in rendered):
        raise ValueError("rendered episode config hash does not match the active pipeline config")
    selected_source_set = set(selected_source)
    unknown_synthetic_sources = sorted(
        {
            index
            for episode in rendered
            if episode.source_kind == "real_demo"
            for index in episode.source_episode_indices
            if index not in selected_source_set
        }
    )
    if unknown_synthetic_sources:
        raise ValueError(
            "rendered synthetic lineage refers to source episodes excluded from this build: "
            f"{unknown_synthetic_sources[:10]}"
        )

    work.mkdir(parents=True, exist_ok=False)
    source_dataset = LeRobotDataset(config.source.repo_id, root=source)
    source_dataset.meta.tasks = _canonical_source_tasks(source)
    _canonicalize_video_depth_metadata(source_dataset.meta.features)
    from lerobot.configs import encoder_config_from_video_info

    source_video_info = info["features"][EXPECTED_CAMERA_KEYS[0]].get("info")
    if not isinstance(source_video_info, dict):
        raise ValueError("source head-left feature lacks video encoder metadata")
    rgb_encoder = encoder_config_from_video_info(source_video_info)
    if rgb_encoder.vcodec != "h264" or rgb_encoder.pix_fmt != "yuv420p":
        raise ValueError("synthetic RGB encoder must preserve source H.264 yuv420p storage")
    if len(selected_source) != config.source.episodes:
        source_dataset = split_dataset(
            source_dataset,
            {"selected": list(selected_source)},
            output_dir=work / "source_subset",
        )["selected"]
    source_video_keys = tuple(source_dataset.meta.video_keys)
    removed_video_keys = [key for key in source_video_keys if key not in EXPECTED_CAMERA_KEYS]
    real_dataset = modify_features(
        source_dataset,
        remove_features=removed_video_keys,
        output_dir=work / "real_three_camera",
        repo_id="Team-RAMEN/flip_table_real_three_camera_staging",
    )
    if tuple(real_dataset.meta.video_keys) != EXPECTED_CAMERA_KEYS:
        raise RuntimeError("real dataset camera reduction did not produce the three-camera contract")

    real_groups, synthetic_groups, ordered_episodes = _lineage_split_groups(
        selected_source, rendered, config
    )
    nonempty_real = _local_real_split_indices(selected_source, real_groups)
    real_split_datasets = split_dataset(
        real_dataset,
        nonempty_real,
        output_dir=work / "real_splits",
    )

    synthetic_split_datasets = {}
    for split_name in SPLIT_NAMES:
        split_episodes = synthetic_groups[split_name]
        if not split_episodes:
            continue
        synthetic_dataset = LeRobotDataset.create(
            repo_id=f"Team-RAMEN/flip_table_synthetic_{split_name}_staging",
            fps=config.source.fps,
            features=_synthetic_features(info),
            root=work / "synthetic_splits" / split_name,
            robot_type=info.get("robot_type"),
            use_videos=True,
            data_files_size_in_mb=config.target.data_shard_size_mb,
            video_files_size_in_mb=config.target.video_shard_size_mb,
            image_writer_threads=4,
            encoder_threads=4,
            rgb_encoder=rgb_encoder,
        )
        for episode in split_episodes:
            numeric = _load_numeric_trace(episode)
            for frame_index in range(episode.frame_count):
                frame = {key: numeric[key][frame_index] for key in NUMERIC_KEYS}
                for camera_key in EXPECTED_CAMERA_KEYS:
                    frame[camera_key] = _image(
                        episode.camera_dirs[camera_key] / f"frame_{frame_index:06d}.png"
                    )
                frame["task"] = episode.task
                synthetic_dataset.add_frame(frame)
            synthetic_dataset.save_episode(parallel_encoding=True)
        synthetic_dataset.finalize()
        synthetic_split_datasets[split_name] = synthetic_dataset

    components = []
    for split_name in SPLIT_NAMES:
        if split_name in real_split_datasets:
            components.append(real_split_datasets[split_name])
        if split_name in synthetic_split_datasets:
            components.append(synthetic_split_datasets[split_name])

    aggregate_datasets(
        repo_ids=[dataset.repo_id for dataset in components],
        roots=[dataset.root for dataset in components],
        aggr_repo_id=config.target.repo_id,
        aggr_root=output,
        data_files_size_in_mb=config.target.data_shard_size_mb,
        video_files_size_in_mb=config.target.video_shard_size_mb,
        concatenate_data=True,
        concatenate_videos=True,
    )
    _set_contiguous_splits(output, ordered_episodes, config)
    recompute_dataset_stats(output, config)
    summary = _write_lineage_sidecars(
        output,
        ordered_episodes=ordered_episodes,
        config=config,
    )
    _write_dataset_card(output, summary, config)
    _write_gitattributes(output)
    manifest_path = write_file_manifest(output)
    return {
        **summary,
        "output_root": str(output),
        "file_manifest": str(manifest_path),
    }
