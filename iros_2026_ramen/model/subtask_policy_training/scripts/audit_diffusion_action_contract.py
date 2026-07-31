#!/usr/bin/env python3
"""Verify the numerical contract of a materialized flip-table Diffusion view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.action_representation import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
    CHUNK_SIZE,
    STATE_ARM_START,
    load_training_stats,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (  # noqa: E402
    UPPER_BODY_ACTION_NAMES,
    UPPER_BODY_STATE_NAMES,
)


CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
SOURCE_RGB_KEYS = (
    "observation.images.cam_0",
    "observation.images.cam_1",
    "observation.images.cam_2",
    "observation.images.cam_3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_rows(dataset_root: Path, train_episodes: set[int]) -> dict[int, list[tuple[int, np.ndarray, np.ndarray]]]:
    by_episode: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        for row in table.to_pylist():
            episode = int(row["episode_index"])
            if episode not in train_episodes:
                continue
            state = np.asarray(row["observation.state"], dtype=np.float64)
            action = np.asarray(row["action"], dtype=np.float64)
            if state.shape != (19,) or action.shape != (ACTION_DIM,):
                raise ValueError(f"invalid state/action shape in episode {episode}: {state.shape}/{action.shape}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"non-finite state/action in episode {episode}")
            by_episode[episode].append((int(row["frame_index"]), state, action))
    for episode, rows in by_episode.items():
        rows.sort(key=lambda value: value[0])
        if [value[0] for value in rows] != list(range(len(rows))):
            raise ValueError(f"episode {episode} has non-contiguous frame indices")
    if set(by_episode) != train_episodes:
        raise ValueError(f"training rows do not cover split: expected={sorted(train_episodes)}, found={sorted(by_episode)}")
    return by_episode


def summarize(values: np.ndarray) -> dict[str, Any]:
    return {
        "count": int(values.shape[0]),
        "per_dimension_mean": values.mean(axis=0).tolist(),
        "per_dimension_std": values.std(axis=0).tolist(),
        "per_dimension_min": values.min(axis=0).tolist(),
        "per_dimension_max": values.max(axis=0).tolist(),
        "fraction_outside_minus_one_to_one": float(np.mean(np.abs(values) > 1.0)),
        "fraction_outside_minus_three_to_three": float(np.mean(np.abs(values) > 3.0)),
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    marker = json.loads((root / "meta" / "team_ramen_training_view.json").read_text(encoding="utf-8"))
    split = json.loads((root / "meta" / "team_ramen_episode_split.json").read_text(encoding="utf-8"))

    features = info["features"]
    if tuple(key for key in CAMERA_KEYS if key in features) != CAMERA_KEYS:
        raise ValueError("training view does not contain the canonical three policy cameras")
    if features["observation.state"].get("names") != list(UPPER_BODY_STATE_NAMES):
        raise ValueError("state joint order differs from the real G1 upper-body contract")
    if features["action"].get("names") != list(UPPER_BODY_ACTION_NAMES):
        raise ValueError("action joint order differs from the real G1 upper-body contract")
    if features["observation.state"].get("shape") != [19] or features["action"].get("shape") != [16]:
        raise ValueError("training view must be state19/action16")
    forbidden_output_names = [
        name for name in features["action"]["names"] if any(token in name for token in ("waist", "hip", "knee", "ankle"))
    ]
    if forbidden_output_names:
        raise ValueError(f"policy action leaks non-arm joints: {forbidden_output_names}")

    source_camera_map = marker.get("camera_map", {})
    expected_source_map = {
        "head_left": "observation.images.cam_0",
        "head_right": "observation.images.cam_1",
        "left_wrist": "observation.images.cam_2",
        "right_wrist": "observation.images.cam_3",
    }
    source_features = json.loads(
        (Path(marker["source_root"]) / "meta" / "info.json").read_text(encoding="utf-8")
    )["features"]
    missing_source_rgb = [key for key in SOURCE_RGB_KEYS if key not in source_features]
    if missing_source_rgb:
        raise ValueError(f"source dataset misses RGB cameras: {missing_source_rgb}")
    if source_camera_map != {
        "observation.images.head_left": expected_source_map["head_left"],
        "observation.images.left_wrist": expected_source_map["left_wrist"],
        "observation.images.right_wrist": expected_source_map["right_wrist"],
    }:
        raise ValueError(f"unexpected policy camera mapping: {source_camera_map}")

    splits = split["splits"]
    train_episodes = {int(value) for value in splits["train"]["episode_indices"]}
    test_episodes = {int(value) for value in splits["test"]["episode_indices"]}
    validation_episodes = {int(value) for value in splits["validation"]["episode_indices"]}
    if not train_episodes or train_episodes & test_episodes or train_episodes & validation_episodes:
        raise ValueError("training split leaks episodes or is empty")
    if set(splits["train"]["source_episode_names"]) & set(splits["test"]["source_episode_names"]):
        raise ValueError("training split leaks source recordings")

    stats = load_training_stats(root, CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER)
    mean = np.asarray(stats["action"]["mean"], dtype=np.float64)
    std = np.asarray(stats["action"]["std"], dtype=np.float64)
    if mean.shape != (ACTION_DIM,) or std.shape != (ACTION_DIM,) or np.any(std <= 0.0):
        raise ValueError("invalid z-score action statistics")

    normalized_chunks: list[np.ndarray] = []
    raw_chunks: list[np.ndarray] = []
    for rows in load_rows(root, train_episodes).values():
        states = np.asarray([row[1] for row in rows], dtype=np.float64)
        actions = np.asarray([row[2] for row in rows], dtype=np.float64)
        for start in range(len(rows)):
            encoded = actions[start : start + CHUNK_SIZE].copy()
            encoded[:, :ARM_DIM] -= states[start, STATE_ARM_START : STATE_ARM_START + ARM_DIM]
            raw_chunks.append(encoded)
            normalized_chunks.append((encoded - mean) / std)
    raw_values = np.concatenate(raw_chunks, axis=0)
    normalized_values = np.concatenate(normalized_chunks, axis=0)
    reconstructed = normalized_values * std + mean
    inverse_error = float(np.max(np.abs(reconstructed - raw_values)))
    if inverse_error > 1.0e-10:
        raise RuntimeError(f"z-score inverse transform error is too large: {inverse_error}")

    from model.subtask_policy_training.scripts.train_native_diffusion_delta import build_policy

    policy, config = build_policy("cpu")
    scheduler_clip_sample = bool(policy.diffusion.noise_scheduler.config.clip_sample)
    del policy
    if bool(config.clip_sample) or scheduler_clip_sample:
        raise ValueError("z-score normalization requires clip_sample=false in config and scheduler")

    report = {
        "schema_version": "flip_table_diffusion_action_contract_v1",
        "dataset_root": str(root),
        "source_repo_id": marker["source_repo_id"],
        "source_revision": marker["source_revision"],
        "training_view_split_sha256": split["sha256"],
        "policy_cameras": list(CAMERA_KEYS),
        "source_rgb_cameras_audited": list(SOURCE_RGB_KEYS),
        "head_right_policy_usage": "excluded; head-left plus two D405 cameras is the real deployment contract",
        "state_dim": 19,
        "action_dim": ACTION_DIM,
        "action_names": features["action"]["names"],
        "action_contract": "14 arm targets relative to q_current at each chunk start; 2 absolute Dex1 commands",
        "normalization": "zscore",
        "clip_sample": False,
        "scheduler_clip_sample": scheduler_clip_sample,
        "padding_loss_mask": bool(config.do_mask_loss_for_padding),
        "inverse_transform_max_abs_error": inverse_error,
        "raw_model_space_action": summarize(raw_values),
        "normalized_model_space_action": summarize(normalized_values),
        "split_episode_counts": {name: len(values["episode_indices"]) for name, values in splits.items()},
        "stats_sha256": sha256_json(stats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
