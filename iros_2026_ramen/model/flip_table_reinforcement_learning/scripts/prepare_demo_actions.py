#!/usr/bin/env python3
"""Export one real flip-table episode as the canonical 16-D policy prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from data.flip_table_data_augmentation.teleop.shared.policy_contract import (
    ACTION_CONVERSION_VERSION,
)


DEFAULT_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPOSITORY_ROOT / ".checkpoints" / "flip_table_episode0_actions.json"
ROBOT_Q_DESIRED_KEY = "action.robot_q_desired"
HAND_COMMAND_KEY = "action.hand_cmd"
# robot_q_desired is root pose (7), lower body (12), waist (3), then both arms (14).
LEGACY_UPPER_BODY_SLICE = slice(19, 36)
ARM_SLICE = slice(22, 36)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_dataset_root(
    repo_id: str,
    *,
    revision: str | None,
    dataset_root: Path | None,
) -> Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=("meta/info.json", "meta/episodes/**", "data/**"),
        )
    ).resolve()


def _finite_float_list(value: Any, *, expected: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"{label} must contain {expected} values, got {value!r}")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains NaN or Inf")
    return result


def export_demo_actions(
    source_root: Path,
    *,
    repo_id: str,
    revision: str | None,
    episode: int,
) -> dict[str, Any]:
    if episode < 0:
        raise ValueError("episode must be non-negative")
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError(f"source must be LeRobot v3, got {info.get('codebase_version')!r}")
    if float(info.get("fps", 0.0)) != 30.0:
        raise ValueError(f"real demonstration must be 30 Hz, got {info.get('fps')!r}")
    features = info.get("features", {})
    expected_features = {ROBOT_Q_DESIRED_KEY: 36, HAND_COMMAND_KEY: 2}
    for key, width in expected_features.items():
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "float32" or feature.get("shape") != [width]:
            raise ValueError(f"source feature {key!r} must be float32[{width}], got {feature!r}")

    data_files = sorted((source_root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"no LeRobot data parquets under {source_root / 'data'}")
    table = pads.dataset([str(path) for path in data_files], format="parquet").to_table(
        columns=["episode_index", "frame_index", ROBOT_Q_DESIRED_KEY, HAND_COMMAND_KEY],
        filter=pads.field("episode_index") == episode,
    )
    if table.num_rows == 0:
        raise ValueError(f"episode {episode} is absent from {source_root}")
    table = table.sort_by("frame_index")
    frame_indices = [int(value) for value in table["frame_index"].to_pylist()]
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"episode {episode} frame_index is not contiguous from zero")

    actions: list[list[float]] = []
    legacy_actions: list[list[float]] = []
    for row_index, (desired_raw, hand_raw) in enumerate(
        zip(
            table[ROBOT_Q_DESIRED_KEY].to_pylist(),
            table[HAND_COMMAND_KEY].to_pylist(),
            strict=True,
        )
    ):
        desired = _finite_float_list(
            desired_raw,
            expected=36,
            label=f"frame {row_index} robot_q_desired",
        )
        hands = _finite_float_list(
            hand_raw,
            expected=2,
            label=f"frame {row_index} hand_cmd",
        )
        if any(value < 0.0 or value > 4.5 for value in hands):
            raise ValueError(f"frame {row_index} hand_cmd is outside the real [0,4.5] range")
        legacy = desired[LEGACY_UPPER_BODY_SLICE] + hands
        legacy_actions.append(legacy)
        actions.append(desired[ARM_SLICE] + hands)

    source_action_bytes = json.dumps(
        legacy_actions,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    episode_metadata = _episode_metadata(source_root, episode)
    return {
        "schema_version": "team_ramen_flip_table_demo_prior_v2",
        "source_repo_id": repo_id,
        "source_revision": revision,
        "source_episode_index": episode,
        "source_episode_name": episode_metadata.get("source_episode_name"),
        "fps": 30,
        "state_layout": "waist_3_left_arm_7_right_arm_7_left_hand_right_hand",
        "action_layout": "left_arm_7_right_arm_7_left_hand_right_hand",
        "state_dim": 19,
        "action_dim": 16,
        "legacy_action_conversion_version": ACTION_CONVERSION_VERSION,
        "legacy_source_actions_sha256": hashlib.sha256(source_action_bytes).hexdigest(),
        "dropped_action_fields": ["waist_3"],
        "hand_command_range": [0.0, 4.5],
        "actions": actions,
    }


def _episode_metadata(source_root: Path, episode: int) -> dict[str, Any]:
    for path in sorted((source_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "=", episode)])
        if table.num_rows:
            rows = table.to_pylist()
            if len(rows) != 1:
                raise ValueError(f"episode {episode} has {len(rows)} metadata rows")
            return rows[0]
    raise ValueError(f"episode metadata is missing for episode {episode}")


def main() -> None:
    args = parse_args()
    source_root = resolve_dataset_root(
        args.repo_id,
        revision=args.revision,
        dataset_root=args.dataset_root,
    )
    payload = export_demo_actions(
        source_root,
        repo_id=args.repo_id,
        revision=args.revision,
        episode=args.episode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"Wrote {len(payload['actions'])} real 16-D targets to {args.output}")


if __name__ == "__main__":
    main()
