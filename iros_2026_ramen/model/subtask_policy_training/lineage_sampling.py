"""Deterministic category and lineage-balanced sampling for augmented datasets."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PLAN_SCHEMA_VERSION = "team_ramen_lineage_sampling_plan/v1"
PLAN_ENV = "TEAM_RAMEN_SAMPLING_PLAN"

SAMPLING_CONDITIONS: dict[str, dict[str, float]] = {
    "real_only": {"real": 1.0},
    "real_sim_teleop": {"real": 0.9, "direct_sim_teleop": 0.1},
    "real_sim_teleop_mimic": {
        "real": 0.5,
        "direct_sim_teleop": 0.1,
        "mimic": 0.4,
    },
}


def canonical_condition(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "real": "real_only",
        "real_sim": "real_sim_teleop",
        "full": "real_sim_teleop_mimic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SAMPLING_CONDITIONS:
        choices = ", ".join(sorted(SAMPLING_CONDITIONS))
        raise ValueError(f"unsupported training condition {value!r}; expected one of: {choices}")
    return normalized


def lineage_category(record: Mapping[str, Any]) -> str:
    kind = record.get("kind")
    if kind == "real":
        return "real"
    if kind != "synthetic":
        raise ValueError(f"unsupported augmentation episode kind: {kind!r}")
    trajectory_kind = record.get("trajectory_kind")
    if trajectory_kind == "direct_sim_teleop":
        if record.get("source_kind") != "sim_teleop":
            raise ValueError("direct sim teleop record must have source_kind='sim_teleop'")
        return "direct_sim_teleop"
    if trajectory_kind == "mimic":
        return "mimic"
    raise ValueError(f"unsupported synthetic trajectory_kind: {trajectory_kind!r}")


def load_augmentation_records(path: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"augmentation line {line_number} is not an object")
        episode_index = value.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"augmentation line {line_number} has invalid episode_index")
        if episode_index in records:
            raise ValueError(f"duplicate augmentation episode_index: {episode_index}")
        lineage = value.get("source_trajectory_lineage")
        split = value.get("split")
        if not isinstance(lineage, str) or not lineage:
            raise ValueError(f"augmentation line {line_number} lacks source lineage")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"augmentation line {line_number} has invalid split: {split!r}")
        lineage_category(value)
        records[episode_index] = value
    if not records:
        raise ValueError(f"augmentation metadata is empty: {path}")
    if sorted(records) != list(range(len(records))):
        raise ValueError("augmentation episode indices must be contiguous from zero")
    return records


def build_sampling_plan(
    *,
    records: Mapping[int, Mapping[str, Any]],
    train_episode_indices: Sequence[int],
    condition: str,
    split_sha256: str,
) -> dict[str, Any]:
    condition = canonical_condition(condition)
    weights = SAMPLING_CONDITIONS[condition]
    episodes: list[dict[str, Any]] = []
    category_counts: dict[str, int] = defaultdict(int)
    for episode_index in train_episode_indices:
        record = records.get(int(episode_index))
        if record is None:
            raise ValueError(f"training episode {episode_index} is absent from augmentation metadata")
        if record.get("split") != "train":
            raise ValueError(f"episode {episode_index} is in the training split but sidecar says otherwise")
        category = lineage_category(record)
        if category not in weights:
            continue
        lineage = str(record["source_trajectory_lineage"])
        appearance_variant = record.get("appearance_variant", 0)
        if isinstance(appearance_variant, bool) or not isinstance(appearance_variant, int):
            raise ValueError(f"episode {episode_index} has invalid appearance_variant")
        episodes.append(
            {
                "episode_index": int(episode_index),
                "category": category,
                "lineage": lineage,
                "appearance_variant": appearance_variant,
            }
        )
        category_counts[category] += 1
    missing = sorted(category for category in weights if category_counts.get(category, 0) == 0)
    if missing:
        raise ValueError(
            f"condition {condition!r} has no training episodes for required categories: {missing}"
        )
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "condition": condition,
        "category_weights": dict(weights),
        "split_sha256": split_sha256,
        "eligible_episode_count": len(episodes),
        "category_episode_counts": dict(sorted(category_counts.items())),
        "episodes": sorted(episodes, key=lambda item: item["episode_index"]),
        "sampling_contract": {
            "category": "exact largest-remainder allocation per epoch",
            "lineage": "uniform within category",
            "appearance_variant": "uniform within physical trajectory lineage",
            "frame": "uniform within selected episode",
            "replacement": True,
        },
    }


@dataclass(frozen=True)
class _EpisodeRange:
    episode_index: int
    start: int
    length: int


def _allocated_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total <= 0:
        raise ValueError("sampling epoch must contain at least one frame")
    if not weights or any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
        raise ValueError("category weights must be finite and positive")
    weight_sum = sum(weights.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"category weights must sum to one, got {weight_sum}")
    raw = {name: total * value for name, value in weights.items()}
    result = {name: int(math.floor(value)) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(weights, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:remainder]:
        result[name] += 1
    return result


class LineageBalancedSampler:
    """Drop-in deterministic replacement for LeRobot's EpisodeAwareSampler.

    Every epoch has the same length as the eligible source frames, but samples
    with replacement using the plan's exact category ratio. Physical lineages
    and their appearance variants are balanced before a frame is selected.
    """

    def __init__(
        self,
        dataset_from_indices: Sequence[int],
        dataset_to_indices: Sequence[int],
        episode_indices_to_use: Sequence[int] | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
        seed: int = 0,
        absolute_to_relative_idx: Mapping[int, int] | None = None,
        *,
        plan_path: str | Path | None = None,
    ) -> None:
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("dropped frame counts must be non-negative")
        source = Path(plan_path or os.environ.get(PLAN_ENV, "")).expanduser()
        if not str(source) or not source.is_file():
            raise FileNotFoundError(f"lineage sampling plan is missing: {source}")
        plan = json.loads(source.read_text(encoding="utf-8"))
        if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported sampling plan schema: {plan.get('schema_version')!r}")
        weights = plan.get("category_weights")
        entries = plan.get("episodes")
        if not isinstance(weights, dict) or not isinstance(entries, list):
            raise ValueError("sampling plan lacks weights or episode records")

        starts = np.asarray(dataset_from_indices, dtype=np.int64)
        ends = np.asarray(dataset_to_indices, dtype=np.int64)
        if starts.shape != ends.shape:
            raise ValueError("dataset episode boundary arrays have different lengths")
        selected = (
            set(range(len(starts)))
            if episode_indices_to_use is None
            else {int(value) for value in episode_indices_to_use}
        )
        entry_by_episode: dict[int, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("episode_index"), int):
                raise ValueError("sampling plan contains an invalid episode record")
            episode_index = int(entry["episode_index"])
            if episode_index in entry_by_episode:
                raise ValueError(f"sampling plan repeats episode {episode_index}")
            entry_by_episode[episode_index] = entry

        by_category: dict[str, dict[str, list[_EpisodeRange]]] = defaultdict(
            lambda: defaultdict(list)
        )
        eligible_frames = 0
        for episode_index in sorted(selected):
            entry = entry_by_episode.get(episode_index)
            if entry is None:
                continue
            if episode_index < 0 or episode_index >= len(starts):
                raise ValueError(f"sampling plan episode is outside dataset: {episode_index}")
            start = int(starts[episode_index]) + drop_n_first_frames
            length = int(ends[episode_index]) - drop_n_last_frames - start
            if length <= 0:
                continue
            category = entry.get("category")
            lineage = entry.get("lineage")
            if category not in weights or not isinstance(lineage, str) or not lineage:
                raise ValueError(f"invalid sampling record for episode {episode_index}")
            by_category[category][lineage].append(
                _EpisodeRange(episode_index=episode_index, start=start, length=length)
            )
            eligible_frames += length
        missing = sorted(category for category in weights if not by_category.get(category))
        if missing:
            raise ValueError(f"selected dataset lacks required sampling categories: {missing}")

        self._weights = {str(key): float(value) for key, value in weights.items()}
        self._counts = _allocated_counts(eligible_frames, self._weights)
        self._groups = {
            category: {
                lineage: tuple(sorted(ranges, key=lambda item: item.episode_index))
                for lineage, ranges in sorted(lineages.items())
            }
            for category, lineages in sorted(by_category.items())
        }
        self._num_frames = eligible_frames
        self._absolute_to_relative = absolute_to_relative_idx
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._start_index = 0
        self.plan_path = source.resolve()

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        self._epoch = int(state["epoch"])
        self._start_index = int(state["start_index"])

    def __len__(self) -> int:
        return self._num_frames

    def __iter__(self) -> Iterator[int]:
        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        return self._iter_epoch(epoch, start)

    def _iter_epoch(self, epoch: int, start: int) -> Iterator[int]:
        if start < 0 or start > self._num_frames:
            raise ValueError(f"invalid sampler resume offset: {start}")
        seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        categories = tuple(sorted(self._counts))
        category_codes = np.concatenate(
            [np.full(self._counts[name], index, dtype=np.int16) for index, name in enumerate(categories)]
        )
        if self.shuffle:
            rng.shuffle(category_codes)

        lineage_orders: dict[str, np.ndarray] = {}
        lineage_positions: dict[str, int] = {}
        variant_orders: dict[tuple[str, str], np.ndarray] = {}
        variant_positions: dict[tuple[str, str], int] = {}
        lineage_names = {
            category: tuple(self._groups[category]) for category in categories
        }

        for position, code in enumerate(category_codes):
            category = categories[int(code)]
            names = lineage_names[category]
            lineage_position = lineage_positions.get(category, 0)
            if lineage_position % len(names) == 0:
                order = np.arange(len(names), dtype=np.int64)
                if self.shuffle:
                    rng.shuffle(order)
                lineage_orders[category] = order
            lineage_index = int(lineage_orders[category][lineage_position % len(names)])
            lineage_positions[category] = lineage_position + 1
            lineage = names[lineage_index]

            variants = self._groups[category][lineage]
            variant_key = (category, lineage)
            variant_position = variant_positions.get(variant_key, 0)
            if variant_position % len(variants) == 0:
                order = np.arange(len(variants), dtype=np.int64)
                if self.shuffle:
                    rng.shuffle(order)
                variant_orders[variant_key] = order
            variant_index = int(variant_orders[variant_key][variant_position % len(variants)])
            variant_positions[variant_key] = variant_position + 1
            episode = variants[variant_index]
            absolute_index = episode.start + int(rng.integers(0, episode.length))
            if position < start:
                continue
            if self._absolute_to_relative is None:
                yield absolute_index
            else:
                yield int(self._absolute_to_relative[absolute_index])

    @property
    def category_counts(self) -> dict[str, int]:
        return dict(self._counts)
