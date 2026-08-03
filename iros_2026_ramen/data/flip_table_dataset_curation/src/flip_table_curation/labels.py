from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from .config import CurationConfig
from .util import atomic_write_json, sha256_file


LABEL_COLUMNS = (
    "episode_id",
    "frame_start",
    "frame_end",
    "task_index",
    "verdict",
    "failure_category",
    "reviewer",
    "reviewed_at",
    "schema_version",
)
ACCEPTED_VERDICTS = {"optimal", "success"}
VALID_VERDICTS = ACCEPTED_VERDICTS | {"ambiguous", "failure"}
SELECTION_SCHEMA = "team_ramen_manual_flip_table_selection/v1"


@dataclass(frozen=True)
class SelectedSegment:
    source_episode_index: int
    source_frame_start: int
    source_frame_end: int
    verdict: str
    reviewer: str
    reviewed_at: str
    label_row: dict[str, Any]
    split: str

    @property
    def length(self) -> int:
        return self.source_frame_end - self.source_frame_start + 1


def selection_path(config: CurationConfig) -> Path:
    return config.workspace / "selection" / "selected_segments.json"


def _load_labels(config: CurationConfig) -> tuple[pd.DataFrame, Path]:
    path = Path(
        hf_hub_download(
            repo_id=config.labels_repo_id,
            filename="labels.parquet",
            repo_type="dataset",
            revision=config.labels_revision,
            cache_dir=config.workspace / "labels_cache",
        )
    )
    return pd.read_parquet(path), path


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _validate_labels(
    labels: pd.DataFrame,
    *,
    source_lengths: dict[int, int],
    expected_schema_version: int,
) -> None:
    if tuple(labels.columns) != LABEL_COLUMNS:
        raise ValueError(f"unexpected label columns: {list(labels.columns)}")
    if labels.empty:
        raise ValueError("curation labels are empty")
    if labels.isnull().any().any():
        nullable = {name: int(value) for name, value in labels.isnull().sum().items() if value}
        allowed = {"failure_category"}
        if set(nullable) - allowed:
            raise ValueError(f"unexpected null label fields: {nullable}")
    if set(labels["schema_version"].unique()) != {expected_schema_version}:
        raise ValueError("curation label schema version mismatch")
    if not set(labels["verdict"].unique()).issubset(VALID_VERDICTS):
        raise ValueError("curation labels contain an unknown verdict")
    if labels.duplicated(["episode_id", "frame_start"]).any():
        raise ValueError("curation labels have duplicate primary keys")
    invalid = labels[(labels["frame_start"] < 0) | (labels["frame_end"] < labels["frame_start"])]
    if not invalid.empty:
        raise ValueError("curation labels contain invalid frame ranges")
    unknown_episodes = sorted(set(labels["episode_id"].astype(int)) - set(source_lengths))
    if unknown_episodes:
        raise ValueError(f"curation labels refer to unknown source episodes: {unknown_episodes[:10]}")
    out_of_bounds = labels[
        labels.apply(
            lambda row: int(row["frame_end"]) >= source_lengths[int(row["episode_id"])],
            axis=1,
        )
    ]
    if not out_of_bounds.empty:
        raise ValueError("curation labels contain out-of-bounds frame ranges")
    failed_without_category = labels[
        (labels["verdict"] == "failure") & labels["failure_category"].isna()
    ]
    if not failed_without_category.empty:
        raise ValueError("failure labels must carry failure_category")


def _split(source_episode_index: int, *, seed: int, count: int, rank: int) -> str:
    # Kept separate for testability. Selection order has already been made stable
    # by the caller's SHA-256 order and rank is therefore reproducible.
    del source_episode_index, seed
    train_end = round(count * 0.80)
    validation_end = train_end + round(count * 0.10)
    if rank < train_end:
        return "train"
    if rank < validation_end:
        return "validation"
    return "test"


def _final_accepted_flip_labels(labels: pd.DataFrame, task_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    flip = labels[labels["task_index"] == task_index].sort_values(["episode_id", "frame_start"])
    last = flip.groupby("episode_id", as_index=False).tail(1)
    return flip, last[last["verdict"].isin(ACCEPTED_VERDICTS)].copy()


def _manual_exclusions(config: CurationConfig) -> dict[tuple[int, int, int], str]:
    raw = config.raw.get("manual_exclusions", [])
    if not isinstance(raw, list):
        raise ValueError("manual_exclusions must be a list")
    exclusions: dict[tuple[int, int, int], str] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("manual_exclusions entries must be mappings")
        try:
            key = (
                int(value["source_episode_index"]),
                int(value["source_frame_start"]),
                int(value["source_frame_end"]),
            )
            reason = str(value["reason"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("manual exclusion has an invalid source segment") from error
        if key[1] < 0 or key[2] < key[1] or not reason:
            raise ValueError("manual exclusion has an invalid range or reason")
        if key in exclusions:
            raise ValueError(f"manual exclusion is duplicated: {key}")
        exclusions[key] = reason
    return exclusions


def select_segments(
    config: CurationConfig,
    *,
    source_lengths: dict[int, int],
) -> tuple[list[SelectedSegment], dict[str, Any]]:
    import hashlib

    labels, labels_path = _load_labels(config)
    labels_config = config.section("labels")
    _validate_labels(
        labels,
        source_lengths=source_lengths,
        expected_schema_version=int(labels_config["schema_version"]),
    )
    task_index = int(labels_config["task_index"])
    flip, accepted = _final_accepted_flip_labels(labels, task_index)
    final_labels = flip.groupby("episode_id", as_index=False).tail(1)
    if accepted.empty:
        raise ValueError("no final successful flip_table labels")

    # A selected final segment must not compete with another annotation. We do
    # not guess which row the reviewer intended; the UI must resolve it first.
    conflicts: list[dict[str, Any]] = []
    for row in accepted.itertuples(index=False):
        same_episode = labels[labels["episode_id"] == row.episode_id]
        overlap = same_episode[
            (same_episode["frame_start"] <= row.frame_end)
            & (same_episode["frame_end"] >= row.frame_start)
            & (same_episode["frame_start"] != row.frame_start)
        ]
        if not overlap.empty:
            conflicts.append(
                {
                    "episode_id": int(row.episode_id),
                    "selected": {
                        "frame_start": int(row.frame_start),
                        "frame_end": int(row.frame_end),
                        "verdict": str(row.verdict),
                    },
                    "overlapping_rows": overlap[list(LABEL_COLUMNS)].to_dict("records"),
                }
            )
    if conflicts:
        raise ValueError(
            "selected final flip_table labels overlap another curation label; "
            f"manual resolution is required: {json.dumps(conflicts[:3], ensure_ascii=False)}"
        )

    manual_exclusions = _manual_exclusions(config)
    accepted_keys = {
        (int(row.episode_id), int(row.frame_start), int(row.frame_end))
        for row in accepted.itertuples(index=False)
    }
    unknown_exclusions = sorted(set(manual_exclusions) - accepted_keys)
    if unknown_exclusions:
        raise ValueError(
            "manual exclusion does not match an accepted final flip_table segment: "
            f"{unknown_exclusions[:3]}"
        )
    user_excluded = []
    if manual_exclusions:
        for row in accepted.itertuples(index=False):
            key = (int(row.episode_id), int(row.frame_start), int(row.frame_end))
            if key in manual_exclusions:
                user_excluded.append(
                    {
                        "source_episode_index": key[0],
                        "frame_start": key[1],
                        "frame_end": key[2],
                        "reason": manual_exclusions[key],
                        "label_row": {
                            name: _json_value(getattr(row, name))
                            for name in LABEL_COLUMNS
                        },
                    }
                )
        accepted = accepted[
            ~accepted.apply(
                lambda row: (
                    int(row["episode_id"]),
                    int(row["frame_start"]),
                    int(row["frame_end"]),
                )
                in manual_exclusions,
                axis=1,
            )
        ].copy()
    if accepted.empty:
        raise ValueError("manual exclusions removed every accepted flip_table segment")

    last_by_episode = {int(row.episode_id): row for row in flip.groupby("episode_id", as_index=False).tail(1).itertuples(index=False)}
    excluded = []
    for episode_id in sorted(source_lengths):
        final = last_by_episode.get(episode_id)
        if final is None:
            excluded.append({"source_episode_index": episode_id, "reason": "no_flip_table_label"})
        elif final.verdict not in ACCEPTED_VERDICTS:
            excluded.append({
                "source_episode_index": episode_id,
                "reason": f"final_flip_table_verdict_{final.verdict}",
                "frame_start": int(final.frame_start),
                "frame_end": int(final.frame_end),
            })

    seed = int(config.raw["seed"])
    accepted["sort_key"] = accepted["episode_id"].map(
        lambda episode: hashlib.sha256(f"{seed}:{int(episode)}".encode()).hexdigest()
    )
    accepted = accepted.sort_values("sort_key").reset_index(drop=True)
    selected = [
        SelectedSegment(
            source_episode_index=int(row.episode_id),
            source_frame_start=int(row.frame_start),
            source_frame_end=int(row.frame_end),
            verdict=str(row.verdict),
            reviewer=str(row.reviewer),
            reviewed_at=str(row.reviewed_at),
            label_row={name: _json_value(getattr(row, name)) for name in LABEL_COLUMNS},
            split=_split(int(row.episode_id), seed=seed, count=len(accepted), rank=rank),
        )
        for rank, row in enumerate(accepted.itertuples(index=False))
    ]
    selected.sort(key=lambda row: ({"train": 0, "validation": 1, "test": 2}[row.split], row.source_episode_index))
    report = {
        "schema_version": SELECTION_SCHEMA,
        "config_sha256": config.digest,
        "source_repo_id": config.source_repo_id,
        "source_revision": config.source_revision,
        "labels_repo_id": config.labels_repo_id,
        "labels_revision": config.labels_revision,
        "labels_sha256": sha256_file(labels_path),
        "selection_rule": "last_flip_table_label_is_success_or_optimal",
        "total_labels": int(len(labels)),
        "flip_table_labels": int(len(flip)),
        "selected_segments": len(selected),
        "selected_frames": sum(row.length for row in selected),
        "excluded_segments": excluded + user_excluded,
        "user_excluded_segments": user_excluded,
        "rejected_final_flip_verdicts": {
            str(key): int(value)
            for key, value in final_labels[~final_labels["verdict"].isin(ACCEPTED_VERDICTS)]["verdict"].value_counts().items()
        },
        "segments": [
            {
                "source_episode_index": row.source_episode_index,
                "source_frame_start": row.source_frame_start,
                "source_frame_end": row.source_frame_end,
                "length": row.length,
                "verdict": row.verdict,
                "reviewer": row.reviewer,
                "reviewed_at": row.reviewed_at,
                "split": row.split,
                "label_row": row.label_row,
            }
            for row in selected
        ],
    }
    return selected, report


def write_selection(config: CurationConfig, *, source_lengths: dict[int, int]) -> dict[str, Any]:
    selected, report = select_segments(config, source_lengths=source_lengths)
    path = selection_path(config)
    atomic_write_json(path, report)
    print(f"[selection] {len(selected)} accepted final flip_table segments -> {path}")
    return report


def load_selection(config: CurationConfig) -> tuple[list[SelectedSegment], dict[str, Any]]:
    path = selection_path(config)
    if not path.is_file():
        raise FileNotFoundError("run `audit-labels` before build")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unsupported selection schema")
    if report.get("config_sha256") != config.digest:
        raise ValueError("selection belongs to a different config")
    selected = [
        SelectedSegment(
            source_episode_index=int(row["source_episode_index"]),
            source_frame_start=int(row["source_frame_start"]),
            source_frame_end=int(row["source_frame_end"]),
            verdict=str(row["verdict"]),
            reviewer=str(row["reviewer"]),
            reviewed_at=str(row["reviewed_at"]),
            label_row=dict(row["label_row"]),
            split=str(row["split"]),
        )
        for row in report["segments"]
    ]
    return selected, report
