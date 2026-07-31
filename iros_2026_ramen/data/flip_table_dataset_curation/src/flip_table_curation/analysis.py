from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit import audit_source
from .clustering import (
    ClusterResult,
    cluster_descriptors,
    cluster_orientation_descriptors,
)
from .config import CurationConfig
from .orientation import OrientationSample, extract_orientation_samples
from .source import (
    VIDEO_KEYS,
    download_source,
    episode_slice,
    fixed_list_numpy,
    read_numeric_table,
)
from .trim import detect_trim, resample_trajectory
from .util import atomic_write_json, sha256_file
from .walking import FootKinematics, detect_steps


DECISION_SCHEMA = "team_ramen_flip_table_curation_decision/v1"
ANALYSIS_SCHEMA = "team_ramen_flip_table_curation_analysis/v1"


def _load_decision(config: CurationConfig) -> dict[str, Any] | None:
    path = config.workspace / "decision.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != DECISION_SCHEMA:
        raise ValueError("unsupported decision schema")
    if value.get("config_sha256") != config.digest:
        raise ValueError("decision belongs to a different curation config")
    return value


def write_decision(
    config: CurationConfig,
    *,
    orientation_clusters: list[int],
    trajectory_cluster: int,
    reviewer: str,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    analysis_path = config.workspace / "analysis" / "analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError("run analyze before recording a decision")
    analysis_sha = sha256_file(analysis_path)
    value = {
        "schema_version": DECISION_SCHEMA,
        "config_sha256": config.digest,
        "analysis_sha256": analysis_sha,
        "orientation_cluster_ids": sorted(set(int(value) for value in orientation_clusters)),
        "trajectory_cluster_id": int(trajectory_cluster),
        "reviewer": reviewer,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(config.workspace / "decision.json", value)
    return value


def _cluster_json(result: ClusterResult, episode_ids: list[int]) -> dict[str, Any]:
    return {
        "labels": {
            str(episode): int(label)
            for episode, label in zip(episode_ids, result.labels, strict=True)
        },
        "stability": {
            str(episode): float(value)
            for episode, value in zip(episode_ids, result.stability, strict=True)
        },
        "cluster_sizes": {str(key): value for key, value in result.cluster_sizes.items()},
        "largest_cluster": result.largest_cluster,
        "representatives": {
            str(label): [episode_ids[index] for index in indices]
            for label, indices in result.representatives.items()
        },
    }


def _split_assignments(records: list[dict[str, Any]], config: CurationConfig) -> None:
    selected = [record for record in records if record["curation_status"] == "accepted_auto"]
    seed = int(config.raw["seed"])
    ordered = sorted(
        selected,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['source_episode_name']}".encode()
        ).hexdigest(),
    )
    count = len(ordered)
    train_end = int(round(count * float(config.section("split")["train"])))
    validation_end = train_end + int(
        round(count * float(config.section("split")["validation"]))
    )
    for index, record in enumerate(ordered):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        record["split"] = split


def _reject_overlapping_source_video_frames(
    records: list[dict[str, Any]],
    *,
    source_rows: tuple[dict[str, Any], ...],
    fps: int,
    source_video_frame_counts: dict[str, int],
) -> None:
    """Keep every physical source video frame in at most one output episode."""

    rows = {int(row["episode_index"]): row for row in source_rows}
    occupied: dict[tuple[str, int], list[tuple[int, int]]] = {}
    candidates = sorted(
        (
            record
            for record in records
            if record["curation_status"] == "accepted_auto"
        ),
        key=lambda record: (
            -float(record["trajectory_stability"]),
            int(record["source_episode_index"]),
        ),
    )
    for record in candidates:
        source_episode = int(record["source_episode_index"])
        row = rows[source_episode]
        intervals: list[tuple[tuple[str, int], int, int]] = []
        conflicts: list[str] = []
        out_of_bounds: list[str] = []
        for key in VIDEO_KEYS:
            chunk_index = int(row[f"videos/{key}/chunk_index"])
            file_index = int(row[f"videos/{key}/file_index"])
            start = int(
                round(float(row[f"videos/{key}/from_timestamp"]) * fps)
            ) + int(record["trim_start"])
            end = start + int(record["trim_length"])
            identity = (key, file_index)
            intervals.append((identity, start, end))
            relative = (
                f"videos/{key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            )
            frame_count = source_video_frame_counts.get(relative)
            if frame_count is None:
                raise RuntimeError(
                    "source video decode inventory is required; "
                    "run `pixi run audit-source` first"
                )
            if start < 0 or end > frame_count:
                out_of_bounds.append(key)
            if any(
                start < used_end and used_start < end
                for used_start, used_end in occupied.get(identity, [])
            ):
                conflicts.append(key)
        if out_of_bounds:
            record["curation_status"] = "rejected"
            record["rejection_reasons"].append("source_video_frame_out_of_bounds")
            record["source_video_out_of_bounds_roles"] = sorted(out_of_bounds)
            record["source_video_overlap_roles"] = []
            continue
        record["source_video_out_of_bounds_roles"] = []
        if conflicts:
            record["curation_status"] = "rejected"
            record["rejection_reasons"].append("source_video_frame_overlap")
            record["source_video_overlap_roles"] = sorted(conflicts)
            continue
        record["source_video_overlap_roles"] = []
        for identity, start, end in intervals:
            occupied.setdefault(identity, []).append((start, end))


def analyze(config: CurationConfig) -> dict[str, Any]:
    source_audit = audit_source(config)
    snapshot = download_source(config, include_videos=True, rgb_only=True)
    table = read_numeric_table(snapshot)
    walking_cfg = config.section("walking")
    trim_cfg = config.section("trim")
    trajectory_cfg = config.section("trajectory")
    orientation_cfg = config.section("orientation")
    kinematics = FootKinematics(config.asset_path("urdf_path"))
    records: list[dict[str, Any]] = []
    trajectory_by_episode: dict[int, np.ndarray] = {}

    for position, row in enumerate(snapshot.episodes):
        episode = int(row["episode_index"])
        episode_table = episode_slice(table, row)
        state = fixed_list_numpy(
            episode_table["observation.state.robot_q_current"], 36
        )
        desired = fixed_list_numpy(episode_table["action.robot_q_desired"], 36)
        hands = fixed_list_numpy(episode_table["action.hand_cmd"], 2)
        trim = detect_trim(
            desired[:, 22:36],
            hands,
            fps=snapshot.fps,
            arm_velocity_threshold=float(trim_cfg["arm_velocity_threshold_rad_s"]),
            hand_velocity_threshold=float(trim_cfg["hand_velocity_threshold_units_s"]),
            persistence_window=int(trim_cfg["persistence_window_frames"]),
            persistence_required=int(trim_cfg["persistence_active_frames"]),
            pre_roll=int(trim_cfg["pre_roll_frames"]),
            post_roll=int(trim_cfg["post_roll_frames"]),
            minimum_frames=int(trim_cfg["minimum_frames"]),
            minimum_terminal_stable_frames=int(
                trim_cfg["minimum_terminal_stable_frames"]
            ),
        )
        steps = detect_steps(
            kinematics.positions(state),
            fps=snapshot.fps,
            median_window=int(walking_cfg["median_window_frames"]),
            floor_tolerance_m=float(walking_cfg["floor_tolerance_m"]),
            maximum_contact_speed_m_s=float(
                walking_cfg["maximum_contact_speed_m_s"]
            ),
            minimum_contact_seconds=float(walking_cfg["minimum_contact_seconds"]),
            step_displacement_m=float(walking_cfg["step_displacement_m"]),
        )
        if trim.valid:
            trajectory = np.concatenate(
                [desired[trim.start : trim.end, 22:36], hands[trim.start : trim.end]],
                axis=1,
            )
            trajectory_by_episode[episode] = resample_trajectory(
                trajectory, int(trajectory_cfg["resample_points"])
            ).ravel()
        records.append(
            {
                "source_episode_index": episode,
                "source_episode_name": str(row["source_episode_name"]),
                "source_length": int(row["length"]),
                "trim_valid": trim.valid,
                "trim_reason": trim.reason,
                "trim_start": trim.start,
                "trim_end": trim.end,
                "trim_length": trim.length,
                "first_active": trim.first_active,
                "last_active": trim.last_active,
                "post_roll_complete": trim.post_roll_complete,
                "walked": steps.walked,
                "step_count": steps.step_count,
                "left_step_count": steps.left_step_count,
                "right_step_count": steps.right_step_count,
                "maximum_contact_displacement_m": steps.maximum_contact_displacement_m,
                "orientation_valid": False,
                "orientation_cluster": -1,
                "orientation_detection_fraction": 0.0,
                "trajectory_cluster": -1,
                "trajectory_stability": 0.0,
                "curation_status": "rejected",
                "rejection_reasons": [],
                "split": None,
            }
        )
        if (position + 1) % 50 == 0:
            print(f"[numeric] {position + 1}/{len(snapshot.episodes)} episodes")

    record_by_episode = {
        int(record["source_episode_index"]): record for record in records
    }
    orientation_candidates = [
        row
        for row in snapshot.episodes
        if (
            not record_by_episode[int(row["episode_index"])]["walked"]
            and record_by_episode[int(row["episode_index"])]["trim_valid"]
        )
    ]
    samples = extract_orientation_samples(
        snapshot,
        rows=orientation_candidates,
        weight_path=config.asset_path("yolo_weight_path"),
        output_dir=config.workspace / "review" / "thumbnails",
        sample_frames=int(orientation_cfg["sample_frames"]),
        sample_window_frames=int(orientation_cfg["sample_window_frames"]),
        confidence=float(orientation_cfg["confidence"]),
        minimum_detection_fraction=float(
            orientation_cfg["minimum_detection_fraction"]
        ),
    )
    valid_orientation = [sample for sample in samples if sample.valid]
    if len(valid_orientation) < int(orientation_cfg["min_cluster_size"]):
        raise RuntimeError("too few valid orientation descriptors")
    orientation_result = cluster_orientation_descriptors(
        np.asarray([sample.descriptor for sample in valid_orientation]),
        seed=int(config.raw["seed"]),
    )
    orientation_ids = [sample.episode_index for sample in valid_orientation]
    for sample in samples:
        record = record_by_episode[sample.episode_index]
        record["orientation_valid"] = sample.valid
        record["orientation_detection_fraction"] = sample.detection_fraction
    for episode, label in zip(
        orientation_ids, orientation_result.labels, strict=True
    ):
        record_by_episode[episode]["orientation_cluster"] = int(label)

    decision = _load_decision(config)
    chosen_orientations = (
        [int(value) for value in decision["orientation_cluster_ids"]]
        if decision
        else (
            [orientation_result.largest_cluster]
            if orientation_result.largest_cluster is not None
            else []
        )
    )
    trajectory_ids = [
        int(record["source_episode_index"])
        for record in records
        if (
            record["orientation_cluster"] in chosen_orientations
            and not record["walked"]
            and record["trim_valid"]
        )
    ]
    min_cluster_size = max(
        int(trajectory_cfg["min_cluster_size_floor"]),
        int(np.ceil(len(trajectory_ids) * float(trajectory_cfg["min_cluster_fraction"]))),
    )
    trajectory_result = cluster_descriptors(
        np.asarray([trajectory_by_episode[episode] for episode in trajectory_ids]),
        min_cluster_size=min_cluster_size,
        min_samples=int(trajectory_cfg["min_samples"]),
        seed=int(config.raw["seed"]),
        pca_variance=float(trajectory_cfg["pca_variance"]),
        pca_max_components=int(trajectory_cfg["pca_max_components"]),
        stability_runs=int(trajectory_cfg["bootstrap_runs"]),
        stability_fraction=float(trajectory_cfg["bootstrap_fraction"]),
        cluster_selection_epsilon_quantile=float(
            trajectory_cfg["cluster_selection_epsilon_quantile"]
        ),
    )
    for episode, label, stability in zip(
        trajectory_ids,
        trajectory_result.labels,
        trajectory_result.stability,
        strict=True,
    ):
        record = record_by_episode[episode]
        record["trajectory_cluster"] = int(label)
        record["trajectory_stability"] = float(stability)

    chosen_trajectory = (
        int(decision["trajectory_cluster_id"])
        if decision
        else trajectory_result.largest_cluster
    )
    minimum_stability = float(trajectory_cfg["minimum_stability"])
    for record in records:
        reasons: list[str] = []
        if not record["trim_valid"]:
            reasons.append(str(record["trim_reason"]))
        if record["walked"]:
            reasons.append("walking_detected")
        if not record["orientation_valid"]:
            reasons.append("orientation_unresolved")
        elif record["orientation_cluster"] not in chosen_orientations:
            reasons.append("orientation_not_selected")
        if record["trajectory_cluster"] != chosen_trajectory:
            reasons.append("trajectory_not_selected")
        elif record["trajectory_stability"] < minimum_stability:
            reasons.append("trajectory_unstable")
        record["rejection_reasons"] = reasons
        if decision and not reasons:
            record["curation_status"] = "accepted_auto"
    _reject_overlapping_source_video_frames(
        records,
        source_rows=snapshot.episodes,
        fps=snapshot.fps,
        source_video_frame_counts={
            relative: int(metadata["decoded_frames"])
            for relative, metadata in source_audit["video_decode"]["files"].items()
        },
    )
    _split_assignments(records, config)
    accepted = sum(record["curation_status"] == "accepted_auto" for record in records)
    report = {
        "schema_version": ANALYSIS_SCHEMA,
        "config_sha256": config.digest,
        "code_sha256": config.code_digest,
        "source_repo_id": config.source_repo_id,
        "source_revision": config.source_revision,
        "source_episodes": len(records),
        "decision": decision,
        "suggested_decision": {
            "orientation_cluster_ids": chosen_orientations,
            "trajectory_cluster_id": trajectory_result.largest_cluster,
        },
        "orientation": _cluster_json(orientation_result, orientation_ids),
        "trajectory": _cluster_json(trajectory_result, trajectory_ids),
        "records": records,
        "summary": {
            "trim_invalid": sum(not record["trim_valid"] for record in records),
            "walking_detected": sum(record["walked"] for record in records),
            "source_video_overlap_rejected": sum(
                "source_video_frame_overlap" in record["rejection_reasons"]
                for record in records
            ),
            "source_video_out_of_bounds_rejected": sum(
                "source_video_frame_out_of_bounds" in record["rejection_reasons"]
                for record in records
            ),
            "orientation_unresolved": sum(
                not record["orientation_valid"] for record in records
            ),
            "accepted": accepted,
            "rejected": len(records) - accepted,
        },
    }
    output = config.workspace / "analysis" / "analysis.json"
    atomic_write_json(output, report)
    print(f"[analysis] report={output} accepted={accepted}")
    return report
