#!/usr/bin/env python3
"""Create a pinned, auditable calibration bundle from the real flip-table data.

This command is intentionally CPU-only.  It prepares the exact real evidence
and 16-D arm/hand replay actions consumed by the WBC simulator runner; physics
fitting and image matching run later on the RTX5090 against this immutable
bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file
from .contracts import (
    CALIBRATION_CAMERA_KEYS,
    CAMERA_ROLES,
    SCHEMA_VERSION,
    SOURCE_FPS,
    SOURCE_EEF_DIM,
    SOURCE_EEF_ORDER,
    SOURCE_EEF_POSE_FORMAT,
    SOURCE_EEF_REFERENCE_FRAME,
    SOURCE_REPO_ID,
    SOURCE_REVISION,
    SOURCE_Q_DIM,
    EpisodeSelection,
    EpisodeSignals,
    episode_signals,
    select_episode_roles,
    source_16d_actions,
    source_19d_observation,
    source_31d_actions,
    source_31d_observation,
)


REQUIRED_COLUMNS = (
    "episode_index",
    "frame_index",
    "timestamp",
    "observation.state.robot_q_current",
    "action.robot_q_desired",
    "action.hand_cmd",
    "observation.state.ee_state",
    "action.ee_action",
    "observation.state.hand_state",
)


def _load_pyarrow():
    try:
        import pyarrow.dataset as pads
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required; use the project training environment") from exc
    return pads


def _dataset_table(dataset_root: Path):
    pads = _load_pyarrow()
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(dataset_root / "data")
    return pads.dataset([str(path) for path in paths], format="parquet").to_table(columns=list(REQUIRED_COLUMNS))


def _as_matrix(table: Any, key: str, width: int) -> np.ndarray:
    values = np.asarray(table[key].to_pylist(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError(f"{key} must be finite [N,{width}], got {values.shape}")
    return values


def audit_numeric_contract(dataset_root: Path) -> tuple[list[EpisodeSignals], dict[str, object], Any]:
    """Validate all frames before selecting any calibration episode."""

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3"):
        raise ValueError("dataset must be LeRobotDataset v3")
    if float(info.get("fps", 0.0)) != SOURCE_FPS:
        raise ValueError(f"dataset fps must be {SOURCE_FPS}")
    features = info.get("features", {})
    expected_shapes = {
        "observation.state.robot_q_current": [36],
        "action.robot_q_desired": [36],
        "action.hand_cmd": [2],
        "observation.state.ee_state": [SOURCE_EEF_DIM],
        "action.ee_action": [SOURCE_EEF_DIM],
        "observation.state.hand_state": [2],
    }
    for key, shape in expected_shapes.items():
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("dtype") != "float32" or feature.get("shape") != shape:
            raise ValueError(f"invalid feature contract for {key}: {feature!r}")

    table = _dataset_table(dataset_root)
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    q_current = _as_matrix(table, "observation.state.robot_q_current", SOURCE_Q_DIM)
    q_desired = _as_matrix(table, "action.robot_q_desired", SOURCE_Q_DIM)
    hand_cmd = _as_matrix(table, "action.hand_cmd", 2)
    ee_state = _as_matrix(table, "observation.state.ee_state", SOURCE_EEF_DIM)
    ee_action = _as_matrix(table, "action.ee_action", SOURCE_EEF_DIM)
    _as_matrix(table, "observation.state.hand_state", 2)
    if not np.isfinite(timestamp).all():
        raise ValueError("timestamp contains NaN or Inf")
    if np.any(hand_cmd < 0.0) or np.any(hand_cmd > 4.5):
        raise ValueError("hand_cmd is outside [0,4.5]")
    episodes = np.unique(episode_index)
    if episodes.tolist() != list(range(len(episodes))):
        raise ValueError("episode_index must be contiguous from zero")
    signals: list[EpisodeSignals] = []
    for episode in episodes:
        rows = np.flatnonzero(episode_index == episode)
        if frame_index[rows].tolist() != list(range(len(rows))):
            raise ValueError(f"episode {episode} has non-contiguous frame indices")
        signals.append(episode_signals(int(episode), timestamp[rows], q_desired[rows], hand_cmd[rows]))
    report = {
        "dataset_root": str(dataset_root),
        "episodes": int(len(episodes)),
        "frames": int(len(episode_index)),
        "fps": SOURCE_FPS,
        "all_numeric_features_finite": True,
        "robot_q_current_max_abs_rad": float(np.abs(q_current[:, 7:]).max()),
        "robot_q_desired_max_abs_rad": float(np.abs(q_desired[:, 7:]).max()),
        "eef_contract": {
            "dimension": SOURCE_EEF_DIM,
            "order": list(SOURCE_EEF_ORDER),
            "pose_format": SOURCE_EEF_POSE_FORMAT,
            "reference_frame": SOURCE_EEF_REFERENCE_FRAME,
            "row_timestamp_alignment": "state/action/EE labels share each source parquet row timestamp",
        },
        "ee_state_max_abs": float(np.abs(ee_state).max()),
        "ee_action_max_abs": float(np.abs(ee_action).max()),
        "signals": [item.json() for item in signals],
    }
    return signals, report, table


def _episodes_metadata(dataset_root: Path, indices: Iterable[int]) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required; use the project training environment") from exc
    expected = {int(index) for index in indices}
    result: dict[int, dict[str, Any]] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, filters=[("episode_index", "in", sorted(expected))])
        for row in table.to_pylist():
            result[int(row["episode_index"])] = row
    if set(result) != expected:
        raise ValueError(f"missing episode metadata: {sorted(expected - set(result))}")
    return result


def _pinned_hf_url(repo_id: str, revision: str, relative_path: str) -> str:
    return (
        "https://huggingface.co/datasets/"
        f"{repo_id}/resolve/{revision}/{quote(relative_path, safe='/')}"
    )


def _download_pinned_file(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        return
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urlopen(url, timeout=60) as response:  # nosec B310 - URL is pinned above
        payload = response.read()
    temporary.write_bytes(payload)
    if sha256_file(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded calibration digest differs from the pinned contract: {url}")
    temporary.replace(destination)


def _raw_camera_provenance(
    config_path: Path,
    raw_root: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    config = load_pipeline_config(config_path)
    head_relative = config.raw_source.head_stereo_calibration_repo_path
    calibration_relative_root = str(Path(head_relative).parent)
    output_calibration = output_dir / "calibration"
    local_paths: dict[str, str] = {}
    if raw_root is not None:
        root = raw_root.expanduser().resolve()
        head_source = root / head_relative
        if not head_source.is_file():
            raise FileNotFoundError(head_source)
        head_destination = output_calibration / "head_camera_params.yaml"
        head_destination.parent.mkdir(parents=True, exist_ok=True)
        head_destination.write_bytes(head_source.read_bytes())
    else:
        root = None
        head_destination = output_calibration / "head_camera_params.yaml"
        _download_pinned_file(
            _pinned_hf_url(config.raw_source.repo_id, config.raw_source.revision, head_relative),
            head_destination,
            config.raw_source.head_stereo_calibration_sha256,
        )
    if sha256_file(head_destination) != config.raw_source.head_stereo_calibration_sha256:
        raise ValueError("head stereo calibration digest differs from the pinned contract")
    local_paths["head_stereo"] = str(head_destination)
    for serial, digest in sorted(config.raw_source.wrist_calibration_sha256_by_serial.items()):
        relative = f"{calibration_relative_root}/camera_{serial}.json"
        destination = output_calibration / f"camera_{serial}.json"
        if root is not None:
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            destination.write_bytes(source.read_bytes())
        else:
            _download_pinned_file(
                _pinned_hf_url(config.raw_source.repo_id, config.raw_source.revision, relative),
                destination,
                digest,
            )
        if sha256_file(destination) != digest:
            raise ValueError(f"D405 calibration digest differs from the pinned contract: {serial}")
        local_paths[f"d405_{serial}"] = str(destination)
    return {
        "raw_source_repo_id": config.raw_source.repo_id,
        "raw_source_revision": config.raw_source.revision,
        "raw_root": str(root) if root is not None else None,
        "pinned_calibration_files": local_paths,
        "head_stereo_baseline_m": config.raw_source.head_stereo_baseline_m,
        "head_stereo_rms_error_px": config.raw_source.head_stereo_rms_error_px,
        "head_stereo_calibration_sha256": config.raw_source.head_stereo_calibration_sha256,
        "head_stereo_calibration_repo_path": config.raw_source.head_stereo_calibration_repo_path,
        "wrist_d405_calibration_sha256_by_serial": config.raw_source.wrist_calibration_sha256_by_serial,
        "camera_roles": CAMERA_ROLES,
        "calibration_camera_keys": list(CALIBRATION_CAMERA_KEYS),
        "note": (
            "Intrinsics and head-stereo baseline are measured raw-MCAP calibration facts. "
            "Robot-link extrinsics, table pose, and contact values remain inferred."
        ),
    }


def _episode_bundle(table: Any, episode: int, episode_metadata: dict[str, Any]) -> dict[str, Any]:
    rows = np.flatnonzero(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64) == episode)
    q_current = np.asarray(table["observation.state.robot_q_current"].to_pylist(), dtype=np.float64)[rows]
    q_desired = np.asarray(table["action.robot_q_desired"].to_pylist(), dtype=np.float64)[rows]
    ee_state = np.asarray(table["observation.state.ee_state"].to_pylist(), dtype=np.float64)[rows]
    ee_action = np.asarray(table["action.ee_action"].to_pylist(), dtype=np.float64)[rows]
    hand_state = np.asarray(table["observation.state.hand_state"].to_pylist(), dtype=np.float64)[rows]
    hand_cmd = np.asarray(table["action.hand_cmd"].to_pylist(), dtype=np.float64)[rows]
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)[rows]
    action_16d = source_16d_actions(q_desired, hand_cmd)
    action_31d = source_31d_actions(q_desired, hand_cmd)
    observed_state_19d = source_19d_observation(q_current, hand_state)
    observed_state_31d = source_31d_observation(q_current, hand_state)
    return {
        "source_episode_index": int(episode),
        "source_episode_name": episode_metadata.get("source_episode_name"),
        "fps": SOURCE_FPS,
        "timestamps_s": timestamps.tolist(),
        # LeRobot source layout is position xyz followed by quaternion wxyz.
        # This is a reference record only: the fixed-base diagnostic never
        # writes it as a per-frame simulator root pose.
        "initial_root_pose_xyz_wxyz": q_current[0, :7].tolist(),
        "observed_root_pose_xyz_wxyz": q_current[:, :7].tolist(),
        "initial_body_joint_position_rad": q_current[0, 7:].tolist(),
        "recorded_root_pose_reference_xyz_wxyz": q_desired[:, :7].tolist(),
        "recorded_full_body_target_rad": q_desired[:, 7:].tolist(),
        "recorded_arm_hand_target_16d": action_16d.tolist(),
        "recorded_full_body_hand_target_31d": action_31d.tolist(),
        # These EEF streams are a second label for the joint state/action and
        # are retained for FK and time-alignment diagnostics. The fixed-base
        # simulator continues to replay only the recorded joint targets.
        "observed_ee_state_xyz_euler_xyz_rad": ee_state.tolist(),
        "recorded_ee_action_xyz_euler_xyz_rad": ee_action.tolist(),
        "eef_layout": {
            "dimension": SOURCE_EEF_DIM,
            "order": list(SOURCE_EEF_ORDER),
            "pose_format": SOURCE_EEF_POSE_FORMAT,
            "reference_frame": SOURCE_EEF_REFERENCE_FRAME,
            "use": "offline FK/time-alignment diagnostic only; never a replay controller input",
        },
        # The camera at source frame zero observed q_current/hand_state, not
        # the controller's q_desired/hand_cmd.  A calibration replay must
        # start from this measured state, then send the first command at its
        # recorded timestamp.  Treating q_desired as the reset pose erases
        # real controller lag and corrupts camera/extrinsic fitting.
        "observed_upper_body_state_and_hand_state": observed_state_19d.tolist(),
        "observed_full_body_state_and_hand_state": observed_state_31d.tolist(),
        "state_layout": "waist_3_left_arm_7_right_arm_7_left_hand_state_right_hand_state",
        "action_layout": "left_arm_7_right_arm_7_left_hand_cmd_right_hand_cmd",
        "state_dim": 19,
        "action_dim": 16,
        "full_body_diagnostic_state_layout": "left_leg_6_right_leg_6_waist_3_left_arm_7_right_arm_7_left_hand_state_right_hand_state",
        "full_body_diagnostic_action_layout": "left_leg_6_right_leg_6_waist_3_left_arm_7_right_arm_7_left_hand_cmd_right_hand_cmd",
        "full_body_diagnostic_dim": 31,
        "root_replay": "forbidden_per_frame; reference_and_initialization_only",
        "lower_body_replay": "retained_for_full_body_diagnostic_only; production_balanced_wbc_replay_uses_arms14_only",
        "policy_camera_keys": list(CALIBRATION_CAMERA_KEYS),
        "video_metadata": {key: value for key, value in episode_metadata.items() if str(key).startswith("videos/")},
    }


def _eef_fk_eligible_episode_indices(path: Path | None) -> tuple[set[int] | None, dict[str, object]]:
    """Load a prior all-episode EEF/FK audit without accepting its fitted values.

    The audit only excludes source episodes whose dual joint/EEF labels are
    inconsistent. Its fitted tool transform is deliberately not imported into
    the simulator by this preparation step.
    """

    if path is None:
        return None, {"status": "not_supplied", "selection_constraint": "numeric_only_pending_eef_fk_audit"}
    source = path.expanduser().resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EEF/FK audit must be a JSON object")
    if value.get("eef_pose_format") != SOURCE_EEF_POSE_FORMAT:
        raise ValueError("EEF/FK audit pose format differs from the source contract")
    if value.get("configured_eef_order") != list(SOURCE_EEF_ORDER):
        raise ValueError("EEF/FK audit left/right order differs from the source contract")
    if value.get("frame_assignment_pass") is not True:
        raise ValueError("EEF/FK audit does not validate the left/right EEF assignment")
    gate = value.get("mimic_source_episode_gate")
    if not isinstance(gate, dict):
        raise ValueError("EEF/FK audit is missing its per-episode eligibility gate")
    raw_indices = gate.get("eligible_episode_indices")
    if not isinstance(raw_indices, list):
        raise ValueError("EEF/FK audit eligible_episode_indices must be a list")
    try:
        indices = {int(index) for index in raw_indices}
    except (TypeError, ValueError) as exc:
        raise ValueError("EEF/FK audit contains a non-integer episode index") from exc
    if len(indices) < 8:
        raise ValueError("EEF/FK audit leaves fewer than eight eligible episodes")
    return indices, {
        "status": "applied",
        "audit_path": str(source),
        "audit_sha256": sha256_file(source),
        "eligible_episode_count": len(indices),
        "rejected_episode_count": int(gate.get("rejected_count", 0)),
        "selection_constraint": "source EEF/FK action residual gate",
        "fitted_tool_transforms_imported": False,
    }


def _selection_from_override(
    path: Path,
    *,
    available_episode_indices: set[int],
    eligible_episode_indices: set[int] | None,
) -> tuple[EpisodeSelection, dict[str, object]]:
    """Load a reviewed visual-selection result without bypassing the audit.

    Numeric activity alone cannot prove that a pre-contact table is visible in
    both head cameras.  A reviewed override is therefore allowed only after
    the full numeric audit above, only for eight already-audited episodes, and
    only when it records the rejected candidate and the direct-CAD evidence
    that replaced it.
    """

    source = path.expanduser().resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema_version") != "team_ramen_flip_table_selection_override/v1":
        raise ValueError("selection override has an unsupported schema")
    selection_value = document.get("selection")
    if not isinstance(selection_value, dict):
        raise ValueError("selection override lacks selection")
    try:
        selection = EpisodeSelection(
            anchor=int(selection_value["anchor"]),
            calibration=tuple(int(value) for value in selection_value["calibration"]),
            validation=tuple(int(value) for value in selection_value["validation"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selection override has invalid episode indices") from exc
    if len(selection.calibration) != 2 or len(selection.validation) != 5:
        raise ValueError("selection override must contain one anchor, two calibration, and five validation episodes")
    selected = set(selection.all_indices())
    if len(selected) != 8:
        raise ValueError("selection override episode roles must be disjoint")
    if not selected <= available_episode_indices:
        raise ValueError("selection override references an episode outside the audited dataset")
    if eligible_episode_indices is not None and not selected <= eligible_episode_indices:
        raise ValueError("selection override references an EEF/FK-ineligible episode")
    rationale = document.get("rationale")
    visual_evidence = document.get("visual_evidence")
    if not isinstance(rationale, str) or not rationale.strip() or not isinstance(visual_evidence, list):
        raise ValueError("selection override requires rationale and visual_evidence")
    return selection, {
        "mode": "reviewed_visual_evidence_override",
        "path": str(source),
        "sha256": sha256_file(source),
        "rationale": rationale,
        "visual_evidence": visual_evidence,
        "numeric_audit_still_executed": True,
        "eef_fk_eligibility_still_enforced": eligible_episode_indices is not None,
    }


def prepare_bundle(
    dataset_root: Path,
    output_dir: Path,
    *,
    config_path: Path,
    raw_root: Path | None,
    eef_fk_audit: Path | None = None,
    selection_override: Path | None = None,
) -> dict[str, Any]:
    source = dataset_root.expanduser().resolve()
    signals, numeric_audit, table = audit_numeric_contract(source)
    eligible_episode_indices, eef_fk_selection = _eef_fk_eligible_episode_indices(eef_fk_audit)
    if selection_override is None:
        selection = select_episode_roles(signals, eligible_episode_indices=eligible_episode_indices)
        selection_provenance: dict[str, object] = {
            "mode": "deterministic_numeric_activity",
            "numeric_audit_still_executed": True,
            "eef_fk_eligibility_still_enforced": eligible_episode_indices is not None,
        }
    else:
        selection, selection_provenance = _selection_from_override(
            selection_override,
            available_episode_indices={item.episode_index for item in signals},
            eligible_episode_indices=eligible_episode_indices,
        )
    metadata = _episodes_metadata(source, selection.all_indices())
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera_provenance = _raw_camera_provenance(config_path, raw_root, output)
    bundles: dict[str, str] = {}
    for role, episode in (("anchor", selection.anchor), *( ("calibration", value) for value in selection.calibration), *( ("validation", value) for value in selection.validation)):
        role_name = role if role == "anchor" else f"{role}_{episode:04d}"
        destination = output / "episodes" / f"{role_name}.json"
        atomic_write_json(destination, _episode_bundle(table, episode, metadata[episode]))
        bundles[role_name] = str(destination)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repo_id": SOURCE_REPO_ID,
            "revision": SOURCE_REVISION,
            "dataset_root": str(source),
            "read_only": True,
        },
        "selection": selection.json(),
        "selection_provenance": selection_provenance,
        "numeric_audit": numeric_audit,
        "eef_fk_selection": eef_fk_selection,
        "camera_provenance": camera_provenance,
        "episode_bundles": bundles,
        "acceptance_contract": {
            "camera_reprojection_median_px_max": 3.0,
            "camera_reprojection_p95_px_max": 8.0,
            "upper_body_joint_rmse_rad_max": 0.03,
            "table_translation_rmse_m_max": 0.020,
            "table_rotation_rmse_deg_max": 3.0,
            "phase_timing_max_error_s": 0.100,
            "mask_iou_min": 0.90,
        },
        "identifiability_limitations": [
            "No source table pose, contact force, robot-to-camera TF, or material coefficient is recorded.",
            "EEF labels are recorded in robot-root xyz/Euler coordinates, but do not themselves provide camera extrinsics or table pose.",
            "All fitted contact and extrinsic values must carry confidence/identifiability evidence.",
            "The fixed-base replay must never apply root pose as a per-frame teleport.",
        ],
    }
    atomic_write_json(output / "calibration_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument(
        "--eef-fk-audit",
        type=Path,
        help="all-episode EEF/FK audit; restricts the selected episodes to passing source labels",
    )
    parser.add_argument(
        "--selection-override",
        type=Path,
        help=(
            "reviewed visual-evidence selection JSON; numeric and EEF/FK audits still run, "
            "and all eight selected episodes must pass them"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_bundle(
        args.dataset_root,
        args.output_dir,
        config_path=args.config,
        raw_root=args.raw_root,
        eef_fk_audit=args.eef_fk_audit,
        selection_override=args.selection_override,
    )
    print(json.dumps({"selection": manifest["selection"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
