#!/usr/bin/env python3
"""Export time-aligned D405 IR stereo for auditable source hand-eye calibration.

The source MCAP has no table pose and no wrist-camera TF.  This command only
extracts recorded D405 IR1/IR2 pairs, maps each pair midpoint to the recorded
root-frame EEF state, and writes an annotation template.  It deliberately does
not infer pixels, a table pose, or an EEF-to-camera transform.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2

try:
    from model.flip_table_reinforcement_learning.scripts.prepare_source_stereo_calibration_workspace import (
        V1_TABLE001_BODY_FIDUCIALS,
        V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
        _parse_times,
        _read_eef_rows,
        _source_row_from_mcap_time,
        decode_compressed_image_cdr,
    )
except ModuleNotFoundError:  # Direct execution from the repository subdirectory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from model.flip_table_reinforcement_learning.scripts.prepare_source_stereo_calibration_workspace import (
        V1_TABLE001_BODY_FIDUCIALS,
        V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
        _parse_times,
        _read_eef_rows,
        _source_row_from_mcap_time,
        decode_compressed_image_cdr,
    )


def _read_closest_stereo_pair(
    mcap_path: Path,
    *,
    first_topic: str,
    second_topic: str,
    target_time_ns: int,
    max_delta_ns: int,
    max_stereo_skew_ns: int,
) -> tuple[Any, int, Any, int]:
    """Read the closest valid pair, never combining adjacent D405 frames."""

    from mcap.reader import make_reader

    if max_delta_ns <= 0 or max_stereo_skew_ns <= 0:
        raise ValueError("message delta and stereo skew limits must be positive")
    candidates: dict[str, list[tuple[int, bytes]]] = {
        first_topic: [],
        second_topic: [],
    }
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, channel, message in reader.iter_messages(
            topics=[first_topic, second_topic],
            start_time=target_time_ns - max_delta_ns,
            end_time=target_time_ns + max_delta_ns + 1,
        ):
            candidates[channel.topic].append((int(message.log_time), message.data))
    if not candidates[first_topic] or not candidates[second_topic]:
        raise RuntimeError(
            f"no complete {first_topic}/{second_topic} candidate set within "
            f"{max_delta_ns / 1e9:.3f}s of "
            f"{target_time_ns} in {mcap_path}"
        )
    first_time, first_data, second_time, second_data = _select_closest_stereo_pair(
        candidates[first_topic],
        candidates[second_topic],
        target_time_ns=target_time_ns,
        max_stereo_skew_ns=max_stereo_skew_ns,
    )
    return (
        decode_compressed_image_cdr(first_data),
        first_time,
        decode_compressed_image_cdr(second_data),
        second_time,
    )


def _select_closest_stereo_pair(
    first_candidates: list[tuple[int, Any]],
    second_candidates: list[tuple[int, Any]],
    *,
    target_time_ns: int,
    max_stereo_skew_ns: int,
) -> tuple[int, Any, int, Any]:
    """Select a synchronized pair by midpoint error, then pair skew."""

    if not first_candidates or not second_candidates or max_stereo_skew_ns <= 0:
        raise ValueError("stereo candidates must be non-empty and the skew limit positive")
    valid_pairs = []
    for first_time, first_data in first_candidates:
        for second_time, second_data in second_candidates:
            skew = abs(first_time - second_time)
            if skew > max_stereo_skew_ns:
                continue
            midpoint = (first_time + second_time) // 2
            valid_pairs.append(
                (
                    abs(midpoint - target_time_ns),
                    skew,
                    first_time,
                    second_time,
                    first_data,
                    second_data,
                )
            )
    if not valid_pairs:
        nearest_skew = min(
            abs(first[0] - second[0])
            for first in first_candidates
            for second in second_candidates
        )
        raise RuntimeError(
            f"no D405 pair satisfies {max_stereo_skew_ns / 1e6:.3f}ms skew; "
            f"nearest candidate is {nearest_skew / 1e6:.3f}ms"
        )
    _midpoint_error, _skew, first_time, second_time, first_data, second_data = min(
        valid_pairs,
        key=lambda pair: pair[:4],
    )
    return first_time, first_data, second_time, second_data


def _write_contact_sheet(records: list[dict[str, Any]], output_path: Path) -> None:
    tiles = []
    for record in records:
        ir1 = cv2.imread(str(record["ir1_image"]), cv2.IMREAD_COLOR)
        ir2 = cv2.imread(str(record["ir2_image"]), cv2.IMREAD_COLOR)
        if ir1 is None or ir2 is None:
            raise RuntimeError("could not reload generated D405 IR images")
        tile = cv2.hconcat((ir1, ir2))
        label = (
            f"t={record['pair_midpoint_episode_time_s']:.3f}s "
            f"row={record['source_row']} skew={record['stereo_skew_ms']:.1f}ms"
        )
        cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        raise ValueError("contact sheet requires at least one D405 stereo pair")
    if not cv2.imwrite(str(output_path), cv2.vconcat(tiles)):
        raise RuntimeError(f"could not write {output_path}")


def _side_eef(full_eef: list[float], wrist_side: str) -> list[float]:
    if len(full_eef) != 12:
        raise ValueError("source EEF label must contain left/right 12 values")
    return full_eef[:6] if wrist_side == "left" else full_eef[6:]


def build_annotation_template(
    records: list[dict[str, Any]],
    *,
    wrist_side: str,
    manifest_path: Path,
    notes: list[str],
    source_episode_index: int | None = None,
    calibration_endpoint: str | None = None,
    source_annotation_workspace_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create an intentionally empty, provenance-bearing annotation payload."""

    if wrist_side not in {"left", "right"}:
        raise ValueError("wrist_side must be left or right")
    binding_values = (
        source_episode_index,
        calibration_endpoint,
        source_annotation_workspace_manifest_sha256,
    )
    if any(value is not None for value in binding_values) and not all(
        value is not None for value in binding_values
    ):
        raise ValueError(
            "source episode, calibration endpoint, and annotation workspace hash must be provided together"
        )
    if source_episode_index is not None and source_episode_index < 0:
        raise ValueError("source_episode_index must be non-negative")
    if calibration_endpoint is not None and calibration_endpoint not in {"initial", "final"}:
        raise ValueError("calibration_endpoint must be initial or final")
    if source_annotation_workspace_manifest_sha256 is not None and (
        len(source_annotation_workspace_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_annotation_workspace_manifest_sha256)
    ):
        raise ValueError("source annotation workspace SHA-256 must be 64 lowercase hexadecimal characters")

    result = {
        "source_table_body_frame": "RoboFinals-IKEA-V1:/Root/Table001_01",
        "source_task_frame_provenance": (
            "replace with MCAP pair IDs and named Table001 registration-site observations"
        ),
        "table_is_static_confirmation": False,
        "d405_is_rigid_to_eef_confirmation": False,
        "wrist_side": wrist_side,
        "wrist_table_observations": [
            {
                "observation_id": record["observation_id"],
                "source_eef_state_root": record["source_eef_state_root"],
                "table_fiducials": [],
            }
            for record in records
        ],
        "v1_table001_body_fiducial_candidates": V1_TABLE001_BODY_FIDUCIALS,
        "v1_table001_body_fiducial_provenance": V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
        "annotation_instructions": notes,
        "workspace_manifest": str(manifest_path),
    }
    if source_episode_index is not None:
        result.update(
            {
                "source_episode_index": source_episode_index,
                "calibration_endpoint": calibration_endpoint,
                "workspace_manifest_sha256": source_annotation_workspace_manifest_sha256,
            }
        )
    return result


def prepare_workspace_from_mcap(
    *,
    source_mcap: Path,
    episode_start_time_ns: int,
    source_parquet: Path,
    source_start_row: int,
    source_start_time_s: float,
    source_fps: float,
    wrist_side: str,
    times_s: list[float],
    output_dir: Path,
    max_message_delta_ms: float = 100.0,
    max_stereo_skew_ms: float = 30.0,
    allow_pre_subtask_calibration_context: bool = False,
    source_episode_index: int | None = None,
    calibration_endpoint: str | None = None,
    source_annotation_workspace_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a static-table annotation workspace without guessing geometry.

    Earlier source rows are rejected unless the explicit calibration-context
    opt-in is set.  Opted-in rows retain their real indices and are offline
    calibration material only, never a policy demonstration.
    """

    if episode_start_time_ns <= 0 or source_start_row < 0 or source_start_time_s < 0.0:
        raise ValueError("episode start, source row, and source start time must be non-negative/positive")
    if source_fps <= 0.0 or not math.isfinite(source_fps):
        raise ValueError("source_fps must be finite and positive")
    if wrist_side not in {"left", "right"}:
        raise ValueError("wrist_side must be left or right")
    if min(max_message_delta_ms, max_stereo_skew_ms) <= 0.0:
        raise ValueError("D405 timestamp tolerances must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    max_delta_ns = int(round(max_message_delta_ms * 1e6))
    records: list[dict[str, Any]] = []
    for index, requested_time_s in enumerate(times_s):
        requested_time_ns = episode_start_time_ns + int(round(requested_time_s * 1e9))
        ir1_topic = f"/camera/{wrist_side}_wrist/ir1/compressed"
        ir2_topic = f"/camera/{wrist_side}_wrist/ir2/compressed"
        ir1, ir1_time_ns, ir2, ir2_time_ns = _read_closest_stereo_pair(
            source_mcap,
            first_topic=ir1_topic,
            second_topic=ir2_topic,
            target_time_ns=requested_time_ns,
            max_delta_ns=max_delta_ns,
            max_stereo_skew_ns=int(round(max_stereo_skew_ms * 1e6)),
        )
        stereo_skew_ms = abs(ir1_time_ns - ir2_time_ns) / 1e6
        if stereo_skew_ms > max_stereo_skew_ms:
            raise RuntimeError(
                f"D405 pair at {requested_time_s:.3f}s has {stereo_skew_ms:.3f}ms skew, "
                f"above {max_stereo_skew_ms:.3f}ms"
            )
        midpoint_time_ns = int(round((ir1_time_ns + ir2_time_ns) / 2.0))
        midpoint_episode_time_s = (midpoint_time_ns - episode_start_time_ns) / 1e9
        source_row = _source_row_from_mcap_time(
            actual_episode_time_s=midpoint_episode_time_s,
            source_start_time_s=source_start_time_s,
            source_start_row=source_start_row,
            source_fps=source_fps,
            allow_pre_subtask_calibration_context=allow_pre_subtask_calibration_context,
        )
        source_label_time_s = source_start_time_s + (source_row - source_start_row) / source_fps
        records.append(
            {
                "observation_id": f"{wrist_side}_wrist_{index:02d}",
                "requested_episode_time_s": float(requested_time_s),
                "ir1_mcap_log_time_ns": ir1_time_ns,
                "ir2_mcap_log_time_ns": ir2_time_ns,
                "pair_midpoint_episode_time_s": midpoint_episode_time_s,
                "stereo_skew_ms": stereo_skew_ms,
                "source_row": source_row,
                "source_label_time_s": source_label_time_s,
                "source_label_time_error_ms": (midpoint_episode_time_s - source_label_time_s) * 1e3,
                "_ir1_frame": ir1,
                "_ir2_frame": ir2,
            }
        )
    eef_by_row = _read_eef_rows(source_parquet, [int(record["source_row"]) for record in records])
    for record in records:
        labels = eef_by_row[int(record["source_row"])]
        record["source_eef_state_root"] = _side_eef(labels["source_eef_state_root"], wrist_side)
        record["source_eef_target_root"] = _side_eef(labels["source_eef_target_root"], wrist_side)
    # Do not leave a deceptively complete-looking partial workspace if a later
    # timestamp violates the stereo-pair gate.
    for index, record in enumerate(records):
        prefix = f"{index:02d}_t{record['requested_episode_time_s']:09.3f}".replace(".", "_")
        ir1_path = frames_dir / f"{prefix}_ir1.png"
        ir2_path = frames_dir / f"{prefix}_ir2.png"
        if not cv2.imwrite(str(ir1_path), record.pop("_ir1_frame")) or not cv2.imwrite(
            str(ir2_path), record.pop("_ir2_frame")
        ):
            raise RuntimeError("could not write source D405 IR frames")
        record["ir1_image"] = str(ir1_path)
        record["ir2_image"] = str(ir2_path)

    manifest = {
        "schema_version": "flip_table_source_wrist_hand_eye_workspace/v1",
        "source_mcap": str(source_mcap),
        "episode_start_time_ns": episode_start_time_ns,
        "source_parquet": str(source_parquet),
        "source_start_row": source_start_row,
        "source_start_time_s": source_start_time_s,
        "source_fps": source_fps,
        "pre_subtask_calibration_context": bool(allow_pre_subtask_calibration_context),
        "wrist_side": wrist_side,
        "frames": records,
        "notes": [
            "Each pair is selected from raw D405 IR1/IR2 MCAP timestamps; source EEF state is nearest at the pair midpoint.",
            "Annotate only named, visible points on the static assembled white Table001 in both IR images.",
            "table_point_m must use the V1 Table001/Table001_01 body frame registration-site coordinates below.",
            "Do not confirm table staticness unless all selected observations precede any table motion.",
            "Rows before source_start_row are calibration-only context when pre_subtask_calibration_context is true; they are never policy demonstrations.",
            "Never use the resulting table pose, EEF-to-camera transform, or residuals as deployed policy, critic, or planner input.",
        ],
    }
    manifest_path = output_dir / "workspace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    template = build_annotation_template(
        records,
        wrist_side=wrist_side,
        manifest_path=manifest_path,
        notes=manifest["notes"],
        source_episode_index=source_episode_index,
        calibration_endpoint=calibration_endpoint,
        source_annotation_workspace_manifest_sha256=source_annotation_workspace_manifest_sha256,
    )
    template_path = output_dir / "source_wrist_hand_eye.template.json"
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(records, output_dir / "d405_ir_stereo_contact_sheet.png")
    return {
        "output_dir": str(output_dir),
        "frame_count": len(records),
        "workspace_manifest": str(manifest_path),
        "observation_template": str(template_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mcap", type=Path, required=True)
    parser.add_argument("--episode-start-time-ns", type=int, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--source-start-row", type=int, required=True)
    parser.add_argument("--source-start-time-s", type=float, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--wrist-side", choices=("left", "right"), required=True)
    parser.add_argument("--times-s", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-message-delta-ms", type=float, default=100.0)
    parser.add_argument("--max-stereo-skew-ms", type=float, default=30.0)
    parser.add_argument("--source-episode-index", type=int)
    parser.add_argument("--calibration-endpoint", choices=("initial", "final"))
    parser.add_argument("--source-annotation-workspace-manifest-sha256")
    parser.add_argument(
        "--allow-pre-subtask-calibration-context",
        action="store_true",
        help="retain earlier raw frames with their true source rows for offline static-table calibration only",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_workspace_from_mcap(
                source_mcap=args.source_mcap,
                episode_start_time_ns=args.episode_start_time_ns,
                source_parquet=args.source_parquet,
                source_start_row=args.source_start_row,
                source_start_time_s=args.source_start_time_s,
                source_fps=args.source_fps,
                wrist_side=args.wrist_side,
                times_s=_parse_times(
                    args.times_s,
                    allow_negative=args.allow_pre_subtask_calibration_context,
                ),
                output_dir=args.output_dir,
                max_message_delta_ms=args.max_message_delta_ms,
                max_stereo_skew_ms=args.max_stereo_skew_ms,
                allow_pre_subtask_calibration_context=args.allow_pre_subtask_calibration_context,
                source_episode_index=args.source_episode_index,
                calibration_endpoint=args.calibration_endpoint,
                source_annotation_workspace_manifest_sha256=(
                    args.source_annotation_workspace_manifest_sha256
                ),
            )
        )
    )


if __name__ == "__main__":
    main()
