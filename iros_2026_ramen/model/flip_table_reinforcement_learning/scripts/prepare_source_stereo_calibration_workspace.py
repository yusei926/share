#!/usr/bin/env python3
"""Prepare auditable raw-stereo material for source table-frame calibration.

This tool never estimates a table pose.  It exports the recorded packed head
stereo frames, the time-aligned root-frame EEF labels, and an empty
correspondence template.  A human or a separately validated detector must add
only genuinely visible, named physical correspondences before passing the JSON
to ``calibrate_source_task_frame.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Any

import cv2
import numpy as np
import pyarrow.parquet as pq

from model.flip_table_reinforcement_learning.teacher.v1_table_geometry import (
    V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
    V1_TABLE001_BODY_FIDUCIALS,
    V1_TABLE001_BODY_FRAME,
)


def _parse_times(value: str, *, allow_negative: bool = False) -> list[float]:
    """Parse source times, retaining pre-subtask context only when requested."""

    times = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not times or not all(math.isfinite(item) for item in times):
        raise ValueError("--times-s must contain one or more finite seconds")
    if not allow_negative and any(item < 0.0 for item in times):
        raise ValueError("--times-s must contain non-negative seconds unless pre-subtask calibration context is enabled")
    return times


def _source_row_from_mcap_time(
    *,
    actual_episode_time_s: float,
    source_start_time_s: float,
    source_start_row: int,
    source_fps: float,
    allow_pre_subtask_calibration_context: bool = False,
) -> int:
    """Map one selected camera timestamp to its source LeRobot row.

    A calibration workspace must never silently pair a source-frame image from
    before the labelled subtask with the first row of that subtask.  Such a
    pairing is particularly harmful at the flip boundary, where the robot may
    already be moving.  The explicit calibration-context opt-in permits an
    earlier raw frame only when its corresponding source row is preserved in
    the manifest.  That context remains offline-only and still requires a
    static-table review before it can calibrate a task frame.
    """

    if (
        not math.isfinite(actual_episode_time_s)
        or not math.isfinite(source_start_time_s)
        or source_start_row < 0
        or not math.isfinite(source_fps)
        or source_fps <= 0.0
    ):
        raise ValueError("source timestamp, row, and FPS must be finite and valid")
    source_row = source_start_row + int(
        round((actual_episode_time_s - source_start_time_s) * source_fps)
    )
    if source_row < source_start_row and not allow_pre_subtask_calibration_context:
        raise ValueError(
            "selected MCAP camera frame maps before the labelled source slice; "
            "request a later --times-s value or explicitly opt in to a preserved "
            "pre-subtask calibration context"
        )
    return source_row


def _read_video_frame(video_path: Path, time_s: float) -> tuple[np.ndarray, float, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise RuntimeError(f"source video has invalid FPS: {fps}")
    requested_frame = int(round(time_s * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)
    ok, frame = capture.read()
    actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read frame {requested_frame} at {time_s:.3f}s from {video_path}")
    return frame, fps, actual_frame


def _cdr_align(cursor: int, alignment: int) -> int:
    return (cursor + alignment - 1) & -alignment


def _cdr_string(payload: memoryview, cursor: int) -> tuple[str, int]:
    cursor = _cdr_align(cursor, 4)
    if cursor + 4 > len(payload):
        raise ValueError("truncated CDR string length")
    length = struct.unpack_from("<I", payload, cursor)[0]
    cursor += 4
    if length == 0 or cursor + length > len(payload):
        raise ValueError("invalid CDR string length")
    raw = bytes(payload[cursor : cursor + length])
    if raw[-1] != 0:
        raise ValueError("CDR string is missing a null terminator")
    return raw[:-1].decode("utf-8"), _cdr_align(cursor + length, 4)


def decode_compressed_image_cdr(message_data: bytes) -> np.ndarray:
    """Decode ROS 2 ``sensor_msgs/CompressedImage`` CDR without ROS runtime."""

    payload = memoryview(message_data)
    if len(payload) < 4 or bytes(payload[:2]) != b"\x00\x01":
        raise ValueError("only little-endian CDR encapsulation is supported")
    cursor = 4
    # std_msgs/Header.stamp (sec:int32, nanosec:uint32), then frame_id.
    if cursor + 8 > len(payload):
        raise ValueError("truncated CompressedImage header")
    cursor += 8
    _frame_id, cursor = _cdr_string(payload, cursor)
    image_format, cursor = _cdr_string(payload, cursor)
    cursor = _cdr_align(cursor, 4)
    if cursor + 4 > len(payload):
        raise ValueError("truncated CompressedImage data length")
    length = struct.unpack_from("<I", payload, cursor)[0]
    cursor += 4
    if length == 0 or cursor + length > len(payload):
        raise ValueError("invalid CompressedImage data length")
    if "jpeg" not in image_format.lower():
        raise ValueError(f"expected JPEG head image, got format {image_format!r}")
    image = cv2.imdecode(np.frombuffer(payload[cursor : cursor + length], dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode CompressedImage JPEG")
    return image


def _read_mcap_frame(
    mcap_path: Path, *, target_time_ns: int, max_delta_ns: int = 100_000_000
) -> tuple[np.ndarray, int]:
    """Read the closest raw packed head-stereo image to one MCAP timestamp."""

    from mcap.reader import make_reader

    if max_delta_ns <= 0:
        raise ValueError("max_delta_ns must be positive")
    closest: tuple[int, int, bytes] | None = None
    with mcap_path.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, _channel, message in reader.iter_messages(
            topics=["/camera/head/image/compressed"],
            start_time=target_time_ns - max_delta_ns,
            end_time=target_time_ns + max_delta_ns + 1,
        ):
            error_ns = abs(message.log_time - target_time_ns)
            candidate = (error_ns, int(message.log_time), message.data)
            if closest is None or candidate[:2] < closest[:2]:
                closest = candidate
    if closest is None:
        raise RuntimeError(
            f"no /camera/head/image/compressed message within {max_delta_ns / 1e9:.3f}s "
            f"of {target_time_ns} in {mcap_path}"
        )
    _error_ns, log_time_ns, message_data = closest
    return decode_compressed_image_cdr(message_data), log_time_ns


def split_packed_head_stereo(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split organizer raw ``1280x480`` head RGB into its two ``640x480`` eyes."""

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"head frame must be HxWx3 BGR, got {frame_bgr.shape}")
    height, width = frame_bgr.shape[:2]
    if width % 2 or (width // 2) * 3 != height * 4:
        raise ValueError(
            "expected two side-by-side 4:3 organizer stereo eyes (1280x480); "
            f"got {width}x{height}"
        )
    midpoint = width // 2
    return frame_bgr[:, :midpoint].copy(), frame_bgr[:, midpoint:].copy()


def _read_eef_rows(parquet_path: Path, rows: list[int]) -> dict[int, dict[str, list[float]]]:
    table = pq.read_table(
        parquet_path,
        columns=["observation.state.ee_state", "action.ee_action"],
    )
    result: dict[int, dict[str, list[float]]] = {}
    for row in sorted(set(rows)):
        if not 0 <= row < table.num_rows:
            raise ValueError(f"source row {row} is outside [0, {table.num_rows})")
        state = [float(value) for value in table.column("observation.state.ee_state")[row].as_py()]
        action = [float(value) for value in table.column("action.ee_action")[row].as_py()]
        if len(state) != 12 or len(action) != 12 or not np.isfinite(state + action).all():
            raise ValueError(f"source row {row} has invalid EEF values")
        result[row] = {"source_eef_state_root": state, "source_eef_target_root": action}
    return result


def _write_contact_sheet(rows: list[dict[str, Any]], output_path: Path) -> None:
    tiles: list[np.ndarray] = []
    for row in rows:
        left = cv2.imread(str(row["left_image"]), cv2.IMREAD_COLOR)
        right = cv2.imread(str(row["right_image"]), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError("could not reload generated stereo frame")
        tile = np.concatenate((left, right), axis=1)
        cv2.putText(
            tile,
            f"t={row['video_time_s']:.3f}s  parquet_row={row['source_row']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"t={row['video_time_s']:.3f}s  parquet_row={row['source_row']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        raise ValueError("contact sheet requires at least one stereo pair")
    if not cv2.imwrite(str(output_path), np.concatenate(tiles, axis=0)):
        raise RuntimeError(f"could not write {output_path}")


def prepare_workspace(
    *,
    source_video: Path,
    source_parquet: Path,
    source_start_row: int,
    source_start_time_s: float,
    source_fps: float,
    times_s: list[float],
    output_dir: Path,
) -> dict[str, Any]:
    """Export source frames and row-aligned EEF candidates without inventing points."""

    if source_start_row < 0 or source_start_time_s < 0.0 or source_fps <= 0.0:
        raise ValueError("source row, source start time, and source FPS must be non-negative/positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    source_rows = [
        source_start_row + int(round((time_s - source_start_time_s) * source_fps))
        for time_s in times_s
    ]
    eef_by_row = _read_eef_rows(source_parquet, source_rows)
    records: list[dict[str, Any]] = []
    for index, time_s in enumerate(times_s):
        packed, video_fps, actual_frame = _read_video_frame(source_video, time_s)
        left, right = split_packed_head_stereo(packed)
        source_row = source_rows[index]
        eef = eef_by_row[source_row]
        prefix = f"{index:02d}_t{time_s:09.3f}".replace(".", "_")
        left_path = frames_dir / f"{prefix}_left.png"
        right_path = frames_dir / f"{prefix}_right.png"
        if not cv2.imwrite(str(left_path), left) or not cv2.imwrite(str(right_path), right):
            raise RuntimeError("could not write source stereo frames")
        records.append(
            {
                "video_time_s": float(time_s),
                "video_fps": video_fps,
                "video_frame_index": actual_frame,
                "source_row": source_row,
                "left_image": str(left_path),
                "right_image": str(right_path),
                **eef,
            }
        )

    manifest = {
        "schema_version": "flip_table_source_stereo_workspace/v1",
        "source_video": str(source_video),
        "source_parquet": str(source_parquet),
        "source_start_row": source_start_row,
        "source_start_time_s": source_start_time_s,
        "source_fps": source_fps,
        "frames": records,
        "notes": [
            "Each source_eef_state_root is a root-frame left-then-right [x,y,z,roll,pitch,yaw] label in metres/radians.",
            "Use a point only when the selected image pixel is the same physical point as its recorded root-frame label.",
            "The 50 mm wrist-forward EEF convention must be respected; do not label an arbitrary finger pixel as an EEF point.",
            "Table point coordinates must be in the V1 Table001/Table001_01 body frame, not an arbitrary tabletop-corner frame.",
            "Do not set table_is_static_confirmation true until all table points come from one unchanged source table pose.",
        ],
    }
    manifest_path = output_dir / "workspace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    template = {
        "table_is_static_confirmation": False,
        "source_task_frame_provenance": "replace with recorded frame IDs and named physical fiducials",
        "source_table_body_frame": V1_TABLE001_BODY_FRAME,
        "root_camera_correspondences": [],
        "table_correspondences": [],
        "annotation_instructions": manifest["notes"],
        "v1_table001_body_fiducial_candidates": V1_TABLE001_BODY_FIDUCIALS,
        "v1_table001_body_fiducial_provenance": V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
        "workspace_manifest": str(manifest_path),
    }
    template_path = output_dir / "source_stereo_fiducials.template.json"
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(records, output_dir / "stereo_contact_sheet.png")
    return {
        "output_dir": str(output_dir),
        "frame_count": len(records),
        "workspace_manifest": str(manifest_path),
        "correspondence_template": str(template_path),
    }


def prepare_workspace_from_mcap(
    *,
    source_mcap: Path,
    episode_start_time_ns: int,
    source_parquet: Path,
    source_start_row: int,
    source_start_time_s: float,
    source_fps: float,
    times_s: list[float],
    output_dir: Path,
    allow_pre_subtask_calibration_context: bool = False,
) -> dict[str, Any]:
    """Build the same workspace from exact MCAP log timestamps.

    ``allow_pre_subtask_calibration_context`` is deliberately false by
    default.  When enabled, an earlier recorded source row is retained as an
    offline calibration observation rather than being relabelled as the first
    policy subtask row.
    """

    if episode_start_time_ns <= 0:
        raise ValueError("episode_start_time_ns must be positive")
    if source_start_row < 0 or source_start_time_s < 0.0 or source_fps <= 0.0:
        raise ValueError("source row, source start time, and source FPS must be non-negative/positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, time_s in enumerate(times_s):
        requested_time_ns = episode_start_time_ns + int(round(time_s * 1e9))
        packed, actual_time_ns = _read_mcap_frame(source_mcap, target_time_ns=requested_time_ns)
        left, right = split_packed_head_stereo(packed)
        actual_episode_time_s = (actual_time_ns - episode_start_time_ns) / 1e9
        source_row = _source_row_from_mcap_time(
            actual_episode_time_s=actual_episode_time_s,
            source_start_time_s=source_start_time_s,
            source_start_row=source_start_row,
            source_fps=source_fps,
            allow_pre_subtask_calibration_context=allow_pre_subtask_calibration_context,
        )
        source_label_time_s = source_start_time_s + (source_row - source_start_row) / source_fps
        prefix = f"{index:02d}_t{time_s:09.3f}".replace(".", "_")
        left_path = frames_dir / f"{prefix}_left.png"
        right_path = frames_dir / f"{prefix}_right.png"
        if not cv2.imwrite(str(left_path), left) or not cv2.imwrite(str(right_path), right):
            raise RuntimeError("could not write source stereo frames")
        records.append(
            {
                "video_time_s": float(time_s),
                "mcap_requested_time_ns": requested_time_ns,
                "mcap_log_time_ns": actual_time_ns,
                "mcap_time_error_ms": (actual_time_ns - requested_time_ns) / 1e6,
                "source_row": source_row,
                "source_label_time_s": source_label_time_s,
                "source_label_time_error_ms": (actual_episode_time_s - source_label_time_s) * 1e3,
                "left_image": str(left_path),
                "right_image": str(right_path),
            }
        )
    eef_by_row = _read_eef_rows(source_parquet, [int(record["source_row"]) for record in records])
    for record in records:
        record.update(eef_by_row[int(record["source_row"])])

    manifest = {
        "schema_version": "flip_table_source_stereo_workspace/v1",
        "source_mcap": str(source_mcap),
        "episode_start_time_ns": episode_start_time_ns,
        "source_parquet": str(source_parquet),
        "source_start_row": source_start_row,
        "source_start_time_s": source_start_time_s,
        "source_fps": source_fps,
        "pre_subtask_calibration_context": bool(allow_pre_subtask_calibration_context),
        "frames": records,
        "notes": [
            "Frames are selected by exact MCAP log timestamp; the displayed timestamp is seconds from episode start.",
            "Each source_eef_state_root is a root-frame left-then-right [x,y,z,roll,pitch,yaw] label in metres/radians.",
            "Use a point only when the selected image pixel is the same physical point as its recorded root-frame label.",
            "The 50 mm wrist-forward EEF convention must be respected; do not label an arbitrary finger pixel as an EEF point.",
            "Table point coordinates must be in the V1 Table001/Table001_01 body frame, not an arbitrary tabletop-corner frame.",
            "Do not set table_is_static_confirmation true until all table points come from one unchanged source table pose.",
            "Rows before source_start_row are calibration-only context when pre_subtask_calibration_context is true; they are never policy demonstrations.",
        ],
    }
    manifest_path = output_dir / "workspace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    template = {
        "table_is_static_confirmation": False,
        "source_task_frame_provenance": "replace with recorded MCAP times and named physical fiducials",
        "source_table_body_frame": V1_TABLE001_BODY_FRAME,
        "root_camera_correspondences": [],
        "table_correspondences": [],
        "annotation_instructions": manifest["notes"],
        "v1_table001_body_fiducial_candidates": V1_TABLE001_BODY_FIDUCIALS,
        "v1_table001_body_fiducial_provenance": V1_TABLE001_BODY_FIDUCIAL_PROVENANCE,
        "workspace_manifest": str(manifest_path),
    }
    template_path = output_dir / "source_stereo_fiducials.template.json"
    template_path.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    _write_contact_sheet(records, output_dir / "stereo_contact_sheet.png")
    return {
        "output_dir": str(output_dir),
        "frame_count": len(records),
        "workspace_manifest": str(manifest_path),
        "correspondence_template": str(template_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-video", type=Path)
    source.add_argument("--source-mcap", type=Path)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--source-start-row", type=int, required=True)
    parser.add_argument("--source-start-time-s", type=float, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--episode-start-time-ns", type=int)
    parser.add_argument("--times-s", required=True, help="comma-separated absolute source-video timestamps")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-pre-subtask-calibration-context",
        action="store_true",
        help="retain earlier raw frames with their true source rows for offline static-table calibration only",
    )
    args = parser.parse_args()
    if args.source_mcap is not None:
        if args.episode_start_time_ns is None:
            parser.error("--source-mcap requires --episode-start-time-ns")
        result = prepare_workspace_from_mcap(
            source_mcap=args.source_mcap,
            episode_start_time_ns=args.episode_start_time_ns,
            source_parquet=args.source_parquet,
            source_start_row=args.source_start_row,
            source_start_time_s=args.source_start_time_s,
            source_fps=args.source_fps,
            times_s=_parse_times(
                args.times_s,
                allow_negative=args.allow_pre_subtask_calibration_context,
            ),
            output_dir=args.output_dir,
            allow_pre_subtask_calibration_context=args.allow_pre_subtask_calibration_context,
        )
    else:
        result = prepare_workspace(
            source_video=args.source_video,
            source_parquet=args.source_parquet,
            source_start_row=args.source_start_row,
            source_start_time_s=args.source_start_time_s,
            source_fps=args.source_fps,
            times_s=_parse_times(args.times_s),
            output_dir=args.output_dir,
        )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
