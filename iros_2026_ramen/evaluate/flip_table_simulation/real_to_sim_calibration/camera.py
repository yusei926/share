#!/usr/bin/env python3
"""Translate pinned real camera intrinsics into V1 camera override candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json, atomic_write_text


FOCAL_LENGTH_MM = 24.0


def _intrinsic_env(prefix: str, intrinsic: dict[str, Any]) -> dict[str, str]:
    width = int(intrinsic["width"])
    height = int(intrinsic["height"])
    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    if (width, height) != (640, 480) or fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"camera intrinsic must be positive 640x480, got {intrinsic!r}")
    return {
        f"{prefix}_FOCAL_LENGTH": f"{FOCAL_LENGTH_MM:.9g}",
        f"{prefix}_HORIZONTAL_APERTURE": f"{width * FOCAL_LENGTH_MM / fx:.12g}",
        f"{prefix}_VERTICAL_APERTURE": f"{height * FOCAL_LENGTH_MM / fy:.12g}",
    }


def _head_raw_intrinsic(payload: dict[str, Any], side: str) -> dict[str, Any]:
    matrix = payload[f"camera_matrix_{side}"]
    return {
        "width": int(payload["image_size"][0]),
        "height": int(payload["image_size"][1]),
        "fx": float(matrix[0][0]),
        "fy": float(matrix[1][1]),
        "ppx": float(matrix[0][2]),
        "ppy": float(matrix[1][2]),
        "distortion": [float(value) for value in payload[f"dist_coeffs_{side}"]],
    }


def _d405_intrinsic(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    serial = str(payload["serial_number"])
    intrinsic = payload["color"]["intrinsics"]
    return serial, {
        "width": int(intrinsic["width"]),
        "height": int(intrinsic["height"]),
        "fx": float(intrinsic["fx"]),
        "fy": float(intrinsic["fy"]),
        "ppx": float(intrinsic["ppx"]),
        "ppy": float(intrinsic["ppy"]),
        "distortion_model": str(intrinsic["model"]),
        "distortion": [float(value) for value in intrinsic["coeffs"]],
    }


def build_candidates(calibration_dir: Path) -> dict[str, Any]:
    head = yaml.safe_load((calibration_dir / "head_camera_params.yaml").read_text(encoding="utf-8"))
    if not isinstance(head, dict) or head.get("success") is not True:
        raise ValueError("head_camera_params.yaml is not a successful calibration")
    left_head = _head_raw_intrinsic(head, "left")
    right_head = _head_raw_intrinsic(head, "right")
    d405 = dict(_d405_intrinsic(path) for path in sorted(calibration_dir.glob("camera_*.json")))
    if len(d405) != 2:
        raise ValueError("exactly two D405 calibration files are required")
    serials = tuple(sorted(d405))
    common = {
        **_intrinsic_env("FLIP_TABLE_HEAD_LEFT_CAMERA", left_head),
        **_intrinsic_env("FLIP_TABLE_HEAD_RIGHT_CAMERA", right_head),
    }
    candidates: dict[str, dict[str, str]] = {}
    for name, left_serial, right_serial in (
        (f"left_{serials[0]}_right_{serials[1]}", serials[0], serials[1]),
        (f"left_{serials[1]}_right_{serials[0]}", serials[1], serials[0]),
    ):
        candidates[name] = {
            **common,
            **_intrinsic_env("FLIP_TABLE_LEFT_WRIST_CAMERA", d405[left_serial]),
            **_intrinsic_env("FLIP_TABLE_RIGHT_WRIST_CAMERA", d405[right_serial]),
        }
    return {
        "schema_version": "team_ramen_flip_table_camera_intrinsics/v1",
        "focal_length_mm": FOCAL_LENGTH_MM,
        "head_left_raw": left_head,
        "head_right_raw": right_head,
        "head_stereo_baseline_m": float(head["baseline"]) / 1000.0,
        "d405_by_serial": d405,
        "candidates": candidates,
        "selection_rule": (
            "The raw metadata does not identify which D405 serial is left/right. "
            "Render both candidates with identical extrinsics, then choose the lower held-out wrist image error."
        ),
        "limitation": (
            "The V1 CameraCfg exposes focal/aperture but not the recorded principal-point offset or lens distortion. "
            "Those residuals remain explicit calibration metrics, not silently ignored."
        ),
    }


def _shell_escape(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_candidates(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "camera_intrinsics_report.json", report)
    for label, values in report["candidates"].items():
        lines = ["# Generated from pinned raw-MCAP camera calibration."]
        lines.extend(f"export {key}={_shell_escape(value)}" for key, value in sorted(values.items()))
        atomic_write_text(output_dir / f"camera_{label}.env", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_candidates(args.calibration_dir.expanduser().resolve())
    write_candidates(report, args.output_dir.expanduser().resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
