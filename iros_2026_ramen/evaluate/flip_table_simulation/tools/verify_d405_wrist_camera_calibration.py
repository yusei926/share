#!/usr/bin/env python3
"""Verify the Dex1+D405 wrist-camera calibration assumptions.

This tool is intentionally lightweight: it does not require a CAD kernel.  It
checks the parts that should stay deterministic in this repository:

* Unitree Device.md identifies the G1 wrist cameras as RealSense D405 mounted
  through the Dex1-1 D405 STEP bracket.
* D405 RGB intrinsics are represented in USD camera parameters by the standard
  pinhole relation FOV = 2 atan(aperture / (2 focal_length)).
* The local STEP bracket exists and has plausible M5010-ring-scale dimensions.
* The active simulator patch still contains the expected wrist-camera parent,
  optical-center offset, quaternion, and 640x480 image size.

When real and simulated wrist-frame directories are provided, the tool also
delegates to compare_wrist_camera_distribution.py and stores that report path in
the output JSON.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATCH = ROOT / "evaluate" / "flip_table_simulation" / "container_overlay" / "patches" / "patch_g1_global_camera.py"
DEFAULT_STEP = os.environ.get("D405_BRACKET_STEP")
DEFAULT_OUTPUT = ROOT / "outputs" / "flip_table_simulation" / "d405_wrist_camera_calibration_report.json"

D405_TARGET_HORIZONTAL_FOV_DEG = 72.67842331799163
D405_TARGET_VERTICAL_FOV_DEG = 57.772836250948245
D405_TARGET_WIDTH = 640
D405_TARGET_HEIGHT = 480
FOV_TOLERANCE_DEG = 0.05
QUATERNION_NORM_TOLERANCE = 1.0e-3


def _literal_from_assignment(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^\n#]+)", text, flags=re.MULTILINE)
    if match is None:
        raise KeyError(f"Could not find assignment for {name}")
    return match.group(1).strip()


def _string_assignment(text: str, name: str) -> str:
    literal = _literal_from_assignment(text, name)
    match = re.fullmatch(r"[\"'](.+)[\"']", literal)
    if match is None:
        raise ValueError(f"{name} is not a simple string literal: {literal}")
    return match.group(1)


def _int_assignment(text: str, name: str) -> int:
    return int(_literal_from_assignment(text, name))


def _float_tuple(value: str) -> list[float]:
    stripped = value.strip()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        raise ValueError(f"Expected tuple literal string, got {value!r}")
    return [float(part.strip()) for part in stripped[1:-1].split(",") if part.strip()]


def _fov_from_aperture(focal_length: float, aperture: float) -> float:
    return math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))


def _aperture_from_fov(focal_length: float, fov_deg: float) -> float:
    return 2.0 * focal_length * math.tan(math.radians(fov_deg / 2.0))


def _parse_step_points(step_path: Path) -> dict[str, object]:
    text = step_path.read_text(encoding="latin-1", errors="ignore")
    point_pattern = re.compile(
        r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
        r"([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*,\s*([-+0-9.Ee]+)\s*\)\s*\)"
    )
    points = [[float(x), float(y), float(z)] for x, y, z in point_pattern.findall(text)]
    if not points:
        raise ValueError(f"No CARTESIAN_POINT entries found in {step_path}")

    mins = [min(point[axis] for point in points) for axis in range(3)]
    maxs = [max(point[axis] for point in points) for axis in range(3)]
    extents = [maxs[axis] - mins[axis] for axis in range(3)]

    radius_values = [
        round(float(value), 4)
        for value in re.findall(r"(?:CIRCLE|CYLINDRICAL_SURFACE)\s*\([^;]*,\s*([-+0-9.Ee]+)\s*\)", text)
    ]
    common_radii = Counter(radius_values).most_common(12)

    return {
        "path": str(step_path),
        "exists": True,
        "cartesian_point_count": len(points),
        "bbox_min_mm": mins,
        "bbox_max_mm": maxs,
        "bbox_extent_mm": extents,
        "bbox_extent_m": [value / 1000.0 for value in extents],
        "common_circle_or_cylindrical_radii_mm": [
            {"radius_mm": radius, "count": count} for radius, count in common_radii
        ],
    }


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _camera_patch_report(patch_path: Path) -> dict[str, object]:
    text = patch_path.read_text(encoding="utf-8")
    focal = float(_string_assignment(text, "D405_FOCAL_LENGTH"))
    horizontal_aperture = float(_string_assignment(text, "D405_HORIZONTAL_APERTURE"))
    vertical_aperture = float(_string_assignment(text, "D405_VERTICAL_APERTURE"))
    left_pos = _float_tuple(_string_assignment(text, "DEX1_D405_LEFT_OPTICAL_CENTER_POS"))
    right_pos = _float_tuple(_string_assignment(text, "DEX1_D405_RIGHT_OPTICAL_CENTER_POS"))
    rot = _float_tuple(_string_assignment(text, "DEX1_D405_OPTICAL_CENTER_ROT"))
    width = _int_assignment(text, "DEFAULT_CAMERA_WIDTH")
    height = _int_assignment(text, "DEFAULT_CAMERA_HEIGHT")

    horizontal_fov = _fov_from_aperture(focal, horizontal_aperture)
    vertical_fov = _fov_from_aperture(focal, vertical_aperture)
    quaternion_norm = math.sqrt(sum(value * value for value in rot))

    checks = [
        _check(
            "d405_horizontal_fov",
            abs(horizontal_fov - D405_TARGET_HORIZONTAL_FOV_DEG) <= FOV_TOLERANCE_DEG,
            f"{horizontal_fov:.4f} deg from focal={focal} aperture={horizontal_aperture}",
        ),
        _check(
            "d405_vertical_fov",
            abs(vertical_fov - D405_TARGET_VERTICAL_FOV_DEG) <= FOV_TOLERANCE_DEG,
            f"{vertical_fov:.4f} deg from focal={focal} aperture={vertical_aperture}",
        ),
        _check(
            "camera_resolution",
            width == D405_TARGET_WIDTH and height == D405_TARGET_HEIGHT,
            f"{width}x{height}",
        ),
        _check(
            "wrist_camera_parents",
            "left_wrist_yaw_link/left_hand_camera" in text
            and "right_wrist_yaw_link/right_hand_camera" in text,
            "left/right wrist cameras are attached to wrist_yaw_link parents",
        ),
        _check(
            "no_torso_wrist_policy_camera",
            "torso_link/left_hand_camera" not in text
            and "torso_link/right_hand_camera" not in text,
            "policy wrist cameras are not torso cameras",
        ),
        _check(
            "same_left_right_mount",
            left_pos == right_pos,
            f"left={left_pos}, right={right_pos}",
        ),
        _check(
            "unit_quaternion",
            abs(quaternion_norm - 1.0) <= QUATERNION_NORM_TOLERANCE,
            f"norm={quaternion_norm:.8f}",
        ),
    ]
    return {
        "path": str(patch_path),
        "focal_length_mm": focal,
        "horizontal_aperture_mm": horizontal_aperture,
        "vertical_aperture_mm": vertical_aperture,
        "derived_horizontal_fov_deg": horizontal_fov,
        "derived_vertical_fov_deg": vertical_fov,
    "expected_horizontal_aperture_mm": _aperture_from_fov(focal, D405_TARGET_HORIZONTAL_FOV_DEG),
    "expected_vertical_aperture_mm": _aperture_from_fov(focal, D405_TARGET_VERTICAL_FOV_DEG),
        "left_optical_center_pos_m": left_pos,
        "right_optical_center_pos_m": right_pos,
        "optical_center_rot_wxyz": rot,
        "optical_center_rot_norm": quaternion_norm,
        "width": width,
        "height": height,
        "checks": checks,
    }


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_distribution_compare(
    real_dirs: list[Path],
    sim_roots: list[Path],
    report_path: Path,
    contact_sheet_path: Path | None,
    rank_by: str,
) -> dict[str, object]:
    if not real_dirs or not sim_roots:
        return {"status": "skipped", "reason": "real/sim wrist image directories were not provided"}
    if not _has_module("numpy"):
        return {
            "status": "skipped",
            "reason": "numpy is not available in this Python environment; run with the training venv Python",
        }

    script = Path(__file__).with_name("compare_wrist_camera_distribution.py")
    command = [sys.executable, str(script)]
    for real_dir in real_dirs:
        command += ["--real-dir", str(real_dir)]
    for sim_root in sim_roots:
        command += ["--sim-root", str(sim_root)]
    command += ["--output", str(report_path), "--rank-by", rank_by]
    if contact_sheet_path is not None:
        command += ["--contact-sheet", str(contact_sheet_path)]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    summary = {"status": "ok", "command": command, "stdout": completed.stdout.strip(), "report_path": str(report_path)}
    if contact_sheet_path is not None:
        summary["contact_sheet_path"] = str(contact_sheet_path)
    if report_path.exists():
        data = json.loads(report_path.read_text(encoding="utf-8"))
        candidates = data.get("candidates", [])
        if candidates:
            summary["best_frame_dir"] = candidates[0].get("frame_dir")
            summary["best_nearest_score"] = candidates[0].get("nearest_score")
            summary["best_mean_reference_score"] = candidates[0].get("mean_reference_score")
            summary["nearest_real_label"] = candidates[0].get("nearest_real_label")
    return summary


def _step_checks(step_report: dict[str, object]) -> list[dict[str, object]]:
    extents = [float(value) for value in step_report["bbox_extent_mm"]]
    radii = {float(item["radius_mm"]) for item in step_report["common_circle_or_cylindrical_radii_mm"]}
    return [
        _check(
            "step_has_many_points",
            int(step_report["cartesian_point_count"]) > 500,
            f"{step_report['cartesian_point_count']} CARTESIAN_POINT entries",
        ),
        _check(
            "step_m5010_ring_scale",
            max(extents) > 70.0 and any(24.0 <= radius <= 35.0 for radius in radii),
            f"extent_mm={extents}, common_radii_mm={sorted(radii)[:12]}",
        ),
    ]


def _all_checks(report: dict[str, object]) -> Iterable[dict[str, object]]:
    yield from report["camera_patch"]["checks"]
    yield from report["step"]["checks"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument(
        "--step",
        type=Path,
        default=Path(DEFAULT_STEP) if DEFAULT_STEP else None,
        required=DEFAULT_STEP is None,
        help="Dex1-1 D405 bracket STEP file; defaults to D405_BRACKET_STEP.",
    )
    parser.add_argument("--real-dir", type=Path, action="append", default=[])
    parser.add_argument("--sim-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--distribution-output", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--rank-by", choices=("nearest", "mean-reference"), default="mean-reference")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.patch.exists():
        raise FileNotFoundError(args.patch)
    if not args.step.exists():
        raise FileNotFoundError(args.step)

    camera_patch = _camera_patch_report(args.patch)
    step = _parse_step_points(args.step)
    step["checks"] = _step_checks(step)

    distribution_output = args.distribution_output
    if distribution_output is None:
        distribution_output = args.output.with_name(args.output.stem + "_distribution.json")
    distribution = _run_distribution_compare(
        args.real_dir,
        args.sim_root,
        distribution_output,
        args.contact_sheet,
        args.rank_by,
    )

    report: dict[str, object] = {
        "version": 1,
        "method": [
            "Use Unitree Device.md to select the G1 wrist RealSense D405 + Dex1-1 bracket hardware path.",
            "Use D405 RGB FOV and the USD pinhole camera equation to compute focal/aperture intrinsics.",
            "Use the Dex1-1 D405 STEP bracket as the mechanical mount source and verify M5010-ring-scale geometry.",
            "Use real flip-table wrist frames only as a distribution gate for remaining simulator-link and reset-pose ambiguity.",
        ],
        "camera_patch": camera_patch,
        "step": step,
        "distribution_compare": distribution,
        "limitations": [
            "The STEP bracket alone does not define the transform from the real M5010 wrist motor frame to the simulator left_wrist_yaw_link/right_wrist_yaw_link frames.",
            "A fully metric extrinsic still requires the real TF tree or a hand-eye calibration capture with an AprilTag/checkerboard target.",
        ],
    }
    checks = list(_all_checks(report))
    report["passed"] = all(bool(check["passed"]) for check in checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    status = "PASS" if report["passed"] else "FAIL"
    print(f"{status}: wrote {args.output}")
    if not report["passed"] and args.fail_on_warning:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
