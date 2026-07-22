#!/usr/bin/env python3
"""Prepare and score parallel fixed-scene camera calibration probes.

This is offline-only: it writes reset candidates and compares rendered RGB.
Neither candidates nor scores are available to a policy, planner, reward, or
inference-time branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.flip_table_data_augmentation.io_utils import atomic_write_json, atomic_write_text

from .compare_scene_candidates import _corner_error, _estimate
from .visual_alignment import compare_images


_PNP_CONFIDENCE_MIN = 0.10
_PNP_REPROJECTION_ERROR_MAX_PX = 8.0


def _is_reliable_pnp(estimate: dict[str, Any]) -> bool:
    """Return whether an RGB geometry estimate is safe to rank as a fit.

    A quadrilateral can land close to the real one while enclosing the wrong
    table feature.  Reprojection error is the guard against presenting that
    accidental match as a useful calibration candidate.
    """

    return (
        float(estimate["confidence"]) >= _PNP_CONFIDENCE_MIN
        and float(estimate["reprojection_error_px"]) <= _PNP_REPROJECTION_ERROR_MAX_PX
    )


def read_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or len(payload) > 64:
        raise ValueError("candidate JSON must be a non-empty list of at most 64 items")
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"candidate {index} must be an object")
        offset, yaw = item.get("offset_local_m"), item.get("yaw_rad")
        if not isinstance(offset, list) or len(offset) != 3:
            raise ValueError(f"candidate {index}.offset_local_m must be [x,y,z]")
        try:
            values, yaw_value = [float(value) for value in offset], float(yaw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {index} has non-numeric values") from exc
        if not np.isfinite([*values, yaw_value]).all():
            raise ValueError(f"candidate {index} must be finite")
        camera_offset = item.get("head_stereo_offset_local_m", [0.0, 0.0, 0.0])
        camera_rpy = item.get("head_stereo_rotation_rpy_deg", [0.0, 0.0, 0.0])
        if not isinstance(camera_offset, list) or len(camera_offset) != 3:
            raise ValueError(f"candidate {index}.head_stereo_offset_local_m must be [x,y,z]")
        if not isinstance(camera_rpy, list) or len(camera_rpy) != 3:
            raise ValueError(f"candidate {index}.head_stereo_rotation_rpy_deg must be [roll,pitch,yaw]")
        try:
            camera_offset_values = [float(value) for value in camera_offset]
            camera_rpy_values = [float(value) for value in camera_rpy]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"candidate {index} has non-numeric camera values") from exc
        if not np.isfinite([*camera_offset_values, *camera_rpy_values]).all():
            raise ValueError(f"candidate {index} camera values must be finite")
        candidates.append(
            {
                "label": str(item.get("label", f"candidate_{index:03d}")),
                "offset_local_m": values,
                "yaw_rad": yaw_value,
                "head_stereo_offset_local_m": camera_offset_values,
                "head_stereo_rotation_rpy_deg": camera_rpy_values,
            }
        )
    return candidates


def write_probe_environment(
    candidates: list[dict[str, Any]], *, output_dir: Path, replay_action_path: Path, frame_index: int
) -> dict[str, str]:
    if frame_index < 0:
        raise ValueError("frame index must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "FLIP_TABLE_SIM_OUTPUT_DIR": str(output_dir.resolve()),
        "FLIP_TABLE_TEST_NUM": "1",
        "FLIP_TABLE_EVAL_MODE": "nominal",
        "FLIP_TABLE_POLICY_NAME": "RecordedJointTargetPolicy",
        "FLIP_TABLE_REPLAY_ACTION_PATH": str(replay_action_path.resolve()),
        "FLIP_TABLE_SAVE_CAMERA_FRAMES": "true",
        "FLIP_TABLE_SAVE_RECORDED_CAMERA_GEOMETRY": "true",
        "FLIP_TABLE_CAMERA_FRAME_INDICES": str(frame_index),
        "FLIP_TABLE_CAMERA_FRAME_BATCH_EXPORT": "true",
        # Per-step scene diagnostics are intentionally single-environment
        # telemetry for an action replay. A parallel static probe has no
        # contact/trajectory fit to perform, so do not invoke that AVP-facing
        # callback for each candidate.
        "FLIP_TABLE_SAVE_CALIBRATION_SCENE_TRACE": "false",
        "FLIP_TABLE_CALIBRATION_NUM_ENVS": str(len(candidates)),
        "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON": json.dumps(candidates, separators=(",", ":")),
    }
    atomic_write_json(
        output_dir / "parallel_probe_manifest.json",
        {
            "schema_version": "team_ramen_parallel_scene_probe/v1",
            "policy_use": "forbidden: offline camera/scene calibration only",
            "frame_index": frame_index,
            "candidates": candidates,
            "rendered_camera_layout": "frame_XXXX/env_NNN/<role>_rgb.png",
            "head_left_saved_in_recorded_raw_geometry": True,
        },
    )
    lines = ["# Generated offline calibration environment. Source before run_eval.sh."]
    lines += [f"export {name}={shlex.quote(value)}" for name, value in environment.items()]
    atomic_write_text(output_dir / "parallel_probe.env", "\n".join(lines) + "\n")
    return environment


def score_probe(real_image: Path, frame_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("parallel probe manifest is missing candidates")
    recorded_geometry = manifest.get("head_left_saved_in_recorded_raw_geometry")
    if not isinstance(recorded_geometry, bool):
        raise ValueError("parallel probe manifest is missing recorded-geometry provenance")
    real = _estimate(real_image, real=True)
    real_corners = np.asarray(real["corners_px"], dtype=np.float64)
    scored: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        image = frame_dir / f"env_{index:03d}" / "head_left_rgb.png"
        entry: dict[str, Any] = {"environment_index": index, "candidate": candidate, "image": str(image)}
        try:
            simulated = _estimate(
                image, real=False, sim_recorded_geometry=recorded_geometry
            )
            rmse, correspondence = _corner_error(real_corners, np.asarray(simulated["corners_px"], dtype=np.float64))
            center_error = float(np.linalg.norm(np.asarray(real["center_px"]) - np.asarray(simulated["center_px"])))
            entry.update(
                {
                    "status": "scored",
                    "corner_rmse_px": rmse,
                    "center_error_px": center_error,
                    "simulated": simulated,
                    "pnp_reliable": _is_reliable_pnp(simulated),
                    "corner_correspondence": {"reversed": correspondence[0], "cyclic_shift": correspondence[1]},
                }
            )
            # PnP is only one diagnostic.  The real image can hide its outer
            # rim behind a hand or leg, in which case an interior brace yields
            # a misleading quadrilateral.  Rank such candidates by the
            # silhouette score, but retain the PnP provenance and never accept
            # a single-frame result as a calibration.
            entry["silhouette_alignment"] = compare_images(real_image, image)
        except Exception as exc:  # noqa: BLE001 - invalid candidates remain evidence.
            entry.update({"status": "invalid", "error": f"{type(exc).__name__}: {exc}"})
        scored.append(entry)
    scored.sort(
        key=lambda item: (
            item["status"] != "scored",
            item.get("silhouette_alignment", {}).get("edge_distance_symmetric_px", float("inf")),
            -item.get("silhouette_alignment", {}).get("mask_iou", -float("inf")),
            not item.get("pnp_reliable", False),
            item.get("corner_rmse_px", float("inf")),
        )
    )
    return {
        "schema_version": "team_ramen_parallel_scene_probe_score/v1",
        "policy_use": "forbidden: offline camera/scene calibration only",
        "real": real,
        "candidates": scored,
        "accepted_candidate": None,
        "decision": "diagnostic_only_pending_multiview_and_heldout_validation",
        "simulated_pnp_gate": {
            "confidence_min": _PNP_CONFIDENCE_MIN,
            "reprojection_error_px_max": _PNP_REPROJECTION_ERROR_MAX_PX,
        },
        "sim_recorded_geometry": recorded_geometry,
        "ranking": "silhouette edge distance then IoU; PnP retained as a fail-closed geometry diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    write_parser = commands.add_parser("write-env")
    write_parser.add_argument("--candidates", type=Path, required=True)
    write_parser.add_argument("--output-dir", type=Path, required=True)
    write_parser.add_argument("--replay-action-path", type=Path, required=True)
    write_parser.add_argument("--frame-index", type=int, default=136)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--real-image", type=Path, required=True)
    score_parser.add_argument("--frame-dir", type=Path, required=True)
    score_parser.add_argument("--manifest", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "write-env":
        environment = write_probe_environment(
            read_candidates(args.candidates.expanduser().resolve()),
            output_dir=args.output_dir.expanduser().resolve(),
            replay_action_path=args.replay_action_path.expanduser().resolve(),
            frame_index=args.frame_index,
        )
        print(json.dumps(environment, indent=2))
    else:
        report = score_probe(
            args.real_image.expanduser().resolve(), args.frame_dir.expanduser().resolve(), args.manifest.expanduser().resolve()
        )
        atomic_write_json(args.output.expanduser().resolve(), report)
        print(json.dumps({"decision": report["decision"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
