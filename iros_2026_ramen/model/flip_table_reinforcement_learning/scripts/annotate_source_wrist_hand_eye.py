#!/usr/bin/env python3
"""Add one auditable named Table001 point to a D405 hand-eye workspace.

The tool never enables calibration acceptance. It records one human-identified
physical feature in a selected raw IR1/IR2 pair, including the exact image
paths and operator provenance. Static-table and rigid-mount confirmations stay
false until independently reviewed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2


def _parse_pixel(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",")]
    if len(values) != 2:
        raise ValueError("pixel must be u,v")
    return values


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def add_annotation(
    template: dict[str, Any],
    manifest: dict[str, Any],
    *,
    observation_id: str,
    table_point_name: str,
    ir1_pixel: list[float],
    ir2_pixel: list[float],
    source_task_frame_provenance: str,
) -> dict[str, Any]:
    """Return an updated template after validating one named point and view."""

    if not source_task_frame_provenance.strip() or source_task_frame_provenance.lower().startswith("replace "):
        raise ValueError("source_task_frame_provenance must describe the actual annotated MCAP frames")
    if template.get("table_is_static_confirmation") is not False:
        raise ValueError("annotation template must keep table_is_static_confirmation false")
    if template.get("d405_is_rigid_to_eef_confirmation") is not False:
        raise ValueError("annotation template must keep d405_is_rigid_to_eef_confirmation false")
    candidates = template.get("v1_table001_body_fiducial_candidates")
    if not isinstance(candidates, list):
        raise ValueError("template is missing V1 Table001 candidate points")
    candidate_by_name = {
        candidate.get("name"): candidate for candidate in candidates if isinstance(candidate, dict)
    }
    candidate = candidate_by_name.get(table_point_name)
    if not isinstance(candidate, dict) or not isinstance(candidate.get("table_point_m"), list):
        raise ValueError(f"unknown V1 Table001 point: {table_point_name}")
    observations = template.get("wrist_table_observations")
    if not isinstance(observations, list):
        raise ValueError("template is missing wrist_table_observations")
    observation = next(
        (item for item in observations if isinstance(item, dict) and item.get("observation_id") == observation_id),
        None,
    )
    if observation is None:
        raise ValueError(f"unknown workspace observation: {observation_id}")
    frame = next(
        (item for item in manifest.get("frames", []) if item.get("observation_id") == observation_id),
        None,
    )
    if not isinstance(frame, dict):
        raise ValueError(f"workspace manifest has no frame for {observation_id}")
    for pixel, name in ((ir1_pixel, "ir1_pixel"), (ir2_pixel, "ir2_pixel")):
        if len(pixel) != 2 or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in pixel
        ):
            raise ValueError(f"{name} must be two finite numbers")
    for image_key, pixel in (("ir1_image", ir1_pixel), ("ir2_image", ir2_pixel)):
        image = cv2.imread(str(frame.get(image_key)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"could not load {image_key} for {observation_id}")
        height, width = image.shape[:2]
        if not 0.0 <= pixel[0] < width or not 0.0 <= pixel[1] < height:
            raise ValueError(f"{image_key} pixel is outside its raw image bounds")
    fiducials = observation.get("table_fiducials")
    if not isinstance(fiducials, list):
        raise ValueError("table_fiducials must be a list")
    if any(item.get("table_point_name") == table_point_name for item in fiducials if isinstance(item, dict)):
        raise ValueError(f"{table_point_name} is already annotated for {observation_id}")
    fiducials.append(
        {
            "table_point_name": table_point_name,
            "table_point_m": candidate["table_point_m"],
            "ir1_pixel_raw": ir1_pixel,
            "ir2_pixel_raw": ir2_pixel,
            "ir1_image": frame.get("ir1_image"),
            "ir2_image": frame.get("ir2_image"),
            "pair_midpoint_episode_time_s": frame.get("pair_midpoint_episode_time_s"),
        }
    )
    template["source_task_frame_provenance"] = source_task_frame_provenance.strip()
    return template


def _interactive_pixels(frame: dict[str, Any]) -> tuple[list[float], list[float]]:
    ir1 = cv2.imread(str(frame["ir1_image"]), cv2.IMREAD_COLOR)
    ir2 = cv2.imread(str(frame["ir2_image"]), cv2.IMREAD_COLOR)
    if ir1 is None or ir2 is None or ir1.shape != ir2.shape:
        raise RuntimeError("could not load equally sized IR1/IR2 images")
    height, width = ir1.shape[:2]
    canvas = cv2.hconcat((ir1, ir2))
    selected: list[list[float]] = []
    window = "D405 IR annotation: click IR1 then IR2; r resets, q cancels"

    def redraw() -> None:
        view = canvas.copy()
        for index, point in enumerate(selected):
            offset = 0 if index == 0 else width
            cv2.circle(view, (int(round(point[0])) + offset, int(round(point[1]))), 5, (0, 255, 0), 2)
        cv2.imshow(window, view)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or len(selected) >= 2 or not 0 <= y < height:
            return
        expected_right = len(selected) == 1
        if expected_right != (x >= width):
            return
        selected.append([float(x - width if expected_right else x), float(y)])
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while len(selected) < 2:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("r"):
            selected.clear()
            redraw()
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError("annotation cancelled")
    cv2.destroyWindow(window)
    return selected[0], selected[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--workspace-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--table-point-name", required=True)
    parser.add_argument("--source-task-frame-provenance", required=True)
    pixels = parser.add_mutually_exclusive_group(required=True)
    pixels.add_argument("--interactive", action="store_true")
    pixels.add_argument("--ir1-pixel", help="raw IR1 u,v; requires --ir2-pixel")
    parser.add_argument("--ir2-pixel", help="raw IR2 u,v; required with --ir1-pixel")
    args = parser.parse_args()
    if args.interactive:
        if args.ir2_pixel is not None:
            parser.error("--interactive cannot be combined with --ir2-pixel")
    elif args.ir2_pixel is None:
        parser.error("--ir1-pixel requires --ir2-pixel")

    template = _load_object(args.template)
    manifest = _load_object(args.workspace_manifest)
    if args.interactive:
        frame = next(
            (item for item in manifest.get("frames", []) if item.get("observation_id") == args.observation_id),
            None,
        )
        if not isinstance(frame, dict):
            parser.error(f"workspace manifest has no frame for {args.observation_id}")
        ir1_pixel, ir2_pixel = _interactive_pixels(frame)
    else:
        ir1_pixel = _parse_pixel(args.ir1_pixel)
        ir2_pixel = _parse_pixel(args.ir2_pixel)
    updated = add_annotation(
        template,
        manifest,
        observation_id=args.observation_id,
        table_point_name=args.table_point_name,
        ir1_pixel=ir1_pixel,
        ir2_pixel=ir2_pixel,
        source_task_frame_provenance=args.source_task_frame_provenance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote annotated workspace to {args.output}")


if __name__ == "__main__":
    main()
