#!/usr/bin/env python3
"""Fail closed when a shared camera candidate regresses on another episode.

Silhouette measurements are diagnostic evidence only.  This tool deliberately
does not turn an RGB score into a camera-mount estimate: it only decides
whether a proposed *shared* correction improved every supplied independent
episode relative to the same V1 baseline.  A rejected proposal must not be
copied into a simulator default or a policy input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_camera_candidate_assessment/v1"
ALIGNMENT_SCHEMA_VERSION = "team_ramen_table_silhouette_alignment/v2"
_METRICS = (
    ("mask_iou", "higher"),
    ("edge_distance_symmetric_px", "lower"),
)


@dataclass(frozen=True)
class Alignment:
    """One typed visual-alignment report with its evidence paths."""

    path: Path
    payload: dict[str, Any]

    @property
    def episode_key(self) -> str:
        # Calibration runs often materialize the same immutable real frame in
        # separate output directories.  Compare its bytes rather than that
        # incidental path, while still failing closed if evidence is absent.
        image = Path(str(self.payload["real_image"])).expanduser()
        if not image.is_file():
            raise FileNotFoundError(f"alignment real-image evidence is missing: {image}")
        return hashlib.sha256(image.read_bytes()).hexdigest()


def _load_alignment(path: Path) -> Alignment:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != ALIGNMENT_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a {ALIGNMENT_SCHEMA_VERSION} report")
    for key in ("real_image", "simulated_image", "roi_xyxy"):
        if key not in payload:
            raise ValueError(f"{path} omits {key}")
    for key, _ in _METRICS:
        value = payload.get(key)
        if not isinstance(value, (int, float)):
            raise ValueError(f"{path} omits numeric {key}")
    return Alignment(path.resolve(), payload)


def assess(pairs: list[tuple[Alignment, Alignment]]) -> dict[str, Any]:
    """Return a conservative shared-camera acceptance decision."""

    if len(pairs) < 2:
        raise ValueError("at least two independent episode pairs are required")
    seen = set()
    results = []
    all_improved = True
    for baseline, candidate in pairs:
        if baseline.episode_key != candidate.episode_key:
            raise ValueError(
                "baseline and candidate must compare the same immutable real image: "
                f"{baseline.path} != {candidate.path}"
            )
        if tuple(baseline.payload["roi_xyxy"]) != tuple(candidate.payload["roi_xyxy"]):
            raise ValueError("baseline and candidate ROI differ")
        if baseline.episode_key in seen:
            raise ValueError("duplicate real-image evidence is not independent")
        seen.add(baseline.episode_key)
        metrics = {}
        episode_improved = True
        for name, direction in _METRICS:
            before = float(baseline.payload[name])
            after = float(candidate.payload[name])
            improved = after > before if direction == "higher" else after < before
            metrics[name] = {
                "direction": direction,
                "baseline": before,
                "candidate": after,
                "delta": after - before,
                "improved": improved,
            }
            episode_improved = episode_improved and improved
        all_improved = all_improved and episode_improved
        results.append(
            {
                "real_image": str(baseline.payload["real_image"]),
                "real_image_sha256": baseline.episode_key,
                "baseline_alignment": str(baseline.path),
                "candidate_alignment": str(candidate.path),
                "metrics": metrics,
                "improved_on_all_visual_metrics": episode_improved,
            }
        )
    accepted = bool(all_improved)
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: offline camera-candidate assessment only",
        "candidate_is_shared": True,
        "episodes": results,
        "accepted_for_shared_simulator_default": accepted,
        "decision": (
            "accepted_for_next_heldout_camera_gate"
            if accepted
            else "rejected_cross_episode_visual_regression"
        ),
        "limitations": [
            "A silhouette score is not an independent metric camera calibration.",
            "Acceptance here is only permission to run the separate held-out camera gate.",
            "No result may be used as a policy input, planner input, or runtime branch.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.baseline) != len(args.candidate):
        raise ValueError("--baseline and --candidate must have equal counts")
    pairs = [
        (_load_alignment(baseline.expanduser()), _load_alignment(candidate.expanduser()))
        for baseline, candidate in zip(args.baseline, args.candidate, strict=True)
    ]
    result = assess(pairs)
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
