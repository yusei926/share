#!/usr/bin/env python3
"""Compose an episode table reset with a consensus head-stereo probe mount."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.flip_table_data_augmentation.io_utils import atomic_write_json


SCHEMA_VERSION = "team_ramen_flip_table_fixed_scene_probe_candidate/v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compose(scene: dict[str, Any], consensus: dict[str, Any]) -> dict[str, Any]:
    if scene.get("schema_version") != "team_ramen_flip_table_source_scene_candidate/v1":
        raise ValueError("scene candidate has an unsupported schema")
    candidates = scene.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("scene candidate must contain exactly one table reset")
    if consensus.get("schema_version") != "team_ramen_flip_table_source_head_mount_consensus/v1":
        raise ValueError("head-mount consensus has an unsupported schema")
    if consensus.get("accepted_for_fixed_scene_probe") is not True:
        raise ValueError("head-mount consensus did not pass its cross-episode probe gate")
    offset = consensus.get("shared_head_stereo_offset_local_m")
    rotation = consensus.get("shared_head_stereo_rotation_rpy_deg")
    if not isinstance(offset, list) or len(offset) != 3 or not isinstance(rotation, list) or len(rotation) != 3:
        raise ValueError("head-mount consensus has no finite shared correction")
    candidate = dict(candidates[0])
    candidate["head_stereo_offset_local_m"] = offset
    candidate["head_stereo_rotation_rpy_deg"] = rotation
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_use": "forbidden: fixed-scene calibration probe only",
        "table_scene_candidate": scene.get("source_alignment"),
        "head_mount_consensus": consensus.get("reports"),
        "candidate": candidate,
        "accepted_for_fixed_scene_probe": True,
        "accepted_for_shared_simulator_default": False,
        "remaining_requirement": "held-out camera metrics using this unchanged candidate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-candidate", type=Path, required=True)
    parser.add_argument("--head-mount-consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compose(
        _load(args.scene_candidate.expanduser().resolve()),
        _load(args.head_mount_consensus.expanduser().resolve()),
    )
    atomic_write_json(args.output.expanduser().resolve(), result)
    print(json.dumps({"candidate": result["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
