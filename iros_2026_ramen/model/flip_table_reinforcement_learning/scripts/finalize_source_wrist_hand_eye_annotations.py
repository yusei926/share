#!/usr/bin/env python3
"""Record a human review before D405 hand-eye calibration can be accepted.

This does not validate geometry or make the resulting calibration usable by a
policy. It records the two physical confirmations that raw MCAP cannot infer:
the table was static and the D405 was rigidly mounted to the recorded EEF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REVIEW_SCHEMA_VERSION = "flip_table_source_wrist_hand_eye_review/v1"


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def finalize_annotations(
    payload: dict[str, Any],
    *,
    reviewer_id: str,
    static_table_evidence: str,
    rigid_mount_evidence: str,
) -> dict[str, Any]:
    """Add one auditable review record to an unconfirmed annotation payload."""

    if payload.get("table_is_static_confirmation") is not False:
        raise ValueError("annotations must be unconfirmed before finalization")
    if payload.get("d405_is_rigid_to_eef_confirmation") is not False:
        raise ValueError("annotations must be unconfirmed before finalization")
    observations = payload.get("wrist_table_observations")
    if not isinstance(observations, list):
        raise ValueError("annotations need wrist_table_observations")
    reviewed_ids = []
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("each wrist_table_observation must be an object")
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("each wrist_table_observation needs observation_id")
        fiducials = observation.get("table_fiducials")
        if not isinstance(fiducials, list) or not fiducials:
            continue
        reviewed_ids.append(observation_id)
    if len(set(reviewed_ids)) < 3:
        raise ValueError("review needs annotated fiducials from at least three EEF views")

    result = dict(payload)
    result["table_is_static_confirmation"] = True
    result["d405_is_rigid_to_eef_confirmation"] = True
    result["acceptance_review"] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "reviewer_id": _nonempty(reviewer_id, name="reviewer_id"),
        "static_table_evidence": _nonempty(
            static_table_evidence, name="static_table_evidence"
        ),
        "d405_rigid_mount_evidence": _nonempty(
            rigid_mount_evidence, name="rigid_mount_evidence"
        ),
        "reviewed_observation_ids": sorted(set(reviewed_ids)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--static-table-evidence", required=True)
    parser.add_argument("--rigid-mount-evidence", required=True)
    args = parser.parse_args()
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("annotations JSON must contain an object")
    finalized = finalize_annotations(
        payload,
        reviewer_id=args.reviewer_id,
        static_table_evidence=args.static_table_evidence,
        rigid_mount_evidence=args.rigid_mount_evidence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finalized, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "review": finalized["acceptance_review"]}))


if __name__ == "__main__":
    main()
