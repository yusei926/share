#!/usr/bin/env python3
"""Verify the mandatory Mimic pilot and production acceptance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json
from data.flip_table_data_augmentation.provenance import CandidateLedger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--production-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "team_ramen_flip_table_mimic_run/v1":
        raise ValueError(f"unsupported Mimic run manifest: {path}")
    if value.get("status") != "finished":
        raise ValueError(f"Mimic run did not finish: {path}")
    result = value.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"Mimic run has no result: {path}")
    for key in ("attempted", "accepted", "rejected"):
        if not isinstance(result.get(key), int) or int(result[key]) < 0:
            raise ValueError(f"Mimic run has invalid {key}: {path}")
    if int(result["attempted"]) != int(result["accepted"]) + int(result["rejected"]):
        raise ValueError(f"Mimic run counters do not reconcile: {path}")
    return value


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    pilot = _load_manifest(args.pilot_manifest)
    production = [_load_manifest(path) for path in args.production_manifest]
    expected_runtime = config.runtime.container_digest
    for manifest in (pilot, *production):
        if manifest.get("config_sha256") != config.digest:
            raise ValueError("Mimic manifest config digest differs from active pipeline")
        if manifest.get("container_digest") != expected_runtime:
            raise ValueError("Mimic manifest container digest differs from V1 pin")

    pilot_result = pilot["result"]
    if int(pilot.get("requested_trials", -1)) != 100:
        raise ValueError("pilot must execute exactly 100 trials")
    if int(pilot_result["attempted"]) != 100:
        raise ValueError("pilot must account for exactly 100 attempts")
    if int(pilot_result["accepted"]) < 50:
        raise ValueError("pilot accepted fewer than 50 physical successes")

    ledger = CandidateLedger(args.ledger_root)
    runtime_manifest_digest = str(pilot.get("runtime_digest", ""))
    if len(runtime_manifest_digest) != 64:
        raise ValueError("pilot runtime manifest digest is invalid")
    if any(item.get("runtime_digest") != runtime_manifest_digest for item in production):
        raise ValueError("production runs must use the pilot's exact verified runtime manifest")
    accepted = [
        record
        for record in ledger.list_records()
        if record.status in {"validated", "rendered", "exported"}
        and record.config_sha256 == config.digest
        and record.runtime_digest == runtime_manifest_digest
    ]
    # Candidate records store a bare SHA-256 while OCI manifests include the
    # conventional prefix. Every accepted trajectory needs immutable physics
    # evidence and an FK validation report before it can count.
    accepted = [
        record
        for record in accepted
        if "physical_randomization" in record.payload
        and isinstance(record.payload.get("action_fk_report"), dict)
        and record.payload["action_fk_report"].get("pass") is True
    ]
    candidate_ids = {record.candidate_id for record in accepted}
    production_attempted = sum(int(item["result"]["attempted"]) for item in production)
    report = {
        "schema_version": "team_ramen_flip_table_mimic_release_gate/v1",
        "status": (
            "passed"
            if len(candidate_ids) >= int(config.raw["generation"]["successful_trajectories_min"])
            else "failed"
        ),
        "pilot": {
            "manifest": str(args.pilot_manifest.resolve()),
            "attempted": int(pilot_result["attempted"]),
            "accepted": int(pilot_result["accepted"]),
            "acceptance_rate": float(pilot_result["acceptance_rate"]),
        },
        "production": {
            "manifests": [str(path.resolve()) for path in args.production_manifest],
            "attempted": production_attempted,
            "validated_physical_trajectory_count": len(candidate_ids),
            "required_validated_physical_trajectory_count": int(
                config.raw["generation"]["successful_trajectories_min"]
            ),
        },
        "config_sha256": config.digest,
        "runtime_digest": expected_runtime,
        "runtime_manifest_sha256": runtime_manifest_digest,
        "ledger_root": str(args.ledger_root.resolve()),
    }
    atomic_write_json(args.output, report)
    if report["status"] != "passed":
        raise SystemExit(
            "Mimic production gate is not complete; see "
            f"{args.output} ({len(candidate_ids)}/"
            f"{config.raw['generation']['successful_trajectories_min']} accepted trajectories)"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
