#!/usr/bin/env python3
"""Validate a non-finalized Furniture-GR00T H100 candidate for simulator use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model.subtask_policy_training.gr00t.n17_contract import (
    validate_furniture_training_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-progress",
        choices=("enabled", "disabled"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_furniture_training_candidate(
        args.checkpoint,
        expected_progress_enabled=args.expected_progress == "enabled",
    )
    report["checkpoint"] = str(args.checkpoint.expanduser().resolve())
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
