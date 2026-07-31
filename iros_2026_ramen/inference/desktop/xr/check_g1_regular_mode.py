#!/usr/bin/env python3
"""Read-only preflight for the G1 Regular/walking high-level FSM."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator


def validate_regular_mode_status(
    *,
    fsm_id: int,
    fsm_mode: int,
    expected_fsm_id: int,
    allowed_fsm_modes: tuple[int, ...],
) -> None:
    """Validate the high-level state without conflating arm-sdk ownership.

    A physical G1 observed in this repository reports mode 0 before
    ``rt/arm_sdk`` is acquired and mode 1 while the official motion-mode arm
    overlay is active.  Both retain FSM 501.  Callers must opt into mode 1;
    the default startup preflight remains strictly mode 0.
    """

    if fsm_id != expected_fsm_id or fsm_mode not in allowed_fsm_modes:
        allowed = ",".join(str(value) for value in allowed_fsm_modes)
        raise RuntimeError(
            "G1 is not in the required Regular/walking FSM: "
            f"actual=({fsm_id},{fsm_mode}) "
            f"expected_id={expected_fsm_id} allowed_modes=[{allowed}]"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--expected-fsm-id", type=int, default=501)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--expected-fsm-mode",
        type=int,
        help="deprecated single-mode spelling retained for operator scripts",
    )
    mode_group.add_argument(
        "--allowed-fsm-mode",
        type=int,
        action="append",
        dest="allowed_fsm_modes",
        help=(
            "allowed high-level FSM submode; repeat to allow more than one "
            "(default: startup-safe mode 0 only)"
        ),
    )
    args = parser.parse_args()

    # Construction initializes the read-only LocoClient. No command method is
    # called by this diagnostic.
    status = G1SDKWalkActuator(interface=args.interface).get_loco_status()
    allowed_fsm_modes = (
        (args.expected_fsm_mode,)
        if args.expected_fsm_mode is not None
        else tuple(args.allowed_fsm_modes or (0,))
    )
    validate_regular_mode_status(
        fsm_id=status.fsm_id,
        fsm_mode=status.fsm_mode,
        expected_fsm_id=args.expected_fsm_id,
        allowed_fsm_modes=allowed_fsm_modes,
    )
    print(
        "g1-regular-mode-ok "
        f"fsm_id={status.fsm_id} fsm_mode={status.fsm_mode} "
        f"allowed_fsm_modes={list(allowed_fsm_modes)} "
        f"sport_api={status.client_api_version}/{status.server_api_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
