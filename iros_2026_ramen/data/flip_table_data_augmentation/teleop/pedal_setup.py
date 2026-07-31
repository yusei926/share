"""Calibrate PCsensor's three physical pedals without starting robot control.

Run this while the robot teleop process is stopped.  It stores an explicit
left=q, middle=s, right=r mapping in the current user's config directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .keyboard import (
    DEFAULT_FOOT_PEDAL_CONFIG,
    DEFAULT_FOOT_PEDAL_DEVICE,
    FootPedalBinding,
    FootPedalReader,
    write_foot_pedal_binding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, default=DEFAULT_FOOT_PEDAL_DEVICE)
    parser.add_argument("--config", type=Path, default=DEFAULT_FOOT_PEDAL_CONFIG)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    # The temporary valid binding only opens/grabs the device.  ``poll_key_code``
    # exposes raw presses, so its placeholder codes are never used as commands.
    reader_binding = FootPedalBinding(
        device=args.device,
        action_to_key_code={"r": 1, "s": 2, "q": 3},
    )
    labels = {
        "q": "LEFT pedal: controlled final release / quit (q)",
        "s": "MIDDLE pedal: record start/save (s)",
        "r": "RIGHT pedal: track / pause / re-anchor-resume (r)",
    }
    codes: dict[str, int] = {}
    print("Foot-pedal calibration only; no robot command is sent.", flush=True)
    with FootPedalReader(reader_binding) as reader:
        # Keep the physical prompt order left, middle, right even though the
        # control-action names themselves have a different lexical order.
        for action in ("q", "s", "r"):
            print(f"Press {labels[action]} within {args.timeout:g}s.", flush=True)
            code = reader.wait_for_key_code(args.timeout)
            if code in codes.values():
                raise ValueError("each physical pedal must emit a distinct key code")
            codes[action] = code
            print(f"  captured Linux key code {code} for {action}", flush=True)
    binding = FootPedalBinding(device=args.device, action_to_key_code=codes)
    write_foot_pedal_binding(args.config, binding)
    print(f"Saved foot-pedal mapping: {args.config}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
