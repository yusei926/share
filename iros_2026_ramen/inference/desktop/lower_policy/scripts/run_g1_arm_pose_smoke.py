"""Arm-only smoke test for the official G1 Regular-Mode ``rt/arm_sdk`` path.

This deliberately does not create a walking command.  It is intended after
the camera/orchestrator Step 2a test, when the action sink itself needs a
small, observable verification before an end-to-end walk-plus-arm run.
"""

from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True, help="G1 DDS NIC")
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="horizontal-pose hold duration in seconds (default: 2)",
    )
    parser.add_argument(
        "--shoulder-roll",
        type=float,
        default=1.0,
        help=(
            "absolute shoulder-roll target [rad]: left=+value, right=-value "
            "(default: 1.0; bounded to [0, 1.5])"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the interactive physical-safety confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if not 0.0 <= args.shoulder_roll <= 1.5:
        raise SystemExit("--shoulder-roll must be in [0, 1.5]")

    from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1ArmActuator
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator
    from inference.desktop.lower_policy.skills.sample_vla_skill import SampleVLASkill

    # This creates the SDK DDS participant but never calls a locomotion API.
    G1SDKWalkActuator(interface=args.interface)
    arm = G1ArmActuator()
    target = SampleVLASkill("arm_pose_smoke").step({})
    target[1] = args.shoulder_roll
    target[8] = -args.shoulder_roll
    print(
        "[preflight] arm_sdk path; no walking command; "
        f"mode_machine={arm._mode_machine}; duration={args.duration:g}s; "
        f"shoulder_roll=+/-{args.shoulder_roll:g} rad"
    )
    if not args.yes:
        input(
            "Harness / E-stop / arm clearance confirmed. "
            "Press Enter to raise both arms, Ctrl-C to cancel: "
        )

    try:
        before = arm.read_arm_positions()
        print(
            "[state] before "
            f"left_shoulder_roll={before[1]:+.3f} "
            f"right_shoulder_roll={before[8]:+.3f} rad"
        )
        arm.start()
        arm.send_action(target)
        print("[run] horizontal arm target publishing")
        deadline = time.monotonic() + args.duration
        next_report = time.monotonic()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_report:
                current = arm.read_arm_positions()
                print(
                    "[state] command-active "
                    f"left_shoulder_roll={current[1]:+.3f} "
                    f"right_shoulder_roll={current[8]:+.3f} rad"
                )
                next_report = now + 0.25
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("[stop] interrupted")
    finally:
        arm.stop()
        print("[stop] arm_sdk publisher stopped; no walking command was sent")


if __name__ == "__main__":
    main()
