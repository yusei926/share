#!/usr/bin/env python3
"""Run a deliberately small, single-Dex1 DDS motion diagnostic.

Unlike the upstream sinusoidal example, this probe never commands the arms,
waist, lower body, or the opposite gripper.  It ramps one gripper by a bounded
angle and then ramps back to the measured starting position while recording
feedback.  Actuation requires both ``--execute`` and an interactive safety
confirmation.
"""

from __future__ import annotations

import argparse
import math
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_


DEX1_MIN_RAD = 0.0
DEX1_MAX_RAD = 5.4
DEFAULT_DELTA_RAD = 0.18
MAX_DIAGNOSTIC_DELTA_RAD = 0.25
PUBLISH_HZ = 200.0
KP = 5.0
KD = 0.05


def _read_position(subscriber: ChannelSubscriber, timeout_s: float) -> float:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        message = subscriber.Read()
        if message is not None and getattr(message, "states", None):
            position = float(message.states[0].q)
            if math.isfinite(position):
                return position
        time.sleep(0.002)
    raise TimeoutError("Dex1 state was not received")


def _message(target_rad: float) -> MotorCmds_:
    command = unitree_go_msg_dds__MotorCmd_()
    command.q = float(target_rad)
    command.dq = 0.0
    command.tau = 0.0
    command.kp = KP
    command.kd = KD
    message = MotorCmds_()
    message.cmds = [command]
    return message


def _ramp(
    publisher: ChannelPublisher,
    subscriber: ChannelSubscriber,
    start_rad: float,
    target_rad: float,
    duration_s: float,
) -> list[float]:
    steps = max(1, round(duration_s * PUBLISH_HZ))
    period_s = 1.0 / PUBLISH_HZ
    feedback: list[float] = []
    next_tick = time.monotonic()
    for step in range(1, steps + 1):
        blend = step / steps
        result = publisher.Write(
            _message(start_rad + blend * (target_rad - start_rad))
        )
        if result is False:
            raise RuntimeError("Dex1 DDS publisher reported a write failure")
        message = subscriber.Read()
        if message is not None and getattr(message, "states", None):
            value = float(message.states[0].q)
            if math.isfinite(value):
                feedback.append(value)
        next_tick += period_s
        time.sleep(max(0.0, next_tick - time.monotonic()))
    return feedback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move only one Dex1-1 by a small bounded angle and return."
    )
    parser.add_argument("--interface", required=True, help="CycloneDDS NIC name")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--delta-rad", type=float, default=DEFAULT_DELTA_RAD)
    parser.add_argument("--ramp-seconds", type=float, default=1.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required actuation gate; without it this command exits read-only",
    )
    args = parser.parse_args()
    if not 0.0 < args.delta_rad <= MAX_DIAGNOSTIC_DELTA_RAD:
        parser.error(f"--delta-rad must be in (0, {MAX_DIAGNOSTIC_DELTA_RAD}]")
    if args.ramp_seconds < 0.5:
        parser.error("--ramp-seconds must be at least 0.5")

    ChannelFactoryInitialize(0, args.interface)
    state_topic = f"rt/dex1/{args.side}/state"
    command_topic = f"rt/dex1/{args.side}/cmd"
    subscriber = ChannelSubscriber(state_topic, MotorStates_)
    subscriber.Init()
    initial_rad = _read_position(subscriber, 5.0)
    print(f"[read-only] {args.side}_q={initial_rad:.6f} rad")
    if not args.execute:
        print("No command sent. Add --execute only after clearing the gripper.")
        return 0

    from inference.desktop.lower_policy.actuators.g1_control_lock import (
        acquire_g1_control_lock,
    )

    acquire_g1_control_lock()

    # Move toward the side with more available travel, preserving a generous
    # margin from both mechanical endpoints.
    signed_delta = args.delta_rad if initial_rad <= DEX1_MAX_RAD / 2.0 else -args.delta_rad
    target_rad = initial_rad + signed_delta
    if not DEX1_MIN_RAD + 0.25 <= target_rad <= DEX1_MAX_RAD - 0.25:
        raise RuntimeError("bounded target is too close to a Dex1 endpoint")
    confirmation = input(
        f"Clear the {args.side} gripper of fingers/objects. "
        f"Type MOVE {args.side.upper()} to move {signed_delta:+.3f} rad and return: "
    )
    if confirmation.strip() != f"MOVE {args.side.upper()}":
        print("Cancelled; no command sent.")
        return 2

    publisher = ChannelPublisher(command_topic, MotorCmds_)
    publisher.Init()
    print(f"[probe] {initial_rad:.6f} -> {target_rad:.6f} -> {initial_rad:.6f} rad")
    try:
        # Establish the writer/reader match at the current position before
        # asking for displacement. This cannot move a correctly calibrated
        # gripper and avoids losing the first samples during DDS discovery.
        _ramp(publisher, subscriber, initial_rad, initial_rad, 0.5)
        outward = _ramp(
            publisher, subscriber, initial_rad, target_rad, args.ramp_seconds
        )
        reached_rad = _read_position(subscriber, 1.0)
        returned = _ramp(
            publisher, subscriber, reached_rad, initial_rad, args.ramp_seconds
        )
    except BaseException:
        # Ctrl-C or a transient diagnostic failure must not intentionally leave
        # the last requested value displaced.  Make one bounded best-effort
        # return using fresh feedback, then preserve the original exception.
        try:
            recovery_start = _read_position(subscriber, 0.5)
            _ramp(
                publisher,
                subscriber,
                recovery_start,
                initial_rad,
                args.ramp_seconds,
            )
            print("[recovery] returned the selected Dex1 toward its start position")
        except Exception as recovery_error:
            print(f"[recovery] failed: {recovery_error}")
        raise
    final_rad = _read_position(subscriber, 1.0)
    observed = outward + [reached_rad] + returned + [final_rad]
    span_rad = max(observed) - min(observed) if observed else 0.0
    print(
        f"[result] side={args.side} commanded_delta={abs(signed_delta):.6f} "
        f"observed_span={span_rad:.6f} reached_q={reached_rad:.6f} "
        f"final_q={final_rad:.6f}"
    )
    if span_rad < min(0.05, abs(signed_delta) * 0.5):
        print("FAIL: command topic was published but the gripper did not respond.")
        return 1
    print("PASS: the selected Dex1 command/state path responded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
