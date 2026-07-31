#!/usr/bin/env python3
"""Validate the four real camera views used by AVP flip-table collection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--rate-window-s", type=float, default=3.0)
    parser.add_argument("--minimum-unique-hz", type=float, default=28.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive")
    if args.rate_window_s <= 0.0:
        raise ValueError("--rate-window-s must be positive")
    if not 0.0 < args.minimum_unique_hz <= 30.0:
        raise ValueError("--minimum-unique-hz must lie in (0, 30]")

    from data.flip_table_data_augmentation.teleop.upstream_compat import (
        install_logging_mp_compat,
    )
    from data.flip_table_data_augmentation.teleop.real.teleimager import (
        create_image_client,
        receive_teleimage,
    )

    install_logging_mp_compat()
    # This standalone checker validates decoded geometry. The teleoperation
    # backend itself consumes JPEG-only latest samples to avoid asynchronous
    # JPEG/BGR generation mismatches.
    client = create_image_client(args.host, request_bgr=True)
    try:
        config = client.get_cam_config()
        expected_config = {
            "head_camera": ([480, 1280], True),
            "left_wrist_camera": ([480, 640], False),
            "right_wrist_camera": ([480, 640], False),
        }
        for role, (shape, binocular) in expected_config.items():
            value = config.get(role, {})
            if value.get("enable_zmq") is not True:
                raise RuntimeError(f"{role} ZMQ stream is disabled")
            if value.get("image_shape") != shape or value.get("binocular") is not binocular:
                raise RuntimeError(f"{role} geometry differs from {shape}, binocular={binocular}")

        deadline = time.monotonic() + args.timeout_s
        head = left = right = None
        while time.monotonic() < deadline:
            head = receive_teleimage(client.get_head_frame).bgr
            left = receive_teleimage(client.get_left_wrist_frame).bgr
            right = receive_teleimage(client.get_right_wrist_frame).bgr
            if head is not None and left is not None and right is not None:
                break
            time.sleep(0.02)
        if head is None or np.asarray(head).shape != (480, 1280, 3):
            raise RuntimeError("head stream did not produce 1280x480 side-by-side RGB")
        if left is None or np.asarray(left).shape != (480, 640, 3):
            raise RuntimeError("left D405 did not produce 640x480 RGB")
        if right is None or np.asarray(right).shape != (480, 640, 3):
            raise RuntimeError("right D405 did not produce 640x480 RGB")
        left_eye = np.asarray(head)[:, :640]
        right_eye = np.asarray(head)[:, 640:]
        if np.array_equal(left_eye, right_eye):
            raise RuntimeError("head-left and head-right are byte-identical; true stereo is unavailable")

        previous = {
            "head": np.asarray(head).copy(),
            "left_d405": np.asarray(left).copy(),
            "right_d405": np.asarray(right).copy(),
        }
        unique_transitions = {role: 0 for role in previous}
        started = time.monotonic()
        while time.monotonic() - started < args.rate_window_s:
            current = {
                "head": receive_teleimage(client.get_head_frame).bgr,
                "left_d405": receive_teleimage(client.get_left_wrist_frame).bgr,
                "right_d405": receive_teleimage(client.get_right_wrist_frame).bgr,
            }
            for role, frame in current.items():
                if frame is None:
                    continue
                array = np.asarray(frame)
                if not np.array_equal(array, previous[role]):
                    unique_transitions[role] += 1
                    previous[role] = array.copy()
            time.sleep(0.002)
        elapsed_s = time.monotonic() - started
        unique_hz = {
            role: count / elapsed_s for role, count in unique_transitions.items()
        }
        slow = {
            role: rate
            for role, rate in unique_hz.items()
            if rate < args.minimum_unique_hz
        }
        if slow:
            rendered = ", ".join(
                f"{role}={rate:.2f}Hz" for role, rate in sorted(slow.items())
            )
            raise RuntimeError(
                "camera stream contains duplicate-padded or slow frames: "
                f"{rendered}; minimum={args.minimum_unique_hz:.2f}Hz"
            )
        print(
            "camera-streams-ok head_left=640x480 head_right=640x480 "
            "left_d405=640x480 right_d405=640x480 "
            + " ".join(
                f"{role}_unique_hz={rate:.2f}"
                for role, rate in sorted(unique_hz.items())
            )
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
