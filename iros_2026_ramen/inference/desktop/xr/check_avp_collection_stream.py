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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_s <= 0.0:
        raise ValueError("--timeout-s must be positive")

    from data.flip_table_data_augmentation.teleop.upstream_compat import (
        install_logging_mp_compat,
    )

    install_logging_mp_compat()
    from teleimager.image_client import ImageClient

    client = ImageClient(host=args.host)
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
            head, _ = client.get_head_frame()
            left, _ = client.get_left_wrist_frame()
            right, _ = client.get_right_wrist_frame()
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
        print(
            "camera-streams-ok head_left=640x480 head_right=640x480 "
            "left_d405=640x480 right_d405=640x480"
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
