#!/usr/bin/env python3
"""Load the pinned GR00T checkpoint and run one non-robot synthetic inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.upper_policy.groot_pick_leg_contract import (
    CAMERA_KEYS,
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    MODEL_STATE_DIM,
    TASK_TEXT,
    extract_executable_action,
)
from model.subtask_policy_training.deployment.real_groot_n17_worker import Runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    stats = json.loads(
        (args.checkpoint / "statistics.json").read_text(encoding="utf-8")
    )["new_embodiment"]["state"]
    state = np.concatenate(
        (
            np.asarray(stats["robot_q"]["mean"], dtype=np.float32),
            np.asarray(stats["hand"]["mean"], dtype=np.float32),
        )
    )
    if state.shape != (MODEL_STATE_DIM,):
        raise ValueError(f"synthetic state must be [{MODEL_STATE_DIM}]")
    image = np.full((480, 640, 3), 127, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("could not construct synthetic JPEG")
    runtime = Runtime(args.checkpoint.resolve(), args.device)
    request = {
        "state": state,
        "cameras": {key: jpeg.tobytes() for key in CAMERA_KEYS},
        "task": TASK_TEXT,
    }
    times: list[float] = []
    for _ in range(args.iterations):
        action, inference_ms = runtime.predict(request)
        times.append(inference_ms)
        if action.shape != (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM):
            raise RuntimeError(f"unexpected decoded action shape: {action.shape}")
    executable = extract_executable_action(action)
    print(
        "groot-model-smoke-ok "
        f"decoded={action.shape} executable={executable.shape} "
        f"inference_ms={[round(value, 1) for value in times]} "
        f"finite={bool(np.isfinite(action).all())} "
        "robot_commands_sent=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
