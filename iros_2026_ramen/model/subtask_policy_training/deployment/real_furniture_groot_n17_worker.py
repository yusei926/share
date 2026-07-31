#!/usr/bin/env python3
"""GPU worker for the finalized flip-table Furniture-GR00T N1.7 checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = (
    REPO_ROOT
    / "model"
    / "subtask_policy_training"
    / "lerobot_policy_furniture_groot"
)
for path in (REPO_ROOT, PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lerobot_policy_furniture_groot  # noqa: E402,F401
from evaluate.flip_table_simulation.groot_runtime.groot_inference_server import (  # noqa: E402
    GrootRuntime,
)
from inference.desktop.upper_policy.furniture_groot_contract import (  # noqa: E402
    CAMERA_KEYS,
    LEROBOT_VERSION,
    MODEL_ACTION_HORIZON,
    TASK_TEXT,
    VIDEO_HORIZON,
    extract_executable_action,
    validate_checkpoint_metadata,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _decode_rgb_history(jpegs: Sequence[bytes], role: str) -> np.ndarray:
    if len(jpegs) != VIDEO_HORIZON:
        raise ValueError(f"{role} requires exactly {VIDEO_HORIZON} JPEG frames")
    frames: list[np.ndarray] = []
    for index, payload in enumerate(jpegs):
        image = cv2.imdecode(
            np.frombuffer(bytes(payload), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None or image.shape != (480, 640, 3):
            shape = None if image is None else tuple(image.shape)
            raise ValueError(
                f"{role}[{index}] must decode to 640x480 BGR, got {shape}"
            )
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.ascontiguousarray(np.stack(frames, axis=0))


class Runtime:
    def __init__(self, checkpoint: Path, device: str, seed: int) -> None:
        if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(
                f"physical Furniture-GR00T inference requires lerobot=={LEROBOT_VERSION}"
            )
        self.contract = validate_checkpoint_metadata(checkpoint)
        self.runtime = GrootRuntime(
            checkpoint=checkpoint,
            device=device,
            # The model process must always return the complete H40 chunk.
            # The validated execution interval is applied by the controller.
            n_action_steps=MODEL_ACTION_HORIZON,
            seed=seed,
        )

    def reset(self) -> None:
        self.runtime.reset()

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (self.contract["state_dim"],) or not np.isfinite(state).all():
            raise ValueError(
                f"state must be finite [{self.contract['state_dim']}]"
            )
        camera_history = request.get("camera_history")
        if not isinstance(camera_history, dict) or set(camera_history) != set(CAMERA_KEYS):
            raise ValueError(f"camera keys must be exactly {CAMERA_KEYS}")
        if request.get("task") != TASK_TEXT:
            raise ValueError(f"task must exactly match training text {TASK_TEXT!r}")

        payload: dict[str, np.ndarray] = {
            "state": state,
            "task": np.asarray(TASK_TEXT),
        }
        for key in CAMERA_KEYS:
            payload[key.rsplit(".", 1)[-1]] = _decode_rgb_history(
                camera_history[key],
                key,
            )
        decoded, _normalized, elapsed = self.runtime.predict(payload)
        return extract_executable_action(decoded), elapsed * 1000.0


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = Runtime(args.checkpoint.resolve(), args.device, args.seed)
    send_message(
        sys.stdout.buffer,
        {
            "type": "ready",
            "contract": runtime.contract,
            "device": str(runtime.runtime.device),
        },
    )
    while True:
        request = receive_message(sys.stdin.buffer)
        if not isinstance(request, dict):
            raise TypeError("policy request must be a mapping")
        message_type = request.get("type")
        if message_type == "close":
            send_message(sys.stdout.buffer, {"type": "closed"})
            return 0
        if message_type == "reset":
            runtime.reset()
            send_message(sys.stdout.buffer, {"type": "reset"})
            continue
        if message_type != "predict":
            raise ValueError(f"unsupported policy request: {message_type!r}")
        try:
            actions, elapsed_ms = runtime.predict(request)
            send_message(
                sys.stdout.buffer,
                {
                    "type": "prediction",
                    "request_id": int(request["request_id"]),
                    "actions": actions.tolist(),
                    "inference_ms": elapsed_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001
            send_message(
                sys.stdout.buffer,
                {
                    "type": "error",
                    "request_id": request.get("request_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )


if __name__ == "__main__":
    raise SystemExit(main())
