#!/usr/bin/env python3
"""GPU worker for the real G1 chunk-relative Diffusion policy.

This process intentionally contains no Unitree SDK imports.  The physical
robot process runs in the pinned Python 3.10 runtime, while this worker runs in
the LeRobot Python 3.12 environment.  They communicate only through anonymous
local pipes.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.upper_policy.worker_protocol import (
    receive_message,
    send_message,
)
from model.subtask_policy_training.action_representation import (
    ACTION_DIM,
    CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
    decode_action_chunk,
)
from model.subtask_policy_training.native_delta_policy import CAMERA_KEYS
from model.subtask_policy_training.scripts.evaluate_delta_chunk_reset import (
    NativeDiffusionEvaluator,
)


EXPECTED_MODEL_TYPE = "flip_table_native_diffusion_chunk_relative"
EXPECTED_MODEL_SHA256 = "1a5786d38b9aad995aaf030b6c38ca8e20d2b15471c644e61f7d1c3a3258fd67"
DEFAULT_MODEL_REPO_ID = (
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_1"
)
DEFAULT_MODEL_REVISION = "3291d3743a25ec8a69570fd7f57599b71fe69a63"
DEFAULT_TASK = "flip table"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected-model-sha256",
        default=EXPECTED_MODEL_SHA256,
        help="Pinned model.safetensors SHA-256 from the model registry.",
    )
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(
    path: Path,
    *,
    expected_model_sha256: str = EXPECTED_MODEL_SHA256,
) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected = {
        "type": EXPECTED_MODEL_TYPE,
        "observation_horizon": 2,
        "action_horizon": 16,
        "action_execution_steps": 8,
        "state_dim": 19,
        "action_dim": 16,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint contract mismatch: {mismatches}")
    train = json.loads((path / "train_config.json").read_text(encoding="utf-8"))
    if train.get("action_representation") != CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER:
        raise ValueError(
            "checkpoint is not chunk_relative_arm_absolute_gripper"
        )
    weights_hash = sha256(path / "model.safetensors")
    if weights_hash != expected_model_sha256:
        raise ValueError(
            f"unexpected model.safetensors SHA-256: {weights_hash}"
        )
    return {
        "model_type": config["type"],
        "weights_sha256": weights_hash,
        "observation_horizon": config["observation_horizon"],
        "action_horizon": config["action_horizon"],
        "action_execution_steps": config["action_execution_steps"],
    }


def decode_rgb(jpeg: bytes, role: str) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        shape = None if image is None else tuple(image.shape)
        raise ValueError(f"{role} JPEG must decode to 640x480 BGR, got {shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


def make_batch(request: dict[str, Any]) -> dict[str, torch.Tensor]:
    states = np.asarray(request.get("state_history"), dtype=np.float32)
    if states.shape != (2, 19) or not np.isfinite(states).all():
        raise ValueError(f"state_history must be finite [2,19], got {states.shape}")
    cameras = request.get("camera_history")
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERA_KEYS):
        raise ValueError(f"camera_history roles differ from {CAMERA_KEYS}")
    batch: dict[str, torch.Tensor] = {
        "observation.state": torch.from_numpy(states).unsqueeze(0)
    }
    for key in CAMERA_KEYS:
        values = cameras[key]
        if not isinstance(values, (list, tuple)) or len(values) != 2:
            raise ValueError(f"{key} history must contain two JPEG frames")
        batch[key] = torch.stack(
            [decode_rgb(bytes(value), key) for value in values]
        ).unsqueeze(0)
    return batch


def predict(
    evaluator: NativeDiffusionEvaluator,
    request: dict[str, Any],
) -> tuple[np.ndarray, float]:
    batch = make_batch(request)
    started = time.perf_counter()
    with torch.inference_mode():
        model_actions = evaluator.predict(batch)
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    decoded = decode_action_chunk(
        model_actions.detach().cpu(),
        batch["observation.state"],
        evaluator.action_representation,
    )
    result = decoded[0].numpy()
    if result.shape != (16, ACTION_DIM) or not np.isfinite(result).all():
        raise RuntimeError(f"policy returned invalid action chunk {result.shape}")
    return result, elapsed_ms


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for physical Diffusion inference")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    contract = validate_checkpoint(
        args.checkpoint,
        expected_model_sha256=args.expected_model_sha256,
    )
    contract.update(
        {
            "model_repo_id": args.model_repo_id,
            "model_revision": args.model_revision,
            "task": args.task,
        }
    )
    # LeRobot currently prints a load notice to stdout. Stdout is the binary
    # parent/child protocol, so third-party informational output belongs on
    # stderr.
    with redirect_stdout(sys.stderr):
        evaluator = NativeDiffusionEvaluator(args.checkpoint, args.device)
    send_message(
        sys.stdout.buffer,
        {
            "type": "ready",
            "contract": contract,
            "device": str(evaluator.device),
        },
    )
    while True:
        request = receive_message(sys.stdin.buffer)
        if not isinstance(request, dict):
            raise TypeError("policy request must be a mapping")
        if request.get("type") == "close":
            send_message(sys.stdout.buffer, {"type": "closed"})
            return 0
        if request.get("type") != "predict":
            raise ValueError(f"unsupported policy request: {request.get('type')!r}")
        try:
            actions, elapsed_ms = predict(evaluator, request)
            send_message(
                sys.stdout.buffer,
                {
                    "type": "prediction",
                    "request_id": int(request["request_id"]),
                    # Keep the wire format independent of the worker's NumPy
                    # 2.x ABI; the physical Python 3.10 environment currently
                    # uses NumPy 1.x.
                    "actions": actions.tolist(),
                    "inference_ms": elapsed_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001 - propagate safely to parent
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
