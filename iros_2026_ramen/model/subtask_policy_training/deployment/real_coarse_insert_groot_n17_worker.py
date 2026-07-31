#!/usr/bin/env python3
"""GPU worker for the pinned coarse-insert GR00T N1.7 checkpoint.

No Unitree/DDS imports are allowed in this process.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.metadata
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

from inference.desktop.upper_policy.coarse_insert_groot_contract import (  # noqa: E402
    CAMERA_KEYS,
    LEROBOT_VERSION,
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    MODEL_STATE_DIM,
    TASK_TEXT,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_SHA256,
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
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    parser.add_argument("--expected-model-sha256", default=MODEL_SHA256)
    return parser.parse_args()


def _decode_rgb(jpeg: bytes, role: str) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        shape = None if image is None else tuple(image.shape)
        raise ValueError(f"{role} JPEG must decode to 640x480 BGR, got {shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


class Runtime:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        seed: int,
        *,
        model_repo_id: str = MODEL_REPO_ID,
        model_revision: str = MODEL_REVISION,
        task: str = TASK_TEXT,
        expected_model_sha256: str = MODEL_SHA256,
    ) -> None:
        if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(f"coarse-insert requires lerobot=={LEROBOT_VERSION}")
        self.contract = validate_checkpoint_metadata(
            checkpoint,
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            task=task,
            expected_model_sha256=expected_model_sha256,
        )
        self.task = task
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable: {device}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with redirect_stdout(sys.stderr):
            self.model = GrootPolicy.from_pretrained(
                str(checkpoint),
                local_files_only=True,
                # This is the setting in the checkpoint author's deployment
                # adapter. Serialized processor validation and the pinned
                # complete-file SHA remain fail-closed.
                strict=False,
            )
            self.model.config.n_action_steps = MODEL_ACTION_HORIZON
            self.model.config.device = str(self.device)
            if self.model.config.action_decode_transform is not None:
                raise ValueError("coarse-insert cannot use a simulator decode transform")
            self.model.to(self.device)
            self.model.eval()
            self.model.reset()
            self.preprocessor, self.postprocessor = (
                make_groot_pre_post_processors_from_pretrained(
                    self.model.config,
                    str(checkpoint),
                    preprocessor_overrides={
                        "device_processor": {"device": str(self.device)},
                        "groot_n1_7_vlm_encode_v1": {"device": str(self.device)},
                    },
                )
            )

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (MODEL_STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError("coarse-insert state must be finite [49]")
        cameras = request.get("cameras")
        if not isinstance(cameras, dict) or set(cameras) != set(CAMERA_KEYS):
            raise ValueError(f"coarse-insert cameras must be exactly {CAMERA_KEYS}")
        if request.get("task") != self.task:
            raise ValueError(f"task must exactly match {self.task!r}")
        raw: dict[str, Any] = {
            "observation.state": torch.from_numpy(state),
            "task": self.task,
        }
        for key in CAMERA_KEYS:
            raw[key] = _decode_rgb(bytes(cameras[key]), key)
        started = time.perf_counter()
        with torch.inference_mode():
            processed = self.preprocessor(raw)
            normalized = self.model.predict_action_chunk(processed)
            decoded = self.postprocessor(normalized)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = decoded.detach().cpu().float().numpy()
        if result.ndim != 3 or result.shape[0] != 1 or result.shape[2] != MODEL_ACTION_DIM:
            raise RuntimeError(f"decoded action must be [1,T,53], got {result.shape}")
        result = result[0, :MODEL_ACTION_HORIZON]
        if result.shape != (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM):
            raise RuntimeError(f"decoded action must be [16,53], got {result.shape}")
        return result, elapsed_ms


def main() -> int:
    args = parse_args()
    runtime = Runtime(
        args.checkpoint.resolve(),
        args.device,
        args.seed,
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
        task=args.task,
        expected_model_sha256=args.expected_model_sha256,
    )
    send_message(
        sys.stdout.buffer,
        {"type": "ready", "contract": runtime.contract, "device": str(runtime.device)},
    )
    while True:
        request = receive_message(sys.stdin.buffer)
        if not isinstance(request, dict):
            raise TypeError("policy request must be a mapping")
        if request.get("type") == "close":
            send_message(sys.stdout.buffer, {"type": "closed"})
            return 0
        if request.get("type") != "predict":
            raise ValueError(f"unsupported request: {request.get('type')!r}")
        try:
            actions, elapsed_ms = runtime.predict(request)
            send_message(
                sys.stdout.buffer,
                {
                    "type": "prediction",
                    "request_id": request.get("request_id"),
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
