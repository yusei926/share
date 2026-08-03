#!/usr/bin/env python3
"""GPU worker for the immutable, unselected flip-table GR00T-v2 20k candidate.

This worker is deliberately separate from the finalized Furniture-GR00T
worker.  It permits an offline/physical evaluation candidate to load without
claiming that simulator selection or release acceptance has passed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Sequence

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPO_ROOT / "model/subtask_policy_training/lerobot_policy_furniture_groot"
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
    validate_legacy_v2_candidate_checkpoint,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)
from model.subtask_policy_training.gr00t.n17_contract import (  # noqa: E402
    BASE_MODEL_REPO_ID,
)


MODEL_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints"
MODEL_REVISION = "1a408d87eda8d01f9b79113f1aed97a5d0811bff"
MODEL_SHA256 = "0a8d3c6756b174df54c4ed8fce24455ec934648c1ca120fec5847cd87c88156f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    return parser.parse_args()


def _decode_rgb_history(jpegs: Sequence[bytes], role: str) -> np.ndarray:
    if len(jpegs) != VIDEO_HORIZON:
        raise ValueError(f"{role} requires exactly {VIDEO_HORIZON} JPEG frames")
    frames: list[np.ndarray] = []
    for index, payload in enumerate(jpegs):
        image = cv2.imdecode(np.frombuffer(bytes(payload), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (480, 640, 3):
            shape = None if image is None else tuple(image.shape)
            raise ValueError(f"{role}[{index}] must decode to 640x480 BGR, got {shape}")
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return np.ascontiguousarray(np.stack(frames, axis=0))


def _portable_runtime_overlay(checkpoint: Path) -> tempfile.TemporaryDirectory[str]:
    """Make only the historical trainer-local base path portable.

    All learned weights and serialized processors remain symlinks to the
    sealed checkpoint.  The original artifact is never modified.
    """

    temporary = tempfile.TemporaryDirectory(prefix="iros-groot-v2-candidate-")
    root = Path(temporary.name)
    for source in checkpoint.iterdir():
        if source.is_file() and source.name != "config.json":
            (root / source.name).symlink_to(source)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    config["base_model_path"] = BASE_MODEL_REPO_ID
    (root / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return temporary


class Runtime:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        seed: int,
        *,
        model_repo_id: str,
        model_revision: str,
        task: str,
        expected_model_sha256: str,
    ) -> None:
        expected_identity = (MODEL_REPO_ID, MODEL_REVISION, TASK_TEXT, MODEL_SHA256)
        actual_identity = (
            model_repo_id,
            model_revision,
            task,
            expected_model_sha256,
        )
        if actual_identity != expected_identity:
            raise ValueError(
                "candidate identity differs from the reviewed 20k artifact: "
                f"{actual_identity}"
            )
        if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
            raise RuntimeError(
                f"physical Furniture-GR00T inference requires lerobot=={LEROBOT_VERSION}"
            )
        self.contract = validate_legacy_v2_candidate_checkpoint(
            checkpoint,
            expected_model_sha256=expected_model_sha256,
        )
        self.contract.update(
            {
                "model_repo_id": model_repo_id,
                "model_revision": model_revision,
                "task": task,
            }
        )
        self._temporary = _portable_runtime_overlay(checkpoint)
        runtime_root = Path(self._temporary.name)
        self.runtime = GrootRuntime(
            checkpoint=runtime_root,
            device=device,
            n_action_steps=MODEL_ACTION_HORIZON,
            seed=seed,
            # The immutable source artifact was validated immediately above.
            # The runtime overlay changes only base_model_path portability.
            furniture_release_validator=lambda _path: self.contract,
        )

    def reset(self) -> None:
        self.runtime.reset()

    def predict(self, request: dict[str, Any]) -> tuple[np.ndarray, float]:
        state = np.asarray(request.get("state"), dtype=np.float32)
        if state.shape != (49,) or not np.isfinite(state).all():
            raise ValueError("state must be finite [49]")
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
            payload[key.rsplit(".", 1)[-1]] = _decode_rgb_history(camera_history[key], key)
        decoded, _normalized, elapsed = self.runtime.predict(payload)
        return extract_executable_action(decoded), elapsed * 1000.0


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
        {"type": "ready", "contract": runtime.contract, "device": str(runtime.runtime.device)},
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
