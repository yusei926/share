#!/usr/bin/env python3
"""GPU-only LeRobot ACT worker for the pinned joint16 pick-leg policy.

This process intentionally imports no Unitree SDK, CycloneDDS, camera transport,
or actuator code.  The physical process supplies four JPEGs and state16 over
anonymous local pipes; this worker returns one finite absolute action chunk.
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
from safetensors.torch import load_file
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference.desktop.upper_policy.act_pick_leg_contract import (  # noqa: E402
    ACTION_MAX,
    ACTION_MIN,
    CAMERA_KEYS,
    MODEL_ACTION_DIM,
    MODEL_ACTION_HORIZON,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_STATE_DIM,
    TASK_TEXT,
)
from inference.desktop.upper_policy.worker_protocol import (  # noqa: E402
    receive_message,
    send_message,
)


EXPECTED_MODEL_SHA256 = "581bde6e608d196f7018af1acf01ea666a2a07fb472f8c898c5f2c116ed45bce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-model-sha256", default=EXPECTED_MODEL_SHA256)
    parser.add_argument("--model-repo-id", default=MODEL_REPO_ID)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--task", default=TASK_TEXT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, expected_hash: str) -> dict[str, Any]:
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    expected_features = {
        **{
            key: {"type": "VISUAL", "shape": [3, 480, 640]}
            for key in CAMERA_KEYS
        },
        "observation.state": {"type": "STATE", "shape": [MODEL_STATE_DIM]},
    }
    expected = {
        "type": "act",
        "n_obs_steps": 1,
        "chunk_size": MODEL_ACTION_HORIZON,
        "n_action_steps": MODEL_ACTION_HORIZON,
        "input_features": expected_features,
        "output_features": {
            "action": {"type": "ACTION", "shape": [MODEL_ACTION_DIM]}
        },
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"ACT checkpoint contract mismatch: {mismatches}")

    pre = json.loads((path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    post = json.loads((path / "policy_postprocessor.json").read_text(encoding="utf-8"))
    pre_names = [step.get("registry_name") for step in pre.get("steps", [])]
    post_names = [step.get("registry_name") for step in post.get("steps", [])]
    if pre_names != [
        "rename_observations_processor",
        "to_batch_processor",
        "device_processor",
        "normalizer_processor",
    ]:
        raise ValueError(f"unexpected ACT preprocessor pipeline: {pre_names}")
    if post_names != ["unnormalizer_processor", "device_processor"]:
        raise ValueError(f"unexpected ACT postprocessor pipeline: {post_names}")

    stats_path = path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    stats = load_file(str(stats_path), device="cpu")
    minimum = stats["action.min"].detach().cpu().numpy().astype(np.float64)
    maximum = stats["action.max"].detach().cpu().numpy().astype(np.float64)
    if not np.allclose(minimum, ACTION_MIN, atol=1.0e-5, rtol=0.0):
        raise ValueError("serialized action minimum differs from physical contract")
    if not np.allclose(maximum, ACTION_MAX, atol=1.0e-5, rtol=0.0):
        raise ValueError("serialized action maximum differs from physical contract")

    model_hash = sha256(path / "model.safetensors")
    if model_hash != expected_hash:
        raise ValueError(
            f"unexpected model.safetensors SHA-256: {model_hash} != {expected_hash}"
        )
    return {
        "model_type": "act",
        "weights_sha256": model_hash,
        "state_dim": MODEL_STATE_DIM,
        "decoded_action_dim": MODEL_ACTION_DIM,
        "executable_action_dim": MODEL_ACTION_DIM,
        "action_horizon": MODEL_ACTION_HORIZON,
        "lower_body_command_dimensions": 0,
        "camera_keys": list(CAMERA_KEYS),
        "dex1_open_value": 4.5,
    }


def decode_rgb(jpeg: bytes, key: str) -> torch.Tensor:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        shape = None if image is None else tuple(image.shape)
        raise ValueError(f"{key} JPEG must decode to 640x480 BGR, got {shape}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0


def make_batch(request: dict[str, Any]) -> dict[str, Any]:
    state = np.asarray(request.get("state"), dtype=np.float32)
    if state.shape != (MODEL_STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"state must be finite [{MODEL_STATE_DIM}], got {state.shape}")
    cameras = request.get("cameras")
    if not isinstance(cameras, dict) or set(cameras) != set(CAMERA_KEYS):
        raise ValueError(f"camera keys must be exactly {CAMERA_KEYS}")
    batch: dict[str, Any] = {
        "observation.state": torch.from_numpy(state),
        "task": request.get("task", TASK_TEXT),
    }
    for key in CAMERA_KEYS:
        batch[key] = decode_rgb(bytes(cameras[key]), key)
    return batch


def predict(policy: Any, preprocessor: Any, postprocessor: Any, request: dict[str, Any]) -> tuple[np.ndarray, float]:
    batch = make_batch(request)
    # A request represents a fresh observation and must produce a complete
    # chunk.  Never consume a stale internal action queue from a prior request.
    policy.reset()
    if hasattr(preprocessor, "reset"):
        preprocessor.reset()
    if hasattr(postprocessor, "reset"):
        postprocessor.reset()
    processed = preprocessor(batch)
    started = time.perf_counter()
    with torch.inference_mode():
        output = policy.predict_action_chunk(processed)
        output = postprocessor(output)
    if str(policy.config.device).startswith("cuda"):
        torch.cuda.synchronize(torch.device(policy.config.device))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(output, dict):
        output = output.get("action", output.get("actions"))
    result = torch.as_tensor(output).detach().cpu().float().numpy()
    if result.ndim == 3 and result.shape[0] == 1:
        result = result[0]
    expected = (MODEL_ACTION_HORIZON, MODEL_ACTION_DIM)
    if result.shape != expected or not np.isfinite(result).all():
        raise RuntimeError(f"ACT policy returned invalid action chunk {result.shape}")
    return result, elapsed_ms


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for physical ACT inference")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    contract = validate_checkpoint(args.checkpoint, args.expected_model_sha256)
    contract.update(
        {
            "model_repo_id": args.model_repo_id,
            "model_revision": args.model_revision,
            "task": args.task,
        }
    )
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    with redirect_stdout(sys.stderr):
        config = PreTrainedConfig.from_pretrained(str(args.checkpoint))
        config.device = args.device
        config.pretrained_path = str(args.checkpoint)
        policy = get_policy_class(config.type).from_pretrained(
            str(args.checkpoint), config=config, strict=True
        ).eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(args.checkpoint),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
    send_message(
        sys.stdout.buffer,
        {"type": "ready", "contract": contract, "device": args.device},
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
            actions, elapsed_ms = predict(policy, preprocessor, postprocessor, request)
            send_message(
                sys.stdout.buffer,
                {
                    "type": "prediction",
                    "request_id": int(request["request_id"]),
                    "actions": actions.tolist(),
                    "inference_ms": elapsed_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001 - safely report to parent
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
