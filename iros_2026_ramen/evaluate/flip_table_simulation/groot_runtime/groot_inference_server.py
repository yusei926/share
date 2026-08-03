"""Serve GR00T N1.7 inference to the simulator over a local Unix socket."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
import random
import socket
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from policy.team_ramen_groot.n17_contract import (
        validate_finalized_furniture_checkpoint,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from model.subtask_policy_training.gr00t.n17_contract import (
        validate_finalized_furniture_checkpoint,
    )


LEROBOT_VERSION = "0.6.0"
STATE_DIM = 49
ACTION_DIM = 53
PACKED_DIM = 132
ACTION_HORIZON = 40
VALID_ACTION_DIM = 46
VIDEO_HORIZON = 2
EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
PINNED_BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
STATE_GROUPS = (
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
)
ACTION_GROUPS = STATE_GROUPS + ("base_height_command", "navigate_command")
ACTION_CONFIGS = {
    "left_wrist_eef_9d": ("relative", "eef", "xyzrot6d", "left_wrist_eef_9d"),
    "right_wrist_eef_9d": ("relative", "eef", "xyzrot6d", "right_wrist_eef_9d"),
    "left_hand": ("absolute", "noneef", "default", "left_hand"),
    "right_hand": ("absolute", "noneef", "default", "right_hand"),
    "left_arm": ("relative", "noneef", "default", "left_arm"),
    "right_arm": ("relative", "noneef", "default", "right_arm"),
    "waist": ("absolute", "noneef", "default", "waist"),
    "base_height_command": (
        "absolute",
        "noneef",
        "default",
        "base_height_command",
    ),
    "navigate_command": ("absolute", "noneef", "default", "navigate_command"),
}
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _feature_dim(config: dict[str, Any], group: str, key: str) -> int | None:
    shape = config.get(group, {}).get(key, {}).get("shape")
    if not isinstance(shape, list) or not shape:
        return None
    return int(shape[-1])


def validate_checkpoint(
    checkpoint: Path,
    *,
    furniture_release_validator: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete LeRobot GR00T checkpoint {checkpoint}: missing {missing}"
        )

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    policy_type = config.get("type")
    if policy_type not in {"groot", "furniture_groot"}:
        raise ValueError(
            "checkpoint policy type must be 'groot' or 'furniture_groot', "
            f"got {policy_type!r}"
        )
    if config.get("model_version", "n1.7") != "n1.7":
        raise ValueError(f"checkpoint model_version must be 'n1.7', got {config.get('model_version')!r}")
    if config.get("embodiment_tag") != EMBODIMENT_TAG:
        raise ValueError(
            f"checkpoint embodiment_tag must be {EMBODIMENT_TAG!r}, "
            f"got {config.get('embodiment_tag')!r}"
        )
    if config.get("use_relative_actions") is not True:
        raise ValueError("checkpoint must set use_relative_actions=true")
    if _feature_dim(config, "input_features", "observation.state") != STATE_DIM:
        raise ValueError("checkpoint observation.state must be 49-D")
    if _feature_dim(config, "output_features", "action") != ACTION_DIM:
        raise ValueError("checkpoint action must be 53-D")
    camera_keys = {
        key for key in config.get("input_features", {}) if key.startswith("observation.images.")
    }
    if camera_keys != set(CAMERA_KEYS):
        raise ValueError(
            "checkpoint policy cameras must be exactly head_left, left_wrist, and right_wrist; "
            f"got {sorted(camera_keys)}"
        )
    for camera_key in CAMERA_KEYS:
        shape = config["input_features"][camera_key].get("shape")
        if shape != [3, 480, 640]:
            raise ValueError(f"checkpoint {camera_key} must have shape [3,480,640], got {shape}")
    excluded = config.get("relative_exclude_joints")
    if not isinstance(excluded, list) or set(excluded) != {"hand", "waist", "base_height", "navigate"}:
        raise ValueError(
            "checkpoint relative_exclude_joints must preserve absolute hand, waist, "
            "base_height, and navigate groups"
        )
    chunk_size = int(config.get("chunk_size", 0))
    if chunk_size != ACTION_HORIZON:
        raise ValueError(
            f"checkpoint must retain the N1.7 H{ACTION_HORIZON}, got H{chunk_size}"
        )
    if int(config.get("max_state_dim", 0)) != PACKED_DIM:
        raise ValueError(f"checkpoint max_state_dim must be {PACKED_DIM}")
    if int(config.get("max_action_dim", 0)) != PACKED_DIM:
        raise ValueError(f"checkpoint max_action_dim must be {PACKED_DIM}")
    if policy_type == "furniture_groot":
        if int(config.get("valid_action_dim", 0)) != VALID_ACTION_DIM:
            raise ValueError(
                f"Furniture-GR00T valid_action_dim must be {VALID_ACTION_DIM}"
            )
        if config.get("base_model_revision") != PINNED_BASE_MODEL_REVISION:
            raise ValueError(
                "Furniture-GR00T base_model_revision must be the pinned N1.7 revision"
            )
        validator = (
            validate_finalized_furniture_checkpoint
            if furniture_release_validator is None
            else furniture_release_validator
        )
        validator(checkpoint)
    return config


def _stat_dim(entry: dict[str, Any]) -> int:
    for key in ("mean", "min", "q01", "max", "q99"):
        value = entry.get(key)
        if isinstance(value, list):
            return len(value[-1]) if value and isinstance(value[-1], list) else len(value)
    return 0


def _config_value(value: Any) -> str:
    value = getattr(value, "value", value)
    return str(value).lower().replace("_", "").replace("+", "")


def _relative_stat_shape(entry: dict[str, Any]) -> tuple[int, int]:
    for key in ("mean", "min", "q01", "max", "q99"):
        value = entry.get(key)
        if isinstance(value, list) and value and isinstance(value[0], list):
            return len(value), len(value[0])
    return 0, 0


def validate_processor_contract(
    preprocessor: Any,
    postprocessor: Any,
    *,
    required_horizon: int,
) -> None:
    from lerobot.policies.groot.processor_groot import (
        GrootN17ActionDecodeStep,
        GrootN17PackInputsStep,
    )

    pack = next(
        (step for step in preprocessor.steps if isinstance(step, GrootN17PackInputsStep)),
        None,
    )
    decode = next(
        (step for step in postprocessor.steps if isinstance(step, GrootN17ActionDecodeStep)),
        None,
    )
    if pack is None or decode is None:
        raise ValueError("serialized checkpoint must contain N1.7 pack and action-decode steps")
    if pack.embodiment_tag != EMBODIMENT_TAG:
        raise ValueError(f"processor embodiment mismatch: {pack.embodiment_tag!r}")
    if tuple(pack.video_modality_keys or ()) != tuple(key.rsplit(".", 1)[-1] for key in CAMERA_KEYS):
        raise ValueError(
            "processor camera order must be head_left, left_wrist, right_wrist; "
            f"got {pack.video_modality_keys!r}"
        )
    if not decode.use_relative_action:
        raise ValueError("processor must use native relative-action decoding")
    if decode.pack_step is not pack:
        raise ValueError("relative-action decoder is not connected to the observation pack step")
    if decode.env_action_dim != ACTION_DIM:
        raise ValueError(f"processor env_action_dim must be {ACTION_DIM}, got {decode.env_action_dim}")
    if pack.action_horizon < required_horizon or pack.valid_action_horizon < required_horizon:
        raise ValueError(
            f"processor action horizons must cover {required_horizon} steps; "
            f"got action={pack.action_horizon}, valid={pack.valid_action_horizon}"
        )
    if pack.action_horizon != ACTION_HORIZON or pack.valid_action_horizon != ACTION_HORIZON:
        raise ValueError("processor must retain the official H40 action contract")
    if pack.video_horizon != VIDEO_HORIZON:
        raise ValueError(
            f"processor video_horizon must be {VIDEO_HORIZON} for [-20,0], "
            f"got {pack.video_horizon}"
        )
    if pack.max_state_dim != PACKED_DIM or pack.max_action_dim != PACKED_DIM:
        raise ValueError("processor packed state/action dimensions must both be 132")

    modality = decode.modality_config or {}
    state_keys = tuple((modality.get("state") or {}).get("modality_keys") or ())
    action_keys = tuple((modality.get("action") or {}).get("modality_keys") or ())
    if state_keys != STATE_GROUPS:
        raise ValueError(f"processor state group order mismatch: {state_keys}")
    if action_keys != ACTION_GROUPS:
        raise ValueError(f"processor action group order mismatch: {action_keys}")
    action_configs = tuple((modality.get("action") or {}).get("action_configs") or ())
    if len(action_configs) != len(ACTION_GROUPS):
        raise ValueError(f"processor must define {len(ACTION_GROUPS)} action configs")
    for key, config in zip(ACTION_GROUPS, action_configs):
        if not isinstance(config, dict):
            raise ValueError(f"processor action config for {key} is not a mapping")
        actual = (
            _config_value(config.get("rep")),
            _config_value(config.get("type")),
            _config_value(config.get("format")),
            str(config.get("state_key") or key),
        )
        if actual != ACTION_CONFIGS[key]:
            raise ValueError(
                f"processor action config mismatch for {key}: "
                f"expected {ACTION_CONFIGS[key]}, got {actual}"
            )
    stats = decode.raw_stats or {}
    state_dim = sum(_stat_dim((stats.get("state") or {}).get(key, {})) for key in state_keys)
    action_dim = sum(_stat_dim((stats.get("action") or {}).get(key, {})) for key in action_keys)
    if state_dim != STATE_DIM or action_dim != ACTION_DIM:
        raise ValueError(
            f"processor statistics dimensions must be state={STATE_DIM}, action={ACTION_DIM}; "
            f"got state={state_dim}, action={action_dim}"
        )
    relative_stats = stats.get("relative_action") or {}
    for key in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_arm", "right_arm"):
        horizon, dim = _relative_stat_shape(relative_stats.get(key, {}))
        expected_dim = _stat_dim((stats.get("action") or {}).get(key, {}))
        if horizon < required_horizon or dim != expected_dim:
            raise ValueError(
                f"relative-action stats for {key} must cover [{required_horizon}, {expected_dim}], "
                f"got [{horizon}, {dim}]"
            )


def _recv_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_archive(connection: socket.socket) -> dict[str, np.ndarray] | None:
    header = _recv_exact(connection, 8)
    if header is None:
        return None
    size = struct.unpack("!Q", header)[0]
    if size <= 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid inference message size: {size}")
    payload = _recv_exact(connection, size)
    if payload is None:
        return None
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def send_archive(connection: socket.socket, **arrays: Any) -> None:
    output = io.BytesIO()
    np.savez(output, **arrays)
    payload = output.getvalue()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"inference response is too large: {len(payload)} bytes")
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def _text(array: np.ndarray, name: str) -> str:
    if array.size != 1:
        raise ValueError(f"{name} must contain one string")
    return str(array.reshape(-1)[0])


class GrootRuntime:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        n_action_steps: int,
        seed: int,
        furniture_release_validator: Callable[[Path], Any] | None = None,
    ) -> None:
        version = importlib.metadata.version("lerobot")
        if version != LEROBOT_VERSION:
            raise RuntimeError(f"expected lerobot=={LEROBOT_VERSION}, found {version}")
        checkpoint_config = validate_checkpoint(
            checkpoint,
            furniture_release_validator=furniture_release_validator,
        )

        import torch
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        policy_class: type[GrootPolicy]
        if checkpoint_config["type"] == "furniture_groot":
            from lerobot_policy_furniture_groot.modeling_furniture_groot import (
                FurnitureGrootPolicy,
            )
            from lerobot_policy_furniture_groot.processor_furniture_groot import (
                FurnitureGrootTemporalProgressStep,
            )

            del FurnitureGrootTemporalProgressStep
            policy_class = FurnitureGrootPolicy
        else:
            policy_class = GrootPolicy

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {device}")
        self._validate_seed(seed)
        self._seed_everything(seed)
        self.model = policy_class.from_pretrained(
            str(checkpoint),
            local_files_only=True,
            strict=True,
        )
        if n_action_steps < 1 or n_action_steps > int(self.model.config.chunk_size):
            raise ValueError(
                f"n_action_steps must be in [1, {self.model.config.chunk_size}], got {n_action_steps}"
            )
        self.model.config.n_action_steps = ACTION_HORIZON
        self.model.config.device = str(self.device)
        if self.model.config.action_decode_transform is not None:
            raise ValueError(
                "real G1 checkpoint must not apply a simulator-specific action_decode_transform"
            )
        self.model.to(self.device)
        self.model.eval()
        self.model.reset()

        self.preprocessor, self.postprocessor = make_groot_pre_post_processors_from_pretrained(
            self.model.config,
            str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "groot_n1_7_vlm_encode_v1": {"device": str(self.device)},
            },
        )
        validate_processor_contract(
            self.preprocessor,
            self.postprocessor,
            required_horizon=ACTION_HORIZON,
        )
        self.execution_steps = n_action_steps
        self.current_seed = seed

    @staticmethod
    def _validate_seed(seed: int) -> None:
        if seed < 0 or seed > np.iinfo(np.uint32).max:
            raise ValueError(f"inference seed must fit uint32, got {seed}")

    def _seed_everything(self, seed: int) -> None:
        self._validate_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed_everything(seed)
            self.current_seed = seed
        self.model.reset()

    def predict(self, request: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
        required = {"state", "head_left", "left_wrist", "right_wrist", "task"}
        missing = sorted(required - request.keys())
        if missing:
            raise ValueError(f"inference request is missing {missing}")
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
            raise ValueError(f"state must be finite shape ({STATE_DIM},), got {state.shape}")

        raw: dict[str, Any] = {
            "observation.state": self.torch.from_numpy(state),
            "task": _text(request["task"], "task"),
        }
        if raw["task"] != "flip table":
            raise ValueError("Furniture-GR00T task instruction must be exactly 'flip table'")
        for target_key, request_key in zip(CAMERA_KEYS, ("head_left", "left_wrist", "right_wrist")):
            image = np.asarray(request[request_key])
            if image.shape != (VIDEO_HORIZON, 480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"{request_key} must be unmodified uint8 [2,480,640,3] images, "
                    f"got shape={image.shape}, dtype={image.dtype}"
                )
            raw[target_key] = self.torch.from_numpy(
                np.ascontiguousarray(image.transpose(0, 3, 1, 2))
            )

        started = time.perf_counter()
        processed = self.preprocessor(raw)
        normalized_chunk = self.model.predict_action_chunk(processed)
        decoded_chunk = self.postprocessor(normalized_chunk)
        elapsed = time.perf_counter() - started
        decoded = decoded_chunk.detach().cpu().float().numpy()
        normalized = normalized_chunk.detach().cpu().float().numpy()
        if decoded.ndim != 3 or decoded.shape[0] != 1 or decoded.shape[-1] != ACTION_DIM:
            raise RuntimeError(f"decoded action chunk must have shape [1,T,{ACTION_DIM}], got {decoded.shape}")
        decoded = decoded[0, :ACTION_HORIZON]
        normalized = normalized[0, :ACTION_HORIZON, :ACTION_DIM]
        if decoded.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(decoded).all():
            raise RuntimeError("decoded action chunk is empty or non-finite")
        if normalized.shape != decoded.shape or not np.isfinite(normalized).all():
            raise RuntimeError(
                f"normalized action chunk must match decoded shape {decoded.shape}, got {normalized.shape}"
            )
        return decoded, normalized, elapsed


def serve(runtime: GrootRuntime, socket_path: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        server.listen(1)
        print(f"GR00T inference server ready: {socket_path}", flush=True)
        try:
            while True:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(600.0)
                    while True:
                        try:
                            request = receive_archive(connection)
                            if request is None:
                                break
                            kind = _text(request.get("kind", np.asarray("predict")), "kind")
                            if kind == "ping":
                                send_archive(connection, ok=np.asarray([1], dtype=np.uint8))
                            elif kind == "reset":
                                seed_array = request.get("seed")
                                seed = None
                                if seed_array is not None:
                                    seed_array = np.asarray(seed_array)
                                    if seed_array.size != 1:
                                        raise ValueError("reset seed must contain one integer")
                                    seed = int(seed_array.reshape(-1)[0])
                                runtime.reset(seed)
                                send_archive(
                                    connection,
                                    ok=np.asarray([1], dtype=np.uint8),
                                    seed=np.asarray([runtime.current_seed], dtype=np.uint64),
                                )
                            elif kind == "predict":
                                action, normalized_action, elapsed = runtime.predict(request)
                                send_archive(
                                    connection,
                                    ok=np.asarray([1], dtype=np.uint8),
                                    action=action,
                                    normalized_action=normalized_action,
                                    inference_seconds=np.asarray([elapsed], dtype=np.float64),
                                )
                            else:
                                raise ValueError(f"unsupported request kind: {kind!r}")
                        except (BrokenPipeError, ConnectionError, socket.timeout):
                            break
                        except Exception as exc:
                            traceback.print_exc()
                            try:
                                send_archive(
                                    connection,
                                    ok=np.asarray([0], dtype=np.uint8),
                                    error=np.asarray(str(exc)),
                                )
                            except (BrokenPipeError, ConnectionError):
                                break
        finally:
            socket_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=10,
        help="Physical execution steps between H40 replans",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Base uint32 seed; each simulator episode sends its explicit episode seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = GrootRuntime(
        args.checkpoint.resolve(),
        args.device,
        args.n_action_steps,
        args.seed,
    )
    serve(runtime, args.socket)


if __name__ == "__main__":
    main()
