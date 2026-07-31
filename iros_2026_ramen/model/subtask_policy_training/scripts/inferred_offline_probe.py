"""Load one inferred HF checkpoint and run one synthetic, non-robot inference.

This process intentionally has no Unitree SDK, DDS, camera, or actuator import.
It validates only that the pinned artifacts can be deserialized by a known
local implementation and produce a finite tensor with the inferred shape.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = (
    REPO_ROOT
    / "model/subtask_policy_training/lerobot_policy_furniture_groot"
)
for path in (REPO_ROOT, PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.desktop.model_evaluation.inferred_artifacts import (  # noqa: E402
    checkpoint_path,
    load_contract,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="perform the demonstrated manipulation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation = validate(args.local_dir)
    contract = load_contract(args.local_dir)
    if not contract.weight_load_supported:
        raise RuntimeError("contract does not support offline weight loading")
    checkpoint = checkpoint_path(contract, args.local_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    started = time.perf_counter()
    policy, preprocessor, postprocessor, config = _load(
        checkpoint, contract.config_type, args.device
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    batch = _synthetic_batch(
        config=config,
        contract=contract,
        task=args.task,
        device=args.device,
    )
    if preprocessor is not None:
        processed = preprocessor(batch)
    else:
        processed = batch
    policy.reset() if hasattr(policy, "reset") else None
    started = time.perf_counter()
    with torch.inference_mode():
        if hasattr(policy, "predict"):
            output = policy.predict(processed)
        else:
            output = policy.predict_action_chunk(processed)
        if postprocessor is not None:
            output = postprocessor(output)
    inference_ms = (time.perf_counter() - started) * 1000.0
    tensor = _as_tensor(output).detach().cpu().float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"prediction must be [B,H,D], got {tuple(tensor.shape)}")
    if contract.action_dim is not None and tensor.shape[-1] != contract.action_dim:
        raise ValueError(
            f"prediction action dim changed: {tensor.shape[-1]} != {contract.action_dim}"
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("prediction contains NaN or Inf")
    forbidden = sorted(
        name
        for name in sys.modules
        if name.startswith(("unitree_sdk2py", "cyclonedds"))
    )
    if forbidden:
        raise RuntimeError(f"offline probe imported physical transports: {forbidden}")
    report = {
        "schema_version": "team_ramen_inferred_offline_probe/v1",
        "repo_id": contract.repo_id,
        "revision": contract.revision,
        "config_type": contract.config_type,
        "loader_kind": contract.loader_kind,
        "checkpoint": str(checkpoint),
        "input_state_shape": [1, int(contract.state_dim or 0)],
        "camera_keys": list(contract.camera_keys),
        "prediction_shape": list(tensor.shape),
        "prediction_min": float(tensor.min()),
        "prediction_max": float(tensor.max()),
        "prediction_mean": float(tensor.mean()),
        "load_ms": load_ms,
        "inference_ms": inference_ms,
        "ignored_training_only_config_fields": list(
            getattr(config, "_offline_ignored_training_fields", ())
        ),
        "offline_runtime_overrides": dict(
            getattr(config, "_offline_runtime_overrides", {})
        ),
        "artifact_validation": validation["tamper_check"],
        "physical_mapping_verified": False,
        "actuation_allowed": False,
        "robot_command_sent": False,
        "dds_initialized": False,
        "physical_transport_imported": False,
    }
    print("__IROS_RAMEN_OFFLINE_REPORT__" + json.dumps(report, ensure_ascii=False))
    return 0


def _load(
    checkpoint: Path, config_type: str | None, device: str
) -> tuple[Any, Any, Any, Any]:
    if config_type in {
        "flip_table_native_act_chunk_relative",
        "flip_table_native_diffusion_chunk_relative",
    }:
        from model.subtask_policy_training.scripts.evaluate_delta_chunk_reset import (
            load_policy,
        )

        policy, preprocessor, postprocessor = load_policy(checkpoint, device)
        return policy, preprocessor, postprocessor, policy.config

    if config_type == "furniture_groot":
        import lerobot_policy_furniture_groot  # noqa: F401
        from lerobot.configs import PreTrainedConfig
        from lerobot_policy_furniture_groot.modeling_furniture_groot import (
            FurnitureGrootPolicy,
        )
        from lerobot.policies.groot.processor_groot import (
            make_groot_pre_post_processors_from_pretrained,
        )

        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.device = device
        runtime_overrides: dict[str, Any] = {}
        base_model_path = str(getattr(config, "base_model_path", ""))
        if base_model_path.startswith("/") and not Path(base_model_path).exists():
            # The published checkpoint retained the trainer's /dev/shm overlay
            # path. The repository contract pins the canonical upstream model
            # and exact revision, so restore that portable identifier in-memory.
            if (
                getattr(config, "base_model_revision", None)
                != "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
            ):
                raise ValueError("Furniture-GR00T base revision is not trusted")
            config.base_model_path = "nvidia/GR00T-N1.7-3B"
            runtime_overrides["base_model_path"] = "nvidia/GR00T-N1.7-3B"
        setattr(config, "_offline_runtime_overrides", runtime_overrides)
        policy = FurnitureGrootPolicy.from_pretrained(
            str(checkpoint), config=config, strict=True
        ).eval()
        preprocessor, postprocessor = make_groot_pre_post_processors_from_pretrained(
            config,
            str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "groot_n1_7_vlm_encode_v1": {"device": device},
            },
        )
        return policy, preprocessor, postprocessor, config

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = _load_standard_config(checkpoint, config_type)
    config.device = device
    runtime_overrides: dict[str, Any] = {}
    if bool(getattr(config, "compile_model", False)):
        # Compilation changes latency, not checkpoint semantics. max-autotune
        # can otherwise take minutes during a one-shot compatibility probe.
        config.compile_model = False
        runtime_overrides["compile_model"] = False
    setattr(config, "_offline_runtime_overrides", runtime_overrides)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(
        str(checkpoint), config=config, strict=True
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint)
    )
    return policy, preprocessor, postprocessor, config


_PI05_TRAINING_ONLY_FIELDS = frozenset(
    {
        "use_per_timestamp_action_stats",
        "per_timestamp_stats_path",
        "use_correlated_noise",
        "correlated_noise_stats_path",
        "correlated_noise_beta",
        "action_dim_weights",
        "smoothness_lambda",
        "smoothness_exclude_dims",
        "action_loss_exclude_dims",
        "use_min_snr_weighting",
        "min_snr_gamma",
        "use_dafd",
        "dafd_gripper_dim",
        "dafd_gripper_threshold",
        "dafd_gripper_open_value",
        "dafd_gripper_close_value",
        "dafd_gripper_weight",
        "dafd_sign_dim",
        "dafd_sign_weight",
        "normalization_clip",
        "use_aux_base_velocity_head",
        "aux_base_velocity_weight",
        "aux_base_velocity_dims",
        "use_ema",
        "ema_decay",
    }
)


def _load_standard_config(checkpoint: Path, config_type: str | None) -> Any:
    """Project known training-only Pi0.5 fields without mutating sealed files."""
    from lerobot.configs import PreTrainedConfig

    try:
        return PreTrainedConfig.from_pretrained(str(checkpoint))
    except Exception:
        if config_type != "pi05":
            raise
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    raw = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(PI05Config)} | {"type"}
    unknown = set(raw) - allowed
    unsupported = unknown - _PI05_TRAINING_ONLY_FIELDS
    if unsupported:
        raise ValueError(
            "Pi0.5 config contains unknown inference-affecting fields: "
            f"{sorted(unsupported)}"
        )
    # These options only affected the training loss/statistics construction.
    # Reject any setting that implies a runtime transform not represented by
    # the standard processor artifacts.
    active_runtime = {
        key: raw.get(key)
        for key in unknown
        if key
        in {
            "use_per_timestamp_action_stats",
            "use_correlated_noise",
            "use_aux_base_velocity_head",
        }
        and bool(raw.get(key))
    }
    if active_runtime:
        raise ValueError(
            "Pi0.5 training extensions require a dedicated runtime adapter: "
            f"{active_runtime}"
        )
    sanitized = {key: value for key, value in raw.items() if key in allowed}
    with tempfile.TemporaryDirectory(prefix="iros-pi05-config-") as temporary:
        Path(temporary, "config.json").write_text(
            json.dumps(sanitized), encoding="utf-8"
        )
        config = PreTrainedConfig.from_pretrained(temporary)
    setattr(config, "_offline_ignored_training_fields", tuple(sorted(unknown)))
    return config


def _synthetic_batch(
    *,
    config: Any,
    contract: Any,
    task: str,
    device: str,
) -> dict[str, Any]:
    if contract.config_type == "furniture_groot":
        state = torch.zeros((int(contract.state_dim),), dtype=torch.float32)
        # The first two 9-D groups are XYZ + the first two rows of a
        # rotation matrix. All-zero ROT6D is singular and not a valid robot
        # observation, so use identity orientation for the synthetic probe.
        state[3:9] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        state[12:18] = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)
        batch: dict[str, Any] = {
            "observation.state": state,
            "task": "flip table",
        }
        for key in contract.camera_keys:
            batch[key] = torch.full(
                (2, 3, 480, 640),
                127,
                dtype=torch.uint8,
            )
        return batch
    if contract.config_type in {
        "flip_table_native_act_chunk_relative",
        "flip_table_native_diffusion_chunk_relative",
    }:
        observation_steps = int(contract.observation_horizon or 1)
        batch: dict[str, Any] = {
            "observation.state": torch.zeros(
                (1, observation_steps, int(contract.state_dim)),
                dtype=torch.float32,
                device=device,
            )
        }
        for key in (
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ):
            batch[key] = torch.full(
                (1, observation_steps, 3, 480, 640),
                0.5,
                dtype=torch.float32,
                device=device,
            )
        return batch

    features = getattr(config, "input_features", None)
    if not isinstance(features, Mapping):
        raise ValueError("LeRobot config has no input_features mapping")
    batch = {"task": [task]}
    for key, feature in features.items():
        shape = tuple(int(value) for value in feature.shape)
        if key == "observation.state":
            batch[key] = torch.zeros((1, *shape), dtype=torch.float32)
        elif key.startswith("observation.images."):
            # Non-uniform pixels ensure image preprocessing is exercised.
            values = torch.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)
            batch[key] = values.unsqueeze(0)
    return batch


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for key in ("action", "actions"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key]
    raise TypeError(f"prediction is not a Tensor: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
