"""Train the specified Diffusion Policy with a real EMA checkpoint.

LeRobot supplies the official ConditionalUnet1D and camera encoders.  This loop
is deliberately local because the benchmark requires EMA=0.9999, which is not
maintained by the stock LeRobot 0.6 trainer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.native_delta_policy import (
    ACTION_DIM,
    ACTION_KEY,
    CAMERA_KEYS,
    CHUNK_SIZE,
    OBS_STEPS,
    STATE_KEY,
    VIDEO_BACKEND,
    normalize,
    normalizer_from_stats,
)
from model.subtask_policy_training.action_representation import (
    CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
    encode_action_chunk,
    load_training_stats,
    semantics as action_semantics,
    validate_representation,
)
from model.subtask_policy_training.scripts.train_native_act_delta import (
    cosine_warmup_lambda,
    make_dataset,
    make_loader,
    prepare_images,
    seed_everything,
    serialize_arguments,
    shutdown_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--worker-restart-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="iros2026-ramen-flip-table")
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--wandb-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--action-representation",
        default=CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
        help="Model-space action encoding; source training view remains absolute executable targets.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_policy(device: str):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    input_features = {
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(19,)),
        **{
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            for key in CAMERA_KEYS
        },
    }
    config = DiffusionConfig(
        input_features=input_features,
        output_features={ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        device=device,
        n_obs_steps=OBS_STEPS,
        horizon=CHUNK_SIZE,
        n_action_steps=8,
        down_dims=(256, 512, 1024),
        resize_shape=(240, 320),
        crop_ratio=1.0,
        crop_is_random=False,
        use_separate_rgb_encoder_per_camera=True,
        num_train_timesteps=100,
        num_inference_steps=10,
        # This policy uses z-score normalization.  Clipping a z-score into
        # [-1, 1] would distort valid arm and gripper targets at sampling time.
        clip_sample=False,
        do_mask_loss_for_padding=True,
        optimizer_lr=1e-4,
        optimizer_weight_decay=1e-4,
        scheduler_warmup_steps=10_000,
    )
    return DiffusionPolicy(config), config


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters(), strict=True):
        if ema_parameter.is_floating_point():
            ema_parameter.lerp_(parameter, 1.0 - decay)
        else:
            ema_parameter.copy_(parameter)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def checkpoint(
    *,
    ema_model: torch.nn.Module,
    config: Any,
    output_dir: Path,
    args: argparse.Namespace,
    stats: dict[str, Any],
    step: int,
) -> Path:
    target = output_dir / "checkpoints" / str(step) / "pretrained_model"
    target.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu() for name, tensor in ema_model.state_dict().items()}, target / "model.safetensors")
    config.save_pretrained(target / "_diffusion_config")
    primary_config = {
        "type": "flip_table_native_diffusion_chunk_relative",
        "observation_horizon": OBS_STEPS,
        "action_horizon": CHUNK_SIZE,
        "action_execution_steps": 8,
        "state_dim": 19,
        "action_dim": ACTION_DIM,
        "camera_encoder": "separate ResNet-18 per camera",
        "unet_channels": [256, 512, 1024],
        "train_diffusion_steps": 100,
        "evaluation_sampler": "DDIM 10 steps",
        "normalization": "zscore",
        "clip_sample": False,
        "do_mask_loss_for_padding": True,
        "ema_decay": args.ema_decay,
    }
    (target / "config.json").write_text(json.dumps(primary_config, indent=2), encoding="utf-8")
    training_config = {
        "policy_type": "native_diffusion_chunk_relative",
        "step": step,
        "seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "action_representation": args.action_representation,
        "action_contract": action_semantics(args.action_representation),
        "image_contract": "cam_0 head-left + cam_2 left D405 + cam_3 right D405; 2 frames; 320x240",
        "video_backend": VIDEO_BACKEND,
        "normalization_contract": "zscore; diffusion clip_sample=false",
        "diffusion_clip_sample": False,
        "ema_decay": args.ema_decay,
        "training_arguments": serialize_arguments(args),
    }
    (target / "train_config.json").write_text(json.dumps(training_config, indent=2), encoding="utf-8")
    (target / "normalization.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (target / name).write_text(json.dumps({"steps": []}, indent=2), encoding="utf-8")
    last = output_dir / "checkpoints" / "last"
    if last.exists() or last.is_symlink():
        last.unlink()
    last.symlink_to(Path(str(step)))
    return target


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.worker_restart_steps <= 0:
        raise ValueError("--steps, --batch-size, and --worker-restart-steps must be positive")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")
    args.action_representation = validate_representation(args.action_representation)
    seed_everything(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset(args.dataset_root)
    stats = load_training_stats(args.dataset_root, args.action_representation)
    state_stats = normalizer_from_stats(stats, STATE_KEY, device=device)
    action_stats = normalizer_from_stats(stats, ACTION_KEY, device=device)
    policy, config = build_policy(str(device))
    policy = policy.to(device)
    ema_policy = copy.deepcopy(policy).to(device).eval()
    for parameter in ema_policy.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_lambda(step, warmup_steps=args.warmup_steps, total_steps=args.steps),
    )
    wandb = None
    if args.wandb_enable:
        import wandb as imported_wandb

        wandb = imported_wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    metrics_path = args.output_dir / "metrics.jsonl"
    start = time.perf_counter()
    iterator: Any | None = None
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        try:
            for step in range(1, args.steps + 1):
                if iterator is None or (step - 1) % args.worker_restart_steps == 0:
                    shutdown_loader(iterator)
                    iterator = iter(make_loader(dataset, args))
                try:
                    batch = next(iterator)
                except StopIteration:
                    shutdown_loader(iterator)
                    iterator = iter(make_loader(dataset, args))
                    batch = next(iterator)
                state = batch[STATE_KEY].to(device, non_blocking=True).float()
                actions = batch[ACTION_KEY].to(device, non_blocking=True).float()
                images = {key: batch[key].to(device, non_blocking=True) for key in CAMERA_KEYS}
                prepared_images = prepare_images(
                    torch.stack([images[key] for key in CAMERA_KEYS], dim=2), training=True
                )
                policy_batch = {
                    STATE_KEY: normalize(state, state_stats),
                    ACTION_KEY: normalize(
                        encode_action_chunk(actions, state, args.action_representation), action_stats
                    ),
                    "action_is_pad": batch["action_is_pad"].to(device, non_blocking=True),
                    **{key: prepared_images[:, :, index] for index, key in enumerate(CAMERA_KEYS)},
                }
                policy.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    loss, _ = policy(policy_batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                # Warm up the moving average so a 1k-update smoke checkpoint is a
                # meaningful EMA model, then hold the requested 0.9999 decay.
                effective_ema_decay = min(args.ema_decay, 1.0 - 1.0 / float(step + 1))
                update_ema(ema_policy, policy, effective_ema_decay)
                if step % args.log_freq == 0 or step == 1 or step == args.steps:
                    elapsed = time.perf_counter() - start
                    record = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "ema_decay": args.ema_decay,
                        "ema_effective_decay": effective_ema_decay,
                        "steps_per_sec": step / elapsed,
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    print(json.dumps(record), flush=True)
                    if wandb is not None:
                        wandb.log(record, step=step)
                if step % args.save_freq == 0 or step == args.steps:
                    checkpoint(
                        ema_model=ema_policy,
                        config=config,
                        output_dir=args.output_dir,
                        args=args,
                        stats=stats,
                        step=step,
                    )
        finally:
            shutdown_loader(iterator)
    summary = {"steps": args.steps, "elapsed_s": time.perf_counter() - start, "wandb_url": getattr(wandb, "url", "")}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
