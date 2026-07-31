"""Train the required two-frame, three-encoder ACT variant on a local LeRobot view."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import functional as image_functional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.native_delta_policy import (
    ACTION_DIM,
    ACTION_KEY,
    CAMERA_KEYS,
    CHUNK_SIZE,
    NativeACTConfig,
    NativeACTDeltaPolicy,
    OBS_STEPS,
    STATE_KEY,
    VIDEO_BACKEND,
    VIDEO_TIMESTAMP_TOLERANCE_S,
    configure_serial_video_decode,
    kl_divergence,
    normalize,
    normalizer_from_stats,
    save_native_act_checkpoint,
)
from model.subtask_policy_training.action_representation import (
    CHUNK_RELATIVE_ARM_ABSOLUTE_GRIPPER,
    encode_action_chunk,
    load_training_stats,
    semantics as action_semantics,
    validate_representation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-freq", type=int, default=10_000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--freeze-backbone-steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def delta_timestamps() -> dict[str, list[float]]:
    frame = 1.0 / 30.0
    observation = [-frame, 0.0]
    return {
        STATE_KEY: observation,
        **{key: observation for key in CAMERA_KEYS},
        ACTION_KEY: [index * frame for index in range(CHUNK_SIZE)],
    }


def make_dataset(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # The H100 runner uses a three-camera short-GOP cache and a bounded
    # decoder cache, so LeRobot's native parallel per-camera decode is safe.
    # Keep a bounded serial fallback for environments that explicitly request
    # a one-entry decoder cache.
    if os.environ.get("FLIP_TABLE_SERIAL_VIDEO_DECODE") == "1":
        configure_serial_video_decode()
    split = json.loads((root / "meta" / "team_ramen_episode_split.json").read_text(encoding="utf-8"))
    train_episodes = [int(index) for index in split["splits"]["train"]["episode_indices"]]
    if not train_episodes:
        raise ValueError("training split has no episodes")
    marker_path = root / "meta" / "team_ramen_training_view.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    repo_id = str(marker.get("source_repo_id", "")).strip()
    if not repo_id:
        raise ValueError(f"training view marker has no source_repo_id: {marker_path}")
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=train_episodes,
        delta_timestamps=delta_timestamps(),
        return_uint8=True,
        video_backend=VIDEO_BACKEND,
        tolerance_s=VIDEO_TIMESTAMP_TOLERANCE_S,
    )


def make_loader(dataset: Any, args: argparse.Namespace) -> DataLoader:
    options: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": True,
    }
    if args.num_workers > 0:
        # TorchCodec owns native decode buffers per worker. Recreating workers
        # at a bounded interval releases those buffers during long H100 runs.
        options.update(persistent_workers=False, prefetch_factor=1)
    return DataLoader(dataset, **options)


def shutdown_loader(iterator: Any | None) -> None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def serialize_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Return CLI provenance that is safe to include in checkpoint JSON."""
    return json.loads(json.dumps(vars(args), default=str))


def prepare_images(images: torch.Tensor, *, training: bool) -> torch.Tensor:
    """Resize and apply only the contractually allowed visual augmentations."""
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    else:
        images = images.float()
    batch, history, cameras, channels, height, width = images.shape
    if (history, cameras, channels) != (OBS_STEPS, len(CAMERA_KEYS), 3):
        raise ValueError(f"unexpected image batch shape {tuple(images.shape)}")
    flattened = images.reshape(batch * history * cameras, channels, height, width)
    flattened = F.interpolate(flattened, size=(240, 320), mode="bilinear", align_corners=False, antialias=True)
    if not training:
        return flattened.reshape(batch, history, cameras, channels, 240, 320)

    brightness = float(torch.empty((), device=images.device).uniform_(0.85, 1.15))
    contrast = float(torch.empty((), device=images.device).uniform_(0.85, 1.15))
    hue = float(torch.empty((), device=images.device).uniform_(-3.0 / 360.0, 3.0 / 360.0))
    flattened = image_functional.adjust_brightness(flattened, brightness)
    flattened = image_functional.adjust_contrast(flattened, contrast)
    flattened = image_functional.adjust_hue(flattened, hue)
    # Two pixels in the 640x480 source correspond to at most one output pixel.
    shift_x = int(torch.randint(-1, 2, (), device=images.device))
    shift_y = int(torch.randint(-1, 2, (), device=images.device))
    if shift_x or shift_y:
        flattened = image_functional.affine(flattened, angle=0.0, translate=[shift_x, shift_y], scale=1.0, shear=[0.0, 0.0])
    flattened = (flattened + torch.randn_like(flattened) * 0.01).clamp_(0.0, 1.0)
    return flattened.reshape(batch, history, cameras, channels, 240, 320)


def cosine_warmup_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    step_offset: int = 0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine scheduler at a fixed global-step offset.

    The final backbone stages join the optimizer after the freeze phase.  A
    fresh scheduler is needed so that the new parameter group receives the
    same global cosine schedule.  ``step_offset`` is intentionally captured
    as a function argument rather than the training-loop ``step`` variable:
    a closure over that mutable loop variable advances the schedule twice as
    fast and can reduce the learning rate to zero mid-run.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda scheduler_step: cosine_warmup_lambda(
            scheduler_step + step_offset,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
        ),
    )


def make_optimizer(model: NativeACTDeltaPolicy, args: argparse.Namespace) -> torch.optim.Optimizer:
    non_backbone = [parameter for name, parameter in model.named_parameters() if not name.startswith("camera_backbones") and parameter.requires_grad]
    return torch.optim.AdamW(non_backbone, lr=args.lr, weight_decay=args.weight_decay)


def add_final_backbone_stages(optimizer: torch.optim.Optimizer, model: NativeACTDeltaPolicy, args: argparse.Namespace) -> None:
    model.set_backbone_final_stages_trainable()
    parameters = [parameter for parameter in model.camera_backbones.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no camera backbone final-stage parameters became trainable")
    optimizer.add_param_group({"params": parameters, "lr": args.backbone_lr, "weight_decay": args.weight_decay})


def checkpoint(
    *, model: NativeACTDeltaPolicy, output_dir: Path, args: argparse.Namespace, stats: dict[str, Any], step: int
) -> Path:
    target = output_dir / "checkpoints" / str(step) / "pretrained_model"
    training_config = {
        "policy_type": "native_act_chunk_relative",
        "step": step,
        "seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "action_representation": args.action_representation,
        "action_contract": action_semantics(args.action_representation),
        "image_contract": "cam_0 head-left + cam_2 left D405 + cam_3 right D405; 2 frames; 320x240",
        "video_backend": VIDEO_BACKEND,
        "training_arguments": serialize_arguments(args),
    }
    save_native_act_checkpoint(model=model, output_dir=target, config=model.config, training_config=training_config, stats=stats)
    last = output_dir / "checkpoints" / "last"
    if last.exists() or last.is_symlink():
        last.unlink()
    last.symlink_to(Path(str(step)))
    return target


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.worker_restart_steps <= 0:
        raise ValueError("--steps, --batch-size, and --worker-restart-steps must be positive")
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
    model = NativeACTDeltaPolicy().to(device)
    model.set_backbone_trainable(False)
    optimizer = make_optimizer(model, args)
    scheduler = make_scheduler(optimizer, args)
    wandb = None
    if args.wandb_enable:
        import wandb as imported_wandb

        wandb = imported_wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))
    metrics_path = args.output_dir / "metrics.jsonl"
    start = time.perf_counter()
    iterator: Any | None = None
    unfrozen = False
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        try:
            for step in range(1, args.steps + 1):
                if iterator is None or (step - 1) % args.worker_restart_steps == 0:
                    shutdown_loader(iterator)
                    iterator = iter(make_loader(dataset, args))
                if step > args.freeze_backbone_steps and not unfrozen:
                    add_final_backbone_stages(optimizer, model, args)
                    scheduler = make_scheduler(optimizer, args, step_offset=step)
                    unfrozen = True
                try:
                    batch = next(iterator)
                except StopIteration:
                    shutdown_loader(iterator)
                    iterator = iter(make_loader(dataset, args))
                    batch = next(iterator)
                state = batch[STATE_KEY].to(device, non_blocking=True).float()
                actions = batch[ACTION_KEY].to(device, non_blocking=True).float()
                images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=2).to(device, non_blocking=True)
                action_is_pad = batch.get(
                    "action_is_pad", torch.zeros(actions.shape[:2], dtype=torch.bool)
                ).to(device)
                images = prepare_images(images, training=True)
                normalized_state = normalize(state, state_stats)
                model_actions = encode_action_chunk(actions, state, args.action_representation)
                normalized_actions = normalize(model_actions, action_stats)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction, mean, log_variance = model(images, normalized_state, normalized_actions)
                    valid = (~action_is_pad).unsqueeze(-1)
                    l1 = (prediction.sub(normalized_actions).abs() * valid).sum() / valid.sum().mul(
                        ACTION_DIM
                    ).clamp_min(1)
                    kl = kl_divergence(mean, log_variance)
                    loss = l1 + 10.0 * kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                if step % args.log_freq == 0 or step == 1 or step == args.steps:
                    elapsed = time.perf_counter() - start
                    record = {
                        "step": step,
                        "loss": float(loss.detach()),
                        "l1_loss": float(l1.detach()),
                        "kl_loss": float(kl.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "backbone_unfrozen": unfrozen,
                        "steps_per_sec": step / elapsed,
                    }
                    metrics_file.write(json.dumps(record) + "\n")
                    metrics_file.flush()
                    print(json.dumps(record), flush=True)
                    if wandb is not None:
                        wandb.log(record, step=step)
                if step % args.save_freq == 0 or step == args.steps:
                    checkpoint(model=model, output_dir=args.output_dir, args=args, stats=stats, step=step)
        finally:
            shutdown_loader(iterator)
    summary = {
        "steps": args.steps,
        "elapsed_s": time.perf_counter() - start,
        "wandb_url": getattr(wandb, "url", ""),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
