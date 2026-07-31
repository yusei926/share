#!/usr/bin/env python3
"""Add optimizer-step gradient accumulation to the pinned LeRobot trainer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

LEROBOT_VERSION = "0.6.0"
ORIGINAL_SHA256 = "95f6d05d21205c03abbfbf4225ed1cb26b7af4826e3e8a84def3eb10ddd988dc"
PATCH_MARKER = "# TEAM_RAMEN_GRADIENT_ACCUMULATION_V1"

REPLACEMENTS = (
    (
        "import logging\nimport sys\n",
        "import logging\nimport os\nimport sys\n",
    ),
    (
        "\n\ndef update_policy(\n",
        """

# TEAM_RAMEN_GRADIENT_ACCUMULATION_V1
def _gradient_accumulation_steps() -> int:
    try:
        value = int(os.environ.get("LEROBOT_GRADIENT_ACCUMULATION_STEPS", "1"))
    except ValueError as exc:
        raise ValueError("LEROBOT_GRADIENT_ACCUMULATION_STEPS must be an integer") from exc
    if value < 1:
        raise ValueError("LEROBOT_GRADIENT_ACCUMULATION_STEPS must be positive")
    return value


def update_policy(
""",
    ),
    (
        """    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )
""",
        """    # Clip only after the final micro-batch has been accumulated.
    if accelerator.sync_gradients:
        if grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), float("inf"), error_if_nonfinite=False
            )
    else:
        grad_norm = torch.zeros((), device=loss.device)
""",
    ),
    (
        """    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
""",
        """    # One scheduler/policy update corresponds to one optimizer update, not a micro-batch.
    if lr_scheduler is not None and accelerator.sync_gradients:
        lr_scheduler.step()

    if accelerator.sync_gradients and has_method(
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
    ):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()
""",
    ),
    (
        """    cfg.validate()

    # Create Accelerator if not provided
""",
        """    cfg.validate()
    gradient_accumulation_steps = _gradient_accumulation_steps()

    # Create Accelerator if not provided
""",
    ),
    (
        """        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)
""",
        """        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )
    elif int(accelerator.gradient_accumulation_steps) != gradient_accumulation_steps:
        raise ValueError(
            "Accelerator gradient_accumulation_steps does not match "
            "LEROBOT_GRADIENT_ACCUMULATION_STEPS"
        )

    init_logging(accelerator=accelerator)
""",
    ),
    (
        """        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
""",
        """        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * gradient_accumulation_steps
        logging.info(
            "Effective batch size: "
            f"{cfg.batch_size} x {num_processes} x {gradient_accumulation_steps} = {effective_bs}"
        )
""",
    ),
    (
        """            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            ckpt_batch_size = saved_batch_size or cfg.batch_size
""",
        """            ckpt_num_processes = saved_num_processes or accelerator.num_processes
            effective_per_process_batch = cfg.batch_size * gradient_accumulation_steps
            ckpt_batch_size = saved_batch_size or effective_per_process_batch
""",
    ),
    (
        """            if is_main_process and saved_batch_size not in (None, cfg.batch_size):
                logging.warning(
                    f"Resuming with batch_size={cfg.batch_size} but the checkpoint was written with "
                    f"batch_size={saved_batch_size}. The data order resumes at the right epoch/offset, "
                    "but per-rank sample-exactness requires the same batch size."
                )
""",
        """            if is_main_process and saved_batch_size not in (
                None,
                effective_per_process_batch,
            ):
                logging.warning(
                    f"Resuming with effective batch_size={effective_per_process_batch} but the "
                    f"checkpoint was written with batch_size={saved_batch_size}. The data order "
                    "resumes at the right epoch/offset, but per-rank sample-exactness requires "
                    "the same effective batch size."
                )
""",
    ),
    (
        """    # Keep global batch size for logging; MetricsTracker handles world size internally.
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        cfg.batch_size,
""",
        """    # Keep global optimizer batch size for logging and sample accounting.
    effective_batch_size = (
        cfg.batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    train_tracker = MetricsTracker(
        cfg.batch_size * gradient_accumulation_steps,
""",
    ),
    (
        """                step_time = train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                if step_time > 0:
                    train_tracker.samples_per_s = effective_batch_size / step_time
""",
        """                micro_step_time = (
                    train_tracker.update_s.avg + train_tracker.dataloading_s.avg
                )
                optimizer_step_time = micro_step_time * gradient_accumulation_steps
                if optimizer_step_time > 0:
                    train_tracker.samples_per_s = (
                        effective_batch_size / optimizer_step_time
                    )
""",
    ),
    (
        """    for _ in range(step, cfg.steps):
""",
        """    while step < cfg.steps:
""",
    ),
    (
        """        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weighter=sample_weighter,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
""",
        """        with accelerator.accumulate(policy):
            train_tracker, output_dict = update_policy(
                train_tracker,
                policy,
                batch,
                optimizer,
                cfg.optimizer.grad_clip_norm,
                accelerator=accelerator,
                lr_scheduler=lr_scheduler,
                sample_weighter=sample_weighter,
            )
        if not accelerator.sync_gradients:
            continue

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
""",
    ),
    (
        """                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size,
""",
        """                    num_processes=accelerator.num_processes,
                    batch_size=cfg.batch_size * gradient_accumulation_steps,
""",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def trainer_path() -> Path:
    spec = importlib.util.find_spec("lerobot.scripts.lerobot_train")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate lerobot.scripts.lerobot_train")
    return Path(spec.origin)


def patch_trainer(path: Path, *, check_only: bool = False) -> bool:
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(f"expected lerobot=={LEROBOT_VERSION}, found {version}")
    source = path.read_text()
    if PATCH_MARKER in source:
        verification_source = source.replace(
            "import os\nimport shutil\nimport sys\n",
            "import os\nimport sys\n",
        )
        for _, replacement in REPLACEMENTS:
            probe = replacement
            if replacement.startswith(f"\n\n{PATCH_MARKER}\n"):
                probe = replacement.rsplit("\n\ndef update_policy(\n", maxsplit=1)[0]
            if probe not in verification_source:
                raise RuntimeError(f"incomplete gradient-accumulation patch in {path}")
        return False
    if check_only:
        raise RuntimeError(f"gradient-accumulation patch is not active in {path}")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected LeRobot trainer {path}: sha256={digest}, "
            f"expected {ORIGINAL_SHA256}"
        )
    patched = source
    for anchor, replacement in REPLACEMENTS:
        if patched.count(anchor) != 1:
            raise RuntimeError(f"LeRobot trainer patch anchor is not unique: {anchor[:80]!r}")
        patched = patched.replace(anchor, replacement)
    compile(patched, str(path), "exec")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(patched)
    temporary.replace(path)
    return True


def main() -> None:
    args = parse_args()
    path = trainer_path()
    changed = patch_trainer(path, check_only=args.check)
    state = "patched" if changed else "verified"
    print(f"LeRobot gradient accumulation {state}: {path}")


if __name__ == "__main__":
    main()
