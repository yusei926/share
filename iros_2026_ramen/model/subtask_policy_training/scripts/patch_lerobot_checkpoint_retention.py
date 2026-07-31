#!/usr/bin/env python3
"""Bound local checkpoint storage after durable remote uploads."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

LEROBOT_VERSION = "0.6.0"
GRADIENT_PATCH_MARKER = "# TEAM_RAMEN_GRADIENT_ACCUMULATION_V1"
PATCH_MARKER = "# TEAM_RAMEN_CHECKPOINT_RETENTION_V1"

REPLACEMENTS = (
    (
        "import os\nimport sys\n",
        "import os\nimport shutil\nimport sys\n",
    ),
    (
        "\n\ndef update_policy(\n",
        """

# TEAM_RAMEN_CHECKPOINT_RETENTION_V1
def _prune_uploaded_checkpoints(checkpoint_dir) -> None:
    raw = os.environ.get("LEROBOT_LOCAL_CHECKPOINT_KEEP", "").strip()
    if not raw:
        return
    try:
        keep = int(raw)
    except ValueError as exc:
        raise ValueError("LEROBOT_LOCAL_CHECKPOINT_KEEP must be an integer") from exc
    if keep < 1:
        raise ValueError("LEROBOT_LOCAL_CHECKPOINT_KEEP must be positive")

    checkpoints = sorted(
        path
        for path in checkpoint_dir.parent.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    for old_checkpoint in checkpoints[:-keep]:
        shutil.rmtree(old_checkpoint)
        logging.info(
            "Removed remotely backed local checkpoint %s",
            old_checkpoint,
        )


def update_policy(
""",
    ),
    (
        """                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()
""",
        """                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
                if cfg.save_checkpoint_to_hub:
                    _prune_uploaded_checkpoints(checkpoint_dir)

            accelerator.wait_for_everyone()
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
    if GRADIENT_PATCH_MARKER not in source:
        raise RuntimeError("gradient-accumulation patch must be applied first")
    if PATCH_MARKER in source:
        legacy_signature = "def _prune_uploaded_checkpoints(checkpoint_dir: Path) -> None:"
        current_signature = "def _prune_uploaded_checkpoints(checkpoint_dir) -> None:"
        changed = legacy_signature in source
        if changed:
            source = source.replace(legacy_signature, current_signature)
        for _, replacement in REPLACEMENTS:
            if replacement not in source:
                raise RuntimeError(f"incomplete checkpoint-retention patch in {path}")
        if changed:
            compile(source, str(path), "exec")
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(source)
            temporary.replace(path)
        return changed
    if check_only:
        raise RuntimeError(f"checkpoint-retention patch is not active in {path}")

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
    print(f"LeRobot checkpoint retention {state}: {path}")


if __name__ == "__main__":
    main()
