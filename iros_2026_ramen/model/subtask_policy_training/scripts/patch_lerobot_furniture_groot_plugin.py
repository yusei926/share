#!/usr/bin/env python3
"""Let GrootConfig subclasses use their own third-party processor factory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path

LEROBOT_VERSION = "0.6.0"
ORIGINAL_SHA256 = "7d8a74352cb8691bd59de44a478b6f3a2bbe212583fdb04a5c19b669311b4ebe"
PATCH_MARKER_V1 = "# TEAM_RAMEN_GROOT_SUBCLASS_PLUGIN_V1"
PATCH_MARKER = "# TEAM_RAMEN_GROOT_SUBCLASS_PLUGIN_V2"
PRETRAINED_ANCHOR = "        if isinstance(policy_cfg, GrootConfig):\n"
PRETRAINED_REPLACEMENT = (
    "        # TEAM_RAMEN_GROOT_SUBCLASS_PLUGIN_V2\n"
    "        if isinstance(policy_cfg, GrootConfig):\n"
)
FACTORY_ANCHOR = "    elif isinstance(policy_cfg, GrootConfig):\n"
FACTORY_REPLACEMENT = '    elif isinstance(policy_cfg, GrootConfig) and policy_cfg.type == "groot":\n'
V1_PRETRAINED_REPLACEMENT = (
    "        # TEAM_RAMEN_GROOT_SUBCLASS_PLUGIN_V1\n"
    '        if isinstance(policy_cfg, GrootConfig) and policy_cfg.type == "groot":\n'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def factory_path() -> Path:
    spec = importlib.util.find_spec("lerobot.policies.factory")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate lerobot.policies.factory")
    return Path(spec.origin)


def patch_factory(path: Path, *, check_only: bool = False) -> bool:
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(f"expected lerobot=={LEROBOT_VERSION}, found {version}")
    source = path.read_text()
    if PATCH_MARKER in source:
        if PRETRAINED_REPLACEMENT not in source or FACTORY_REPLACEMENT not in source:
            raise RuntimeError(f"incomplete Furniture-GR00T plugin patch in {path}")
        return False
    if PATCH_MARKER_V1 in source:
        if check_only:
            raise RuntimeError(f"obsolete Furniture-GR00T V1 plugin patch in {path}")
        if source.count(V1_PRETRAINED_REPLACEMENT) != 1:
            raise RuntimeError(f"incomplete Furniture-GR00T V1 plugin patch in {path}")
        patched = source.replace(V1_PRETRAINED_REPLACEMENT, PRETRAINED_REPLACEMENT)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(patched)
        temporary.replace(path)
        return True
    if check_only:
        raise RuntimeError(f"Furniture-GR00T plugin patch is not active in {path}")
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != ORIGINAL_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected LeRobot source {path}: sha256={digest}, "
            f"expected {ORIGINAL_SHA256}"
        )
    if source.count(PRETRAINED_ANCHOR) != 1 or source.count(FACTORY_ANCHOR) != 1:
        raise RuntimeError(f"LeRobot plugin patch anchors are not unique in {path}")
    patched = source.replace(PRETRAINED_ANCHOR, PRETRAINED_REPLACEMENT).replace(
        FACTORY_ANCHOR,
        FACTORY_REPLACEMENT,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(patched)
    temporary.replace(path)
    return True


def main() -> None:
    args = parse_args()
    path = factory_path()
    changed = patch_factory(path, check_only=args.check)
    print(f"LeRobot Furniture-GR00T plugin factory {'patched' if changed else 'verified'}: {path}")


if __name__ == "__main__":
    main()
