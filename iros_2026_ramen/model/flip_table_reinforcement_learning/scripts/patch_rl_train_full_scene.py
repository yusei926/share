#!/usr/bin/env python3
"""Apply the minimal flip-table hooks to the organizer RL trainer."""

from __future__ import annotations

import argparse
from pathlib import Path


FULL_SCENE_ARGUMENT = "            enable_full_local_scene=args_cli.enable_full_local_scene,\n"
FULL_SCENE_ANCHOR = "            headless_mode=args_cli.headless,\n            physics_backend=args_cli.physics_backend,\n"
RUNNER_ANCHOR = "    runner = Runner(env, agent_cfg)\n"
LEGACY_ZERO_INIT_POLICY_BLOCK = '''


    # A residual policy must start by reproducing its teacher exactly. The
    # default random output layer can otherwise destroy a valid grasp before
    # PPO receives a useful lift signal.
    if os.environ.get("FLIP_TABLE_RL_ZERO_INIT_POLICY_OUTPUT", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }:
        import torch

        policy = getattr(runner.agent, "policy", None)
        output_layer = getattr(policy, "policy_layer", None)
        if output_layer is None:
            output_layer = getattr(policy, "output_layer", None)
        if not isinstance(output_layer, torch.nn.Linear):
            raise RuntimeError("Could not locate the PPO policy output Linear layer for zero initialization")
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)
        print("[flip_table] zero-initialized PPO policy output; initial mean action is the teacher residual")
'''
ZERO_INIT_POLICY_BLOCK = '''

    # A residual policy must start by reproducing its teacher exactly. The
    # default random output layer can otherwise destroy a valid grasp before
    # PPO receives a useful lift signal.
    if not args_cli.checkpoint and os.environ.get(
        "FLIP_TABLE_RL_ZERO_INIT_POLICY_OUTPUT", "true"
    ).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        import torch

        policy = getattr(runner.agent, "policy", None)
        output_layer = getattr(policy, "policy_layer", None)
        if output_layer is None:
            output_layer = getattr(policy, "output_layer", None)
        if not isinstance(output_layer, torch.nn.Linear):
            raise RuntimeError("Could not locate the PPO policy output Linear layer for zero initialization")
        torch.nn.init.zeros_(output_layer.weight)
        torch.nn.init.zeros_(output_layer.bias)
        print("[flip_table] zero-initialized PPO policy output; initial mean action is the teacher residual")
    elif args_cli.checkpoint:
        print("[flip_table] preserving checkpoint policy output layer during resumed training")
'''


def patch_train_script(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "scripts" / "rl" / "train.py"
    text = target.read_text(encoding="utf-8")
    changed = False
    if FULL_SCENE_ARGUMENT not in text:
        if FULL_SCENE_ANCHOR not in text:
            raise RuntimeError(f"Could not find parse_env_cfg anchor in {target}")
        text = text.replace(FULL_SCENE_ANCHOR, FULL_SCENE_ARGUMENT + FULL_SCENE_ANCHOR, 1)
        changed = True
    if LEGACY_ZERO_INIT_POLICY_BLOCK.strip() in text:
        text = text.replace(LEGACY_ZERO_INIT_POLICY_BLOCK, "", 1)
        changed = True
    if ZERO_INIT_POLICY_BLOCK.strip() not in text:
        if RUNNER_ANCHOR not in text:
            raise RuntimeError(f"Could not find Runner anchor in {target}")
        text = text.replace(RUNNER_ANCHOR, RUNNER_ANCHOR + ZERO_INIT_POLICY_BLOCK, 1)
        changed = True
    if changed:
        target.write_text(text, encoding="utf-8")
        print(f"[flip_table] applied RL trainer hooks: {target}", flush=True)
    else:
        print(f"[flip_table] RL trainer hooks already active: {target}", flush=True)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robofinals-root", type=Path, required=True)
    args = parser.parse_args()
    patch_train_script(args.robofinals_root)


if __name__ == "__main__":
    main()
