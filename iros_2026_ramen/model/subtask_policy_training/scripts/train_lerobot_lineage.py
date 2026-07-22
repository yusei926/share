#!/usr/bin/env python3
"""Run LeRobot training with the repository's deterministic lineage sampler."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.lineage_sampling import (  # noqa: E402
    PLAN_ENV,
    LineageBalancedSampler,
)


def main() -> None:
    plan_path = Path(os.environ.get(PLAN_ENV, "")).expanduser()
    if not plan_path.is_file():
        raise FileNotFoundError(f"{PLAN_ENV} must point to a sampling plan: {plan_path}")

    import lerobot.scripts.lerobot_train as lerobot_train

    # LeRobot 0.6 constructs this class inside train(). Replacing that imported
    # symbol keeps the upstream trainer, checkpointing, and Accelerate behavior
    # intact while changing only its deterministic frame sampler.
    lerobot_train.EpisodeAwareSampler = LineageBalancedSampler
    lerobot_train.main()


if __name__ == "__main__":
    main()
