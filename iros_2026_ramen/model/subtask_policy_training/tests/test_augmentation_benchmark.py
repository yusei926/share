from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_augmentation_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_augmentation_benchmark_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_comparison_matrix_uses_one_seed_and_budget_per_policy() -> None:
    module = _load_module()
    jobs = module.build_jobs(
        output_root=Path("/tmp/benchmark"),
        seed=17,
        act_steps=123,
        flow_steps=456,
    )

    assert {(job.policy, job.condition) for job in jobs} == {
        (policy, condition)
        for policy in ("act", "flow_matching")
        for condition in ("real_only", "real_sim_teleop", "real_sim_teleop_mimic")
    }
    assert {job.steps for job in jobs if job.policy == "act"} == {123}
    assert {job.steps for job in jobs if job.policy == "flow_matching"} == {456}
    assert all(job.command[-2:] == ("--seed", "17") for job in jobs)
