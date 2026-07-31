from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts" / "patch_lerobot_gradient_accumulation.py"
RETENTION_PATCH_PATH = ROOT / "scripts" / "patch_lerobot_checkpoint_retention.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("gradient_accumulation_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gradient_accumulation_patch_is_hash_guarded_and_idempotent(tmp_path: Path) -> None:
    module = _load_patch_module()
    patched_source = module.trainer_path().read_text()
    original_source = patched_source
    retention_spec = importlib.util.spec_from_file_location(
        "checkpoint_retention_patch",
        RETENTION_PATCH_PATH,
    )
    assert retention_spec is not None and retention_spec.loader is not None
    retention_module = importlib.util.module_from_spec(retention_spec)
    retention_spec.loader.exec_module(retention_module)
    for anchor, replacement in reversed(retention_module.REPLACEMENTS):
        original_source = original_source.replace(replacement, anchor)
    for anchor, replacement in reversed(module.REPLACEMENTS):
        original_source = original_source.replace(replacement, anchor)
    assert hashlib.sha256(original_source.encode()).hexdigest() == module.ORIGINAL_SHA256

    target = tmp_path / "lerobot_train.py"
    target.write_text(original_source)
    assert module.patch_trainer(target)
    assert not module.patch_trainer(target, check_only=True)
    compile(target.read_text(), str(target), "exec")


def test_gradient_accumulation_check_accepts_checkpoint_retention_patch() -> None:
    module = _load_patch_module()
    assert not module.patch_trainer(module.trainer_path(), check_only=True)


def test_gradient_accumulation_performs_one_optimizer_update() -> None:
    torch = pytest.importorskip("torch")
    accelerate = pytest.importorskip("accelerate")
    trainer = pytest.importorskip("lerobot.scripts.lerobot_train")

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, batch):
            del batch
            return self.weight.square(), {}

    accelerator = accelerate.Accelerator(
        cpu=True,
        gradient_accumulation_steps=4,
        step_scheduler_with_optimizer=False,
    )
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    policy, optimizer = accelerator.prepare(policy, optimizer)
    metrics = SimpleNamespace()
    observed = []
    for _ in range(4):
        with accelerator.accumulate(policy):
            trainer.update_policy(
                metrics,
                policy,
                {},
                optimizer,
                0.0,
                accelerator=accelerator,
            )
        observed.append(float(accelerator.unwrap_model(policy).weight.detach()))

    assert observed[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert observed[3] == pytest.approx(0.8)
