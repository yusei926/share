from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts" / "patch_lerobot_checkpoint_retention.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("checkpoint_retention_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_retention_patch_is_idempotent(tmp_path: Path) -> None:
    module = _load_patch_module()
    source = module.trainer_path().read_text()
    source = source.replace(
        "def _prune_uploaded_checkpoints(checkpoint_dir: Path) -> None:",
        "def _prune_uploaded_checkpoints(checkpoint_dir) -> None:",
    )
    for anchor, replacement in reversed(module.REPLACEMENTS):
        source = source.replace(replacement, anchor)
    target = tmp_path / "lerobot_train.py"
    target.write_text(source)

    assert module.patch_trainer(target)
    assert not module.patch_trainer(target, check_only=True)
    compile(target.read_text(), str(target), "exec")


def test_checkpoint_retention_keeps_latest_numeric_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module()
    module.patch_trainer(module.trainer_path())
    trainer = importlib.import_module("lerobot.scripts.lerobot_train")
    trainer = importlib.reload(trainer)

    checkpoints = tmp_path / "checkpoints"
    for name in ("005000", "010000", "015000", "020000"):
        (checkpoints / name).mkdir(parents=True)
    (checkpoints / "last").symlink_to("020000")
    monkeypatch.setenv("LEROBOT_LOCAL_CHECKPOINT_KEEP", "2")

    trainer._prune_uploaded_checkpoints(checkpoints / "020000")

    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "015000",
        "020000",
        "last",
    ]
