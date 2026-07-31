from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = ROOT / "scripts" / "patch_lerobot_furniture_groot_plugin.py"


def _load_patch_module():
    spec = importlib.util.spec_from_file_location("groot_plugin_patch", PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_source(module) -> str:
    source = module.factory_path().read_text()
    source = source.replace(module.PRETRAINED_REPLACEMENT, module.PRETRAINED_ANCHOR)
    source = source.replace(module.V1_PRETRAINED_REPLACEMENT, module.PRETRAINED_ANCHOR)
    source = source.replace(module.FACTORY_REPLACEMENT, module.FACTORY_ANCHOR)
    return source


def test_groot_plugin_patch_uses_connected_pretrained_loader(tmp_path: Path) -> None:
    module = _load_patch_module()
    original = _original_source(module)
    assert hashlib.sha256(original.encode()).hexdigest() == module.ORIGINAL_SHA256

    target = tmp_path / "factory.py"
    target.write_text(original)
    assert module.patch_factory(target)
    patched = target.read_text()
    assert module.PRETRAINED_REPLACEMENT in patched
    assert module.FACTORY_REPLACEMENT in patched
    assert not module.patch_factory(target, check_only=True)


def test_groot_plugin_patch_migrates_v1(tmp_path: Path) -> None:
    module = _load_patch_module()
    v1 = _original_source(module)
    v1 = v1.replace(module.PRETRAINED_ANCHOR, module.V1_PRETRAINED_REPLACEMENT, 1)
    v1 = v1.replace(module.FACTORY_ANCHOR, module.FACTORY_REPLACEMENT, 1)
    target = tmp_path / "factory.py"
    target.write_text(v1)

    with pytest.raises(RuntimeError, match="obsolete"):
        module.patch_factory(target, check_only=True)
    assert module.patch_factory(target)
    assert module.PRETRAINED_REPLACEMENT in target.read_text()
