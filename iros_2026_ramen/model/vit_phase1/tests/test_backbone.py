"""DinoV3Backbone smoke tests.

Gated HF repo を実 download するため、`HF_TOKEN` が無い環境では skip する。
"""

from __future__ import annotations

import os

import pytest
import torch
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import hf_hub_download

# Workspace root の .env から HF_TOKEN を読む
load_dotenv(find_dotenv(usecwd=True))

DINOV3_REPO_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _has_dinov3_access() -> tuple[bool, str]:
    if not os.getenv("HF_TOKEN"):
        return False, "HF_TOKEN required for gated dinov3 repo"
    try:
        hf_hub_download(DINOV3_REPO_ID, "config.json")
    except Exception as exc:  # noqa: BLE001
        return False, f"{DINOV3_REPO_ID} access required ({type(exc).__name__})"
    return True, ""


HAS_DINOV3_ACCESS, DINOV3_SKIP_REASON = _has_dinov3_access()
requires_hf = pytest.mark.skipif(not HAS_DINOV3_ACCESS, reason=DINOV3_SKIP_REASON)


@requires_hf
def test_forward_shape_and_no_grad():
    from model.vit_phase1.model.backbone import DinoV3Backbone

    bb = DinoV3Backbone()
    x = torch.randn(2, 3, 384, 384)
    out = bb(x)
    assert out.shape == (2, 384), f"expected (2, 384), got {tuple(out.shape)}"
    assert not out.requires_grad, "backbone output must be detached (Knowledge Insulation)"
    assert bb.out_dim == 384


@requires_hf
def test_buffers_registered():
    from model.vit_phase1.model.backbone import DinoV3Backbone

    bb = DinoV3Backbone()
    assert bb.normalize_mean.shape == (1, 3, 1, 1)
    assert bb.normalize_std.shape == (1, 3, 1, 1)
    # ImageNet 系の典型値 (0..1 range) であること
    assert torch.all((bb.normalize_mean >= 0) & (bb.normalize_mean <= 1))
    assert torch.all((bb.normalize_std > 0) & (bb.normalize_std <= 1))


@requires_hf
def test_train_mode_locked_to_eval():
    from model.vit_phase1.model.backbone import DinoV3Backbone

    bb = DinoV3Backbone()
    # 親 module の .train() を呼んでも backbone は eval を維持
    bb.train(True)
    assert not bb.training, "backbone must remain in eval mode even when train(True) is called"
    bb.train(False)
    assert not bb.training


@requires_hf
def test_params_frozen():
    from model.vit_phase1.model.backbone import DinoV3Backbone

    bb = DinoV3Backbone()
    trainable = [p for p in bb.parameters() if p.requires_grad]
    assert len(trainable) == 0, f"backbone has {len(trainable)} trainable params (must be 0)"
