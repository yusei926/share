"""Vit4HeadLoss の mask / 合算 / NaN 耐性 テスト."""

from __future__ import annotations

import pytest
import torch

from model.vit_phase1.train.losses import Vit4HeadLoss


def _make_batch(batch_size: int = 4, num_skills: int = 8, *, predictable=True, skip=False):
    """テスト用 batch を生成。preds は logits、targets は spec 通り."""
    preds = {
        "skill_logits": torch.randn(batch_size, num_skills, requires_grad=True),
        "phase": torch.rand(batch_size, requires_grad=True),
        "tte_log": torch.randn(batch_size, requires_grad=True),
        "anomaly_logit": torch.randn(batch_size, requires_grad=True),
    }
    targets = {
        "skill_id": torch.randint(0, num_skills, (batch_size,), dtype=torch.long),
        "phase": torch.rand(batch_size),
        "tte_log": torch.randn(batch_size),
        "anomaly": torch.randint(0, 2, (batch_size,)).float(),
        "predictable_mask": torch.full((batch_size,), predictable, dtype=torch.bool),
        "skip_mask": torch.full((batch_size,), skip, dtype=torch.bool),
    }
    return preds, targets


def _default_loss(num_skills: int = 8) -> Vit4HeadLoss:
    return Vit4HeadLoss(
        class_weight=torch.ones(num_skills),
        pos_weight=1.0,
        weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
        label_smoothing=0.1,
    )


def test_forward_returns_all_keys():
    loss_fn = _default_loss()
    preds, targets = _make_batch()
    out = loss_fn(preds, targets)
    expected = {"loss", "loss_skill", "loss_phase", "loss_tte", "loss_anomaly"}
    assert set(out.keys()) == expected


def test_combined_loss_weighted_sum():
    """total = w_skill*L_skill + w_phase*L_phase + w_tte*L_tte + w_anom*L_anom."""
    weights = {"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5}
    loss_fn = Vit4HeadLoss(torch.ones(8), 1.0, weights, label_smoothing=0.1)
    preds, targets = _make_batch()
    out = loss_fn(preds, targets)
    expected = (
        weights["skill"] * out["loss_skill"]
        + weights["phase"] * out["loss_phase"]
        + weights["tte"] * out["loss_tte"]
        + weights["anomaly"] * out["loss_anomaly"]
    )
    assert torch.isclose(out["loss"], expected, atol=1e-6)


def test_phase_mask_zeros_out_non_predictable():
    """predictable_mask=0 の frame の phase 誤差は合算に寄与しないこと."""
    loss_fn = _default_loss()

    # predictable=True で通常 loss
    preds_ok, tgt_ok = _make_batch(predictable=True)
    out_ok = loss_fn(preds_ok, tgt_ok)

    # 同じ preds/target で predictable=False にすると phase/tte loss = 0
    preds_masked, tgt_masked = _make_batch(predictable=False)
    # 同じ preds/target を差し替え (mask 以外を揃える)
    for k in preds_ok:
        preds_masked[k] = preds_ok[k].detach().clone().requires_grad_(True)
    for k in tgt_ok:
        if k in {"predictable_mask", "skip_mask"}:
            continue
        tgt_masked[k] = tgt_ok[k].clone()
    out_masked = loss_fn(preds_masked, tgt_masked)

    # skill / anomaly は同値、phase / tte は 0 に
    assert torch.isclose(out_ok["loss_skill"], out_masked["loss_skill"])
    assert torch.isclose(out_ok["loss_anomaly"], out_masked["loss_anomaly"])
    assert out_masked["loss_phase"].item() == 0.0
    assert out_masked["loss_tte"].item() == 0.0


def test_skip_mask_zeros_out_phase_tte():
    """skip_mask=1 の frame (single-frame run) は phase/tte loss から除外."""
    loss_fn = _default_loss()

    preds, tgt = _make_batch(predictable=True, skip=False)
    out_full = loss_fn(preds, tgt)

    preds2, tgt2 = _make_batch(predictable=True, skip=True)
    for k in preds:
        preds2[k] = preds[k].detach().clone().requires_grad_(True)
    for k in tgt:
        if k == "skip_mask":
            continue
        tgt2[k] = tgt[k].clone()
    out_skipped = loss_fn(preds2, tgt2)

    assert out_skipped["loss_phase"].item() == 0.0
    assert out_skipped["loss_tte"].item() == 0.0
    # skill / anomaly は影響なし
    assert torch.isclose(out_full["loss_skill"], out_skipped["loss_skill"])


def test_all_masked_batch_returns_zero_not_nan():
    """batch 全 mask=0 でも NaN にならず 0 が返ること."""
    loss_fn = _default_loss()
    preds, targets = _make_batch(predictable=False, skip=True)  # 全 mask=0
    out = loss_fn(preds, targets)
    assert not torch.isnan(out["loss"])
    assert out["loss_phase"].item() == 0.0
    assert out["loss_tte"].item() == 0.0
    # skill / anomaly は mask 対象外なので依然として > 0
    assert out["loss_skill"].item() > 0


def test_class_weight_length_assertion():
    """class_weight が誤ったサイズなら init で assert."""
    with pytest.raises(AssertionError):
        Vit4HeadLoss(
            class_weight=torch.ones(3, 3),  # 2-D
            pos_weight=1.0,
            weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
        )


def test_weights_key_assertion():
    """weights dict に必要 key が欠けたら init で assert."""
    with pytest.raises(AssertionError):
        Vit4HeadLoss(
            class_weight=torch.ones(8),
            pos_weight=1.0,
            weights={"skill": 1.0, "phase": 1.0, "tte": 0.3},  # anomaly 欠落
        )


def test_label_smoothing_prevents_zero_loss():
    """label_smoothing > 0 なら完全 prediction でも loss > 0 (calibration)."""
    loss_fn = _default_loss()
    B, K = 4, 8
    preds = {
        "skill_logits": torch.zeros(B, K),
        "phase": torch.zeros(B),
        "tte_log": torch.zeros(B),
        "anomaly_logit": torch.zeros(B),
    }
    # 完全に正しい target (skill_logits はすべて 0 だが特定クラスを正解に)
    # → 正解クラス確率 1/K (uniform)、smoothed target との CE > 0
    targets = {
        "skill_id": torch.zeros(B, dtype=torch.long),
        "phase": torch.zeros(B),
        "tte_log": torch.zeros(B),
        "anomaly": torch.zeros(B),
        "predictable_mask": torch.ones(B, dtype=torch.bool),
        "skip_mask": torch.zeros(B, dtype=torch.bool),
    }
    out = loss_fn(preds, targets)
    # label_smoothing=0.1 なので skill CE は必ず > 0
    assert out["loss_skill"].item() > 0.0


def test_backward_flows_to_preds():
    """total.backward() で preds 側 gradient が流れる (frozen 化バグ検出)."""
    loss_fn = _default_loss()
    preds, targets = _make_batch()
    out = loss_fn(preds, targets)
    out["loss"].backward()
    for k, v in preds.items():
        assert v.grad is not None, f"preds[{k}] に grad が流れていない"


# ------------------------------------------------------------
# worldstate head
# ------------------------------------------------------------
def _make_batch_5head(batch_size: int = 4, num_skills: int = 8, num_legs: int = 5,
                      *, predictable=True, skip=False):
    """5 head 用 batch (base 4 head + worldstate)."""
    preds, targets = _make_batch(batch_size, num_skills, predictable=predictable, skip=skip)
    preds["worldstate_logits"] = torch.randn(batch_size, num_legs, requires_grad=True)
    targets["num_legs_inserted"] = torch.randint(0, num_legs, (batch_size,), dtype=torch.long)
    return preds, targets


def _default_loss_5head(num_skills: int = 8, num_legs: int = 5) -> Vit4HeadLoss:
    return Vit4HeadLoss(
        class_weight=torch.ones(num_skills),
        pos_weight=1.0,
        weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5, "worldstate": 1.0},
        label_smoothing=0.1,
        class_weight_worldstate=torch.ones(num_legs),
        label_smoothing_worldstate=0.1,
    )


def test_worldstate_disabled_by_default_output_keys():
    """class_weight_worldstate=None (default) で output に loss_worldstate 無し (backward-compat)."""
    loss_fn = _default_loss()
    preds, targets = _make_batch()
    out = loss_fn(preds, targets)
    assert "loss_worldstate" not in out
    assert not loss_fn.include_worldstate


def test_worldstate_enabled_returns_loss_worldstate():
    """class_weight_worldstate 有りで output に loss_worldstate が返る."""
    loss_fn = _default_loss_5head()
    preds, targets = _make_batch_5head()
    out = loss_fn(preds, targets)
    assert "loss_worldstate" in out
    assert out["loss_worldstate"].item() > 0
    assert loss_fn.include_worldstate


def test_worldstate_combined_loss_weighted_sum_5head():
    """total = 5 head 合算 (base 4 + worldstate)."""
    weights = {"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5, "worldstate": 0.7}
    loss_fn = Vit4HeadLoss(
        class_weight=torch.ones(8),
        pos_weight=1.0,
        weights=weights,
        class_weight_worldstate=torch.ones(5),
    )
    preds, targets = _make_batch_5head()
    out = loss_fn(preds, targets)
    expected = (
        weights["skill"] * out["loss_skill"]
        + weights["phase"] * out["loss_phase"]
        + weights["tte"] * out["loss_tte"]
        + weights["anomaly"] * out["loss_anomaly"]
        + weights["worldstate"] * out["loss_worldstate"]
    )
    assert torch.isclose(out["loss"], expected, atol=1e-6)


def test_worldstate_missing_weight_key_assertion():
    """class_weight_worldstate 有りで weights['worldstate'] が無いと assert."""
    with pytest.raises(AssertionError):
        Vit4HeadLoss(
            class_weight=torch.ones(8),
            pos_weight=1.0,
            weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},  # worldstate 欠落
            class_weight_worldstate=torch.ones(5),
        )


def test_worldstate_backward_flows():
    """5 head 版でも worldstate_logits に grad が流れる."""
    loss_fn = _default_loss_5head()
    preds, targets = _make_batch_5head()
    out = loss_fn(preds, targets)
    out["loss"].backward()
    for k, v in preds.items():
        assert v.grad is not None, f"preds[{k}] に grad が流れていない"


# ---------- Pass2 (Issue #36): phase 末尾 weight boost ----------

def test_phase_tail_weight_default_matches_no_boost():
    """tail_weight_ratio=1.0 (default) で無効化、旧挙動と一致."""
    torch.manual_seed(0)
    loss_default = _default_loss()
    preds, targets = _make_batch(predictable=True)
    out_default = loss_default(preds, targets)

    torch.manual_seed(0)
    loss_no_boost = Vit4HeadLoss(
        class_weight=torch.ones(8), pos_weight=1.0,
        weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
        label_smoothing=0.1,
        phase_tail_weight_ratio=1.0,
    )
    torch.manual_seed(0)
    p2, t2 = _make_batch(predictable=True)
    out_no_boost = loss_no_boost(p2, t2)
    assert torch.isclose(out_default["loss_phase"], out_no_boost["loss_phase"], atol=1e-6)


def test_phase_tail_weight_boosts_only_tail_frames():
    """target phase >= threshold の frame の per-frame loss が N 倍される."""
    loss_fn = Vit4HeadLoss(
        class_weight=torch.ones(8), pos_weight=1.0,
        weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
        label_smoothing=0.1,
        phase_tail_weight_ratio=3.0,
        phase_tail_threshold=0.9,
    )
    # 半分 tail、半分 head の合成 target
    preds = {
        "skill_logits": torch.zeros(4, 8, requires_grad=True),
        "phase": torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True),
        "tte_log": torch.zeros(4, requires_grad=True),
        "anomaly_logit": torch.zeros(4, requires_grad=True),
    }
    targets = {
        "skill_id": torch.tensor([0, 0, 0, 0]),
        "phase": torch.tensor([0.0, 0.0, 0.95, 0.95]),  # 前 2 = head、後 2 = tail
        "tte_log": torch.zeros(4),
        "anomaly": torch.zeros(4),
        "predictable_mask": torch.ones(4, dtype=torch.bool),
        "skip_mask": torch.zeros(4, dtype=torch.bool),
    }
    out = loss_fn(preds, targets)
    # per-frame MSE: head=(0.5-0)^2=0.25、tail=(0.5-0.95)^2=0.2025
    # weight (ratio=3.0): head=1.0, tail=3.0
    # weighted sum = 2*0.25*1.0 + 2*0.2025*3.0 = 0.5 + 1.215 = 1.715
    # denom = 2*1.0 + 2*3.0 = 8.0
    # weighted mean = 1.715 / 8.0 = 0.214375
    expected = (2 * 0.25 * 1.0 + 2 * 0.2025 * 3.0) / (2 * 1.0 + 2 * 3.0)
    assert torch.isclose(out["loss_phase"], torch.tensor(expected), atol=1e-4)


def test_phase_tail_weight_respects_predictable_mask():
    """predictable=False の frame は tail であっても boost 対象にならない."""
    loss_fn = Vit4HeadLoss(
        class_weight=torch.ones(8), pos_weight=1.0,
        weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
        label_smoothing=0.1,
        phase_tail_weight_ratio=5.0, phase_tail_threshold=0.8,
    )
    preds = {
        "skill_logits": torch.zeros(2, 8, requires_grad=True),
        "phase": torch.tensor([0.0, 0.0], requires_grad=True),
        "tte_log": torch.zeros(2, requires_grad=True),
        "anomaly_logit": torch.zeros(2, requires_grad=True),
    }
    targets = {
        "skill_id": torch.tensor([0, 0]),
        "phase": torch.tensor([0.9, 0.9]),
        "tte_log": torch.zeros(2),
        "anomaly": torch.zeros(2),
        "predictable_mask": torch.tensor([False, False]),  # 全 mask
        "skip_mask": torch.zeros(2, dtype=torch.bool),
    }
    out = loss_fn(preds, targets)
    assert out["loss_phase"].item() == 0.0


def test_phase_tail_weight_ratio_below_one_asserts():
    with pytest.raises(AssertionError):
        Vit4HeadLoss(
            class_weight=torch.ones(8), pos_weight=1.0,
            weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
            phase_tail_weight_ratio=0.5,
        )


def test_phase_tail_threshold_out_of_range_asserts():
    with pytest.raises(AssertionError):
        Vit4HeadLoss(
            class_weight=torch.ones(8), pos_weight=1.0,
            weights={"skill": 1.0, "phase": 1.0, "tte": 0.3, "anomaly": 0.5},
            phase_tail_threshold=1.5,
        )
