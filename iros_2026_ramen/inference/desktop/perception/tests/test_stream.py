"""DetectionStream の unit test。

主眼:
    - warmup 期間の Optional[None] 挙動
    - flush の残り frame emit
    - batch API (`clean_all_frames`) との等価性 (中央 index 範囲で)
    - median enabled / disabled 両方の latency 差分
"""

from __future__ import annotations

import numpy as np
import pytest

from inference.desktop.perception.cleaner import clean_all_frames
from inference.desktop.perception.stream import DetectionStream
from inference.desktop.perception.yolo_obb import OBBDetection


def _det(
    class_name: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    class_id: int = 0,
    confidence: float = 0.9,
) -> OBBDetection:
    """axis-aligned bbox から OBB 4 頂点 (時計回り) を作って detection にする。"""
    verts = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    return OBBDetection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        verts=verts,
    )


def _base_config(median_enabled: bool) -> dict:
    """test 用 config (yaml 経由じゃなく直接 dict を渡す)。"""
    return {
        "max_count": {"leg": 4, "hand_left": 1, "hand_right": 1},
        "over_max_continue_iou": 0.3,
        "under_max_similar_iou": 0.3,
        "median_filter": {"enabled": median_enabled, "iou_match_min": 0.5},
    }


# =========================================================================
# median disabled (1 frame delay)
# =========================================================================
class TestMedianDisabled:
    def test_first_push_returns_none(self):
        stream = DetectionStream(_base_config(median_enabled=False))
        assert stream.push([_det("leg", 0.1, 0.1, 0.2, 0.2)]) is None

    def test_second_push_emits_first_cleaned(self):
        """1 frame delay: 2 回目の push で 1 frame 目の cleaned が emit される。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]  # 1 frame 目に似た位置
        stream.push(f0)
        out = stream.push(f1)
        assert out is not None
        # f0 の leg は f1 に similar → continuing で保持される
        assert len(out) == 1
        assert out[0].class_name == "leg"

    def test_flush_emits_last_frame(self):
        """flush で最後の raw を next_raw=[] で clean、emit。"""
        stream = DetectionStream(_base_config(median_enabled=False))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        stream.push(f0)
        stream.push(f1)
        stream.push(f2)
        remaining = stream.flush()
        # flush で cleaned_2 emit
        assert len(remaining) == 1


# =========================================================================
# median enabled (2 frame delay)
# =========================================================================
class TestMedianEnabled:
    def test_first_three_pushes_return_none(self):
        """median enabled: warmup は Step 1 x1 + median x2 = 計 3 push 必要。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        assert stream.push(f0) is None
        assert stream.push(f1) is None
        assert stream.push(f2) is None

    def test_fourth_push_emits_first_median(self):
        """4 push 目で cleaned_1 の 3-tap median が emit される。

        flow:
            push(f0): Step 1 warmup (raw_curr=None) → None
            push(f1): Step 1 で cleaned_0、median_ring size 1 → None
            push(f2): Step 1 で cleaned_1、median_ring size 2 → None
            push(f3): Step 1 で cleaned_2、median_ring size 3 → emit median(c0,c1,c2)
        """
        stream = DetectionStream(_base_config(median_enabled=True))
        f0 = [_det("leg", 0.10, 0.10, 0.20, 0.20)]
        f1 = [_det("leg", 0.11, 0.11, 0.21, 0.21)]
        f2 = [_det("leg", 0.12, 0.12, 0.22, 0.22)]
        f3 = [_det("leg", 0.13, 0.13, 0.23, 0.23)]
        stream.push(f0)
        stream.push(f1)
        stream.push(f2)
        out = stream.push(f3)
        assert out is not None
        assert len(out) == 1


# =========================================================================
# batch API との等価性 (最重要 regression 防止)
# =========================================================================
def _run_streaming(
    all_dets: list[list[OBBDetection]], config: dict
) -> list[list[OBBDetection]]:
    """streaming で全 frame push + flush して emit を集める。"""
    stream = DetectionStream(config)
    emitted: list[list[OBBDetection]] = []
    for raw in all_dets:
        out = stream.push(raw)
        if out is not None:
            emitted.append(out)
    emitted.extend(stream.flush())
    return emitted


def _sample_ep(n: int) -> list[list[OBBDetection]]:
    """n frame の simple ep (leg が徐々に移動する) を生成。"""
    ep = []
    for t in range(n):
        offset = 0.001 * t
        ep.append(
            [
                _det("leg", 0.10 + offset, 0.10 + offset, 0.20 + offset, 0.20 + offset),
                _det("hand_left", 0.50, 0.50, 0.60, 0.60),
                _det("hand_right", 0.65, 0.55, 0.75, 0.65),
            ]
        )
    return ep


class TestBatchEquivalence:
    @pytest.mark.parametrize("median_enabled", [True, False])
    @pytest.mark.parametrize("n_frames", [5, 10, 20])
    def test_middle_frames_match_batch(
        self, median_enabled: bool, n_frames: int
    ):
        """streaming の emit と batch API の中央 index 範囲が verts で一致する。

        - median disabled 時: streaming は全 n frame emit、batch と 1:1 一致すべき
        - median enabled 時: streaming は index 1..n-2 emit、batch との対応 index で一致
        """
        config = _base_config(median_enabled=median_enabled)
        ep = _sample_ep(n_frames)

        batch_out = clean_all_frames(ep, config)
        streaming_out = _run_streaming(ep, config)

        if not median_enabled:
            # median disabled: streaming は n frame emit (batch と同数)
            assert len(streaming_out) == n_frames
            for t in range(n_frames):
                _assert_frame_equal(streaming_out[t], batch_out[t])
        else:
            # median enabled: streaming は index 1..n-2 emit (batch の n-2 個と対応)
            assert len(streaming_out) == n_frames - 2
            for i, t in enumerate(range(1, n_frames - 1)):
                _assert_frame_equal(streaming_out[i], batch_out[t])


def _assert_frame_equal(
    a: list[OBBDetection], b: list[OBBDetection]
) -> None:
    """2 つの frame の detection list が (class_name, verts) で等価か検証。"""
    assert len(a) == len(b), f"detection count mismatch: {len(a)} vs {len(b)}"
    # class_name でグループ化して比較 (順序不問)
    def _key(d: OBBDetection) -> tuple:
        return (d.class_name, tuple(d.verts.flatten().tolist()))

    a_sorted = sorted(a, key=_key)
    b_sorted = sorted(b, key=_key)
    for da, db in zip(a_sorted, b_sorted):
        assert da.class_name == db.class_name
        np.testing.assert_allclose(da.verts, db.verts, atol=1e-5)


# =========================================================================
# 端 case
# =========================================================================
class TestEdgeCases:
    def test_empty_raw_all_frames(self):
        """全 frame で detection 0 個 → streaming も空 emit。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        for _ in range(5):
            stream.push([])
        remaining = stream.flush()
        # median enabled で 5 frame → emit は index 1..3 (3 個) + flush で index 4 の中央 = 1 個
        # ただし全 detection 空なので中身も空
        # 具体的 count は buffer 動作次第、全 emit が [] であることのみ確認
        assert all(len(r) == 0 for r in remaining)

    def test_flush_with_no_pushes(self):
        """push 0 回で flush → 空 list。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        assert stream.flush() == []

    def test_single_push_then_flush(self):
        """1 push だけ + flush の極端 case。median enabled では端 frame drop で emit 0。"""
        stream = DetectionStream(_base_config(median_enabled=True))
        stream.push([_det("leg", 0.10, 0.10, 0.20, 0.20)])
        remaining = stream.flush()
        # ring size < 3 なので flush の median emit は 0 個 (design: 端 frame drop)
        assert remaining == []
