"""YoloObbPerception の smoke test。

- 実 weight (m_lowaug_v3/best.pt) と predict_video_ep64.py が使うのと同じ
  head cam frame を 1 枚使い、predict() が想定形式で返ることを検証する。
- weight or ultralytics 依存が無い環境では skip (CI 分離)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WEIGHT_PATH = REPO_ROOT / "model/yolo_obb/runs/m_lowaug_v3/weights/best.pt"

pytestmark = pytest.mark.skipif(
    not WEIGHT_PATH.exists(),
    reason=f"weight not found: {WEIGHT_PATH}",
)


def test_init_fails_fast_on_missing_weight(tmp_path: Path) -> None:
    """weight file が無ければ init 時に FileNotFoundError。"""
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    with pytest.raises(FileNotFoundError):
        YoloObbPerception(tmp_path / "does_not_exist.pt")


def test_init_rejects_out_of_range_conf() -> None:
    """conf は [0, 1] 範囲外なら ValueError。"""
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    with pytest.raises(ValueError):
        YoloObbPerception(WEIGHT_PATH, conf=1.5)


def test_predict_rejects_bad_frame_shape() -> None:
    """predict() は (H, W, 3) 以外の shape を弾く。"""
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    perception = YoloObbPerception(WEIGHT_PATH)
    with pytest.raises(ValueError):
        perception.predict(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        perception.predict(np.zeros((480, 640), dtype=np.uint8))
    with pytest.raises(ValueError):
        perception.predict(np.zeros((480, 640, 4), dtype=np.uint8))


def test_class_names_matches_dataset_yaml() -> None:
    """class_names は dataset.yaml の 7 class と一致する (m_lowaug_v3 前提)。"""
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    perception = YoloObbPerception(WEIGHT_PATH)
    expected = {
        0: "workspace",
        1: "leg",
        2: "leg_tip",
        3: "hole",
        4: "table_top",
        5: "hand_right",
        6: "hand_left",
    }
    assert perception.class_names == expected


def test_predict_returns_wellformed_detections_on_synthetic_frame() -> None:
    """合成 frame でも predict() は list[OBBDetection] を返す (検出 0 でも空 list、error にしない)。

    合成 frame は真っ黒でモデルは何も検出しない可能性が高いが、それはそれで
    空 list を返すことが仕様。verts が shape (4, 2) normalized [0, 1] であることも
    検出があった場合は検証する。
    """
    from inference.desktop.perception.yolo_obb import OBBDetection, YoloObbPerception

    perception = YoloObbPerception(WEIGHT_PATH)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = perception.predict(frame)

    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, OBBDetection)
        assert 0.0 <= d.confidence <= 1.0
        assert d.class_name in perception.class_names.values()
        assert d.verts.shape == (4, 2)
        # normalized 座標: すべて [0, 1] 範囲 (近似で少し外にはみ出しても許容 +/- 0.02)
        assert (d.verts >= -0.02).all() and (d.verts <= 1.02).all()
