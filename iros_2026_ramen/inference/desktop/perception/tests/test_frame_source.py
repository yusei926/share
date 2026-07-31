"""FrameSource / LerobotFrameSource / Ros2FrameSource の unit test。

Ros2FrameSource の test は cyclonedds / SDK install 無しで動くよう、__init__ を
bypass して _cb を直接叩く形。fake msg (data / header.stamp を持つ Namespace) を
渡す。cyclonedds が deserialize する CompressedImage_ instance と同じ shape
(data: bytes-like / header.stamp.sec / header.stamp.nanosec) を SimpleNamespace で
再現する。
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from inference.desktop.perception.frame_source import (
    FrameData,
    LerobotFrameSource,
    Ros2FrameSource,
)


def _make_frame(t: int, h: int = 4, w: int = 4) -> FrameData:
    rgb = np.full((h, w, 3), t % 256, dtype=np.uint8)
    return FrameData(rgb=rgb, t=t)


class TestFrameData:
    def test_frame_data_immutable(self):
        """frozen dataclass: field 書き換えは FrozenInstanceError。"""
        frame = _make_frame(t=0)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            frame.t = 1

    def test_frame_data_holds_bgr_shape(self):
        frame = _make_frame(t=5, h=8, w=12)
        assert frame.rgb.shape == (8, 12, 3)
        assert frame.t == 5


class TestLerobotFrameSource:
    def test_iterates_and_ends_with_none(self):
        """3 frame の iter → get() 3 回で順に返す、4 回目は None、以降永久 None。"""
        frames = [_make_frame(t=i) for i in range(3)]
        source = LerobotFrameSource(frames)

        got = [source.get() for _ in range(3)]
        assert all(g is not None for g in got)
        assert [g.t for g in got] == [0, 1, 2]

        # 4 回目以降は None (exhausted)
        for _ in range(5):
            assert source.get() is None

    def test_empty_iter_immediately_none(self):
        """空 iter → 最初の get() から None。"""
        source = LerobotFrameSource([])
        assert source.get() is None
        assert source.get() is None  # 繰り返しでも None

    def test_generator_iter_supported(self):
        """iterable は list じゃなく generator でも動く (LeRobot dataset は generator 想定)。"""

        def _gen():
            for i in range(2):
                yield _make_frame(t=i * 10)

        source = LerobotFrameSource(_gen())
        f0 = source.get()
        f1 = source.get()
        f2 = source.get()
        assert f0 is not None and f0.t == 0
        assert f1 is not None and f1.t == 10
        assert f2 is None


# =========================================================================
# Ros2FrameSource (rclpy install 不要、_cb を直接呼ぶ形で test)
# =========================================================================
def _make_ros2_source_no_init() -> Ros2FrameSource:
    """__init__ を bypass して internal state だけ初期化した instance を返す。

    cyclonedds / SDK install 無しの main env で test するため、SDK ChannelFactory
    経由の subscription 登録を skip し、_cb / get の behavior だけ検証する。
    """
    source = Ros2FrameSource.__new__(Ros2FrameSource)
    source._latest = None  # type: ignore[attr-defined]
    source._lock = threading.Lock()  # type: ignore[attr-defined]
    source._stereo_view = "packed"  # type: ignore[attr-defined]
    source._closed = False  # type: ignore[attr-defined]
    source._channel = MagicMock()  # type: ignore[attr-defined]
    return source


def _fake_compressed_image_msg(rgb: np.ndarray, stamp_sec: int, stamp_nanosec: int) -> object:
    """rgb を JPEG encode して CompressedImage 互換の fake msg を作る。"""
    ok, encoded = cv2.imencode(".jpg", rgb)
    assert ok, "JPEG encode failed"
    stamp = types.SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec)
    header = types.SimpleNamespace(stamp=stamp)
    return types.SimpleNamespace(data=bytes(encoded.tobytes()), header=header)


class TestRos2FrameSource:
    def test_invalid_stereo_view_rejected_before_subscription_setup(self):
        with pytest.raises(ValueError, match="stereo_view"):
            Ros2FrameSource("/camera", stereo_view="invalid")

    def test_get_returns_none_before_any_callback(self):
        source = _make_ros2_source_no_init()
        assert source.get() is None

    def test_close_is_idempotent_and_releases_reader(self):
        source = _make_ros2_source_no_init()
        channel = source._channel  # type: ignore[attr-defined]
        source._latest = _make_frame(t=1)  # type: ignore[attr-defined]

        source.close()
        source.close()

        channel.CloseReader.assert_called_once_with()
        assert source.get() is None

    def test_cb_updates_latest(self):
        """_cb を直接呼んで latest が更新される + get() で取れる。"""
        source = _make_ros2_source_no_init()
        rgb_in = np.full((8, 8, 3), 100, dtype=np.uint8)
        msg = _fake_compressed_image_msg(rgb_in, stamp_sec=2, stamp_nanosec=500_000_000)

        source._cb(msg)

        result = source.get()
        assert result is not None
        # t = 2 * 1e9 + 5e8 = 2_500_000_000
        assert result.t == 2_500_000_000
        # JPEG round-trip なので値は完全一致しないが shape は保持
        assert result.rgb.shape == (8, 8, 3)

    def test_cb_overwrites_with_latest(self):
        """後続 callback で latest が上書きされる (latest-only policy)。"""
        source = _make_ros2_source_no_init()
        rgb_a = np.full((4, 4, 3), 50, dtype=np.uint8)
        rgb_b = np.full((4, 4, 3), 200, dtype=np.uint8)

        source._cb(_fake_compressed_image_msg(rgb_a, 1, 0))
        source._cb(_fake_compressed_image_msg(rgb_b, 3, 0))

        result = source.get()
        assert result is not None
        assert result.t == 3_000_000_000  # 後の msg
        # 値の平均で B の方に近いことを確認 (JPEG 圧縮誤差込み)
        assert result.rgb.mean() > 100  # 200 に近い側

    @pytest.mark.parametrize(
        ("view", "expected_mean_check"),
        [
            ("left", lambda mean: mean < 100),
            ("right", lambda mean: mean > 100),
        ],
    )
    def test_cb_splits_packed_stereo(self, view, expected_mean_check):
        source = _make_ros2_source_no_init()
        source._stereo_view = view  # type: ignore[attr-defined]
        left = np.full((8, 8, 3), 20, dtype=np.uint8)
        right = np.full((8, 8, 3), 220, dtype=np.uint8)
        packed = np.concatenate([left, right], axis=1)

        source._cb(_fake_compressed_image_msg(packed, 1, 0))

        result = source.get()
        assert result is not None
        assert result.rgb.shape == (8, 8, 3)
        assert expected_mean_check(float(result.rgb.mean()))

    def test_cb_broken_jpeg_drops_frame_and_keeps_previous(self):
        """壊れ JPEG (imdecode → None) は drop、latest は前回維持。"""
        source = _make_ros2_source_no_init()
        # 1 回目: 正常 frame で latest 更新
        rgb_ok = np.full((4, 4, 3), 100, dtype=np.uint8)
        source._cb(_fake_compressed_image_msg(rgb_ok, 1, 0))
        prev = source.get()
        assert prev is not None

        # 2 回目: 壊れ JPEG (適当な bytes)、latest は前回維持
        stamp = types.SimpleNamespace(sec=2, nanosec=0)
        header = types.SimpleNamespace(stamp=stamp)
        broken_msg = types.SimpleNamespace(
            data=b"\x00\x01\x02not_a_jpeg", header=header
        )
        source._cb(broken_msg)  # 例外なく return

        after = source.get()
        assert after is not None
        assert after.t == prev.t  # 1 回目の t を維持

    def test_cb_broad_exception_does_not_propagate(self):
        """msg 構造 mismatch 等の例外を _cb 内で catch (rclpy executor thread 保護)。"""
        source = _make_ros2_source_no_init()
        # data 属性が無い msg → np.frombuffer の中で AttributeError が起きる
        bad_msg = types.SimpleNamespace()
        source._cb(bad_msg)  # 例外が伝播しないことを確認 (raise すれば test fail)
        # latest は None のまま (最初から)
        assert source.get() is None
