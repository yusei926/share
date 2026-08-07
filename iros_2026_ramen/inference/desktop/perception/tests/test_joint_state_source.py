"""JointStateSource + JointState_ IDL の unit test (Issue #75)。

JointStateSource の test は cyclonedds / SDK install 無しで動くよう、__init__ を
bypass して _cb を直接叩く形 (test_frame_source.py の Ros2FrameSource 部分と同 pattern)。

IDL 定義 test は cyclonedds 依存なので importorskip で main env では skip
(test_sensor_msgs_idl.py と同 pattern)。
"""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.desktop.perception.joint_state_source import (
    JointStateData,
    JointStateSource,
)


# =========================================================================
# JointStateData (frozen dataclass)
# =========================================================================
class TestJointStateData:
    def test_frozen(self):
        d = JointStateData(
            name=("j0",),
            position=np.zeros(1),
            velocity=np.zeros(1),
            effort=np.zeros(1),
            t=0,
        )
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            d.t = 1


# =========================================================================
# JointStateSource (rclpy / cyclonedds install 不要、_cb を直接呼ぶ形で test)
# =========================================================================
def _make_source_no_init() -> JointStateSource:
    """__init__ を bypass して internal state だけ初期化した instance を返す。

    cyclonedds / SDK install 無しの main env で test するため、SDK ChannelFactory
    経由の subscription 登録を skip し、_cb / get の behavior だけ検証する。
    """
    source = JointStateSource.__new__(JointStateSource)
    source._latest = None  # type: ignore[attr-defined]
    source._lock = threading.Lock()  # type: ignore[attr-defined]
    source._closed = False  # type: ignore[attr-defined]
    source._channel = MagicMock()  # type: ignore[attr-defined]
    return source


def _fake_joint_state_msg(
    name: list[str],
    position: list[float],
    velocity: list[float],
    effort: list[float],
    stamp_sec: int,
    stamp_nanosec: int,
) -> object:
    """cyclonedds が deserialize する JointState_ instance と同 shape の fake msg。"""
    stamp = types.SimpleNamespace(sec=stamp_sec, nanosec=stamp_nanosec)
    header = types.SimpleNamespace(stamp=stamp)
    return types.SimpleNamespace(
        header=header,
        name=name,
        position=position,
        velocity=velocity,
        effort=effort,
    )


class TestJointStateSource:
    def test_get_returns_none_before_any_callback(self):
        source = _make_source_no_init()
        assert source.get() is None

    def test_cb_updates_latest_snapshot(self):
        """_cb を直接呼んで latest snapshot が更新される + get() で取れる。"""
        source = _make_source_no_init()
        msg = _fake_joint_state_msg(
            name=["j0", "j1", "j2"],
            position=[0.1, 0.2, 0.3],
            velocity=[1.0, 2.0, 3.0],
            effort=[10.0, 20.0, 30.0],
            stamp_sec=2,
            stamp_nanosec=500_000_000,
        )
        source._cb(msg)

        result = source.get()
        assert result is not None
        assert result.name == ("j0", "j1", "j2")
        assert np.allclose(result.position, [0.1, 0.2, 0.3])
        assert np.allclose(result.velocity, [1.0, 2.0, 3.0])
        assert np.allclose(result.effort, [10.0, 20.0, 30.0])
        # t = 2 * 1e9 + 5e8
        assert result.t == 2_500_000_000
        # position は float64 ndarray に normalize されている
        assert result.position.dtype == np.float64

    def test_cb_overwrites_with_latest(self):
        """後続 callback で latest が上書きされる (latest-only policy)。"""
        source = _make_source_no_init()
        source._cb(
            _fake_joint_state_msg(
                name=["j0"], position=[1.0], velocity=[0.0], effort=[0.0],
                stamp_sec=1, stamp_nanosec=0,
            )
        )
        source._cb(
            _fake_joint_state_msg(
                name=["j0"], position=[9.0], velocity=[0.0], effort=[0.0],
                stamp_sec=3, stamp_nanosec=0,
            )
        )

        result = source.get()
        assert result is not None
        assert result.t == 3_000_000_000  # 後の msg
        assert np.allclose(result.position, [9.0])

    def test_cb_broad_exception_does_not_propagate(self):
        """msg 構造 mismatch 等の例外を _cb 内で catch (listener thread 保護)。"""
        source = _make_source_no_init()
        bad_msg = types.SimpleNamespace()  # name 属性が無い
        source._cb(bad_msg)  # 例外が伝播しないことを確認
        assert source.get() is None

    def test_cb_numpy_ndarray_normalization(self):
        """position/velocity/effort が list でも np.ndarray でも float64 に normalize される。"""
        source = _make_source_no_init()
        source._cb(
            _fake_joint_state_msg(
                name=["j0"],
                position=np.array([1.5], dtype=np.float32),  # float32 → float64 に変換
                velocity=[0.5],
                effort=[0.0],
                stamp_sec=0, stamp_nanosec=0,
            )
        )
        result = source.get()
        assert result is not None
        assert result.position.dtype == np.float64
        assert result.position[0] == 1.5

    def test_close_is_idempotent_and_releases_reader(self):
        """close は冪等で DataReader を解放し、以後 get() は None (#101、Ros2FrameSource と同型)。"""
        source = _make_source_no_init()
        channel = source._channel  # type: ignore[attr-defined]
        source._latest = JointStateData(  # type: ignore[attr-defined]
            name=("j0",), position=np.zeros(1), velocity=np.zeros(1),
            effort=np.zeros(1), t=1,
        )
        source.close()
        source.close()
        channel.CloseReader.assert_called_once_with()
        assert source.get() is None

    def test_cb_after_close_does_not_register(self):
        """close 後の callback は stale snapshot を再登録しない (native teardown 競合対策、#101)。"""
        source = _make_source_no_init()
        source.close()
        source._cb(
            _fake_joint_state_msg(
                name=["j0"], position=[1.0], velocity=[0.0], effort=[0.0],
                stamp_sec=1, stamp_nanosec=0,
            )
        )
        assert source.get() is None


# =========================================================================
# JointState_ IDL (cyclonedds 依存、runtime env でのみ動く)
# =========================================================================
class TestJointStateIdl:
    """IDL 定義 test は cyclonedds 依存。autouse fixture で cyclonedds 無しの default
    env では skip する。module レベル importorskip だと cyclonedds 非依存の
    TestJointStateSource まで巻き込んで skip されてしまうため class に閉じる (#101)。"""

    @pytest.fixture(autouse=True)
    def _require_cyclonedds(self):
        pytest.importorskip("cyclonedds")

    def test_typename_matches_ros2_canonical(self):
        """DDS typename が ROS2 canonical と一致 (rclpy publisher と wire 互換の要)。"""
        from inference.desktop.perception.sensor_msgs_idl import JointState_

        assert JointState_.__idl_typename__ == "sensor_msgs::msg::dds_::JointState_"

    def test_fields_match_ros2_msg_definition(self):
        """field 順序と名前が ROS2 sensor_msgs/JointState.msg と一致。

        ROS2 msg:
            std_msgs/Header header
            string[] name
            float64[] position
            float64[] velocity
            float64[] effort
        """
        from dataclasses import fields

        from inference.desktop.perception.sensor_msgs_idl import JointState_

        field_names = [f.name for f in fields(JointState_)]
        assert field_names == ["header", "name", "position", "velocity", "effort"]

    def test_header_type_is_sdk_header(self):
        """header field type が unitree_sdk2py の Header_ を参照している。"""
        from dataclasses import fields

        from inference.desktop.perception.sensor_msgs_idl import JointState_

        header_field = next(f for f in fields(JointState_) if f.name == "header")
        assert "std_msgs" in str(header_field.type)
        assert "Header_" in str(header_field.type)

    def test_idl_runtime_type_resolution(self):
        """CycloneDDS が nested Header_ を実際に解決できることを確認。"""
        from inference.desktop.perception.sensor_msgs_idl import JointState_

        JointState_.__idl__.populate()
