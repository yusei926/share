"""実G1 entrypointの停止処理に対する副作用なしの契約テスト。"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from inference.desktop.entrypoint import (
    HEAD_CAMERA_STEREO_VIEW,
    _positive_float,
    _safe_shutdown,
)


def test_safe_shutdown_stops_velocity_without_entering_damp():
    """arm_actuator 無し (`--no-arm-actuator` 相当 or 起動失敗ケース) の shutdown。"""
    actuator = MagicMock()
    log_sink = MagicMock()
    source = MagicMock()

    _safe_shutdown(actuator, log_sink, source=source)

    actuator.set_velocity.assert_called_once_with(
        0.0, 0.0, 0.0, duration=1.0
    )
    actuator.damp.assert_not_called()
    actuator.stand_up.assert_not_called()
    source.close.assert_called_once_with()
    log_sink.close.assert_called_once_with()


def test_safe_shutdown_stops_arm_actuator_before_walk_stop():
    """arm_actuator 有りの shutdown: arm.stop() を walk 停止より先に呼ぶ (Issue #75)。

    順序が逆だと walk 止めた後も arm publish 続いて firmware timeout → unpower
    (脱力) のリスク。arm を先に止めれば publish 途絶 → LowCmd_ から walking mode に
    切り替わり → walk 停止指令が生きる。
    """
    walk_actuator = MagicMock()
    arm_actuator = MagicMock()
    log_sink = MagicMock()
    call_order: list[str] = []
    arm_actuator.stop.side_effect = lambda: call_order.append("arm_stop")
    walk_actuator.set_velocity.side_effect = lambda *a, **k: call_order.append("walk_stop")

    _safe_shutdown(walk_actuator, log_sink, arm_actuator=arm_actuator)

    # 順序: arm_stop → walk_stop
    assert call_order == ["arm_stop", "walk_stop"]
    arm_actuator.stop.assert_called_once_with()
    walk_actuator.set_velocity.assert_called_once_with(
        0.0, 0.0, 0.0, duration=1.0
    )
    log_sink.close.assert_called_once_with()


def test_safe_shutdown_swallows_arm_stop_exception():
    """arm.stop() が raise しても walk 停止 / log close は続行 (二段構え保護)。"""
    walk_actuator = MagicMock()
    arm_actuator = MagicMock()
    arm_actuator.stop.side_effect = RuntimeError("simulated arm stop failure")
    log_sink = MagicMock()

    _safe_shutdown(walk_actuator, log_sink, arm_actuator=arm_actuator)

    walk_actuator.set_velocity.assert_called_once()
    log_sink.close.assert_called_once()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_rejects_unsafe_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_float(value)


def test_positive_float_accepts_valid_value():
    assert _positive_float("0.5") == 0.5


def test_entrypoint_is_fixed_to_left_head_camera():
    assert HEAD_CAMERA_STEREO_VIEW == "left"
