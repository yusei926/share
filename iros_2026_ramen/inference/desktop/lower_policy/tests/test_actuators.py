"""MockWalkActuator + G1SDKWalkActuator の module-level tests。

G1SDKWalkActuator は SDK を lazy import する契約なので、SDK が入っていない
main env でも import できることを検証する。__init__ 呼び出しは main env で
できない (SDK が無いので) が、class 定義の import は成立する必要がある。
"""

from __future__ import annotations

from inference.desktop.lower_policy.actuators.mock import MockWalkActuator


def test_empty_events_initially():
    actuator = MockWalkActuator()
    assert actuator.events == []


def test_set_velocity_records_event():
    actuator = MockWalkActuator()
    actuator.set_velocity(0.1, 0.0, 0.05)
    assert actuator.events == [("set_velocity", 0.1, 0.0, 0.05)]


def test_stand_up_records_event():
    actuator = MockWalkActuator()
    actuator.stand_up()
    assert actuator.events == [("stand_up",)]


def test_damp_records_event():
    actuator = MockWalkActuator()
    actuator.damp()
    assert actuator.events == [("damp",)]


def test_g1_sdk_actuator_module_importable_without_sdk():
    """SDK 無しの main env でも G1SDKWalkActuator を import できる (lazy 契約)。
    __init__ を呼ばない限り unitree_sdk2py に触れない。
    """
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    assert G1SDKWalkActuator is not None
    # top-level に SDK 依存を持ち込んでいないことの sanity check
    import inference.desktop.lower_policy.actuators.g1_sdk as mod

    assert not hasattr(mod, "ChannelFactoryInitialize")
    assert not hasattr(mod, "LocoClient")


def test_g1_sdk_set_velocity_checks_continuous_move_result(capsys):
    """duration 省略時は持続 SetVelocity の成功コードを検査する。"""
    from unittest.mock import MagicMock

    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    client = MagicMock()
    client.SetVelocity.return_value = 0
    actuator = G1SDKWalkActuator(client=client)
    actuator.set_velocity(0.05, 0.0, 0.0)
    client.SetVelocity.assert_called_once_with(0.05, 0.0, 0.0, duration=864000.0)
    assert "SetVelocity(duration=864000s): code=0" in capsys.readouterr().err


def test_g1_sdk_get_loco_status_reads_api_and_fsm_without_writes():
    """preflightはread-only APIだけを使い、速度・姿勢命令を発行しない。"""
    from unittest.mock import MagicMock

    from inference.desktop.lower_policy.actuators.g1_sdk import (
        G1LocoStatus,
        G1SDKWalkActuator,
    )

    client = MagicMock()
    client.GetApiVersion.return_value = "1.0.0.0"
    client.GetServerApiVersion.return_value = (0, "1.0.0.0")
    client.GetFsmId.return_value = (0, 501)
    client._Call.return_value = (0, '{"data": 0}')
    actuator = G1SDKWalkActuator(client=client)

    assert actuator.get_loco_status() == G1LocoStatus(
        client_api_version="1.0.0.0",
        server_api_version="1.0.0.0",
        fsm_id=501,
        fsm_mode=0,
    )
    client._Call.assert_called_once_with(7002, "{}")
    client.SetVelocity.assert_not_called()
    client.SetStandHeight.assert_not_called()
    client.SetFsmId.assert_not_called()


def test_g1_sdk_set_velocity_passes_finite_duration(capsys):
    """実機診断の有限時間は SDK RPC にそのまま渡す。"""
    from unittest.mock import MagicMock

    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    client = MagicMock()
    client.SetVelocity.return_value = 0
    actuator = G1SDKWalkActuator(client=client)
    actuator.set_velocity(0.03, 0.0, 0.0, duration=2.0)
    client.SetVelocity.assert_called_once_with(0.03, 0.0, 0.0, duration=2.0)
    assert "SetVelocity(duration=2s): code=0" in capsys.readouterr().err


def test_g1_sdk_stand_up_checks_high_stand_result(capsys):
    """stand_up は冪等なhigh-stand setpointの成功コードを検査する。"""
    from unittest.mock import MagicMock

    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    client = MagicMock()
    client.SetStandHeight.return_value = 0
    actuator = G1SDKWalkActuator(client=client)
    actuator.stand_up()
    client.SetStandHeight.assert_called_once_with((1 << 32) - 1)
    client.Squat2StandUp.assert_not_called()
    assert "SetStandHeight(high): code=0" in capsys.readouterr().err


def test_g1_sdk_damp_checks_result(capsys):
    """damp はdamping FSM requestの成功コードを検査する。"""
    from unittest.mock import MagicMock

    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator

    client = MagicMock()
    client.SetFsmId.return_value = 0
    actuator = G1SDKWalkActuator(client=client)
    actuator.damp()
    client.SetFsmId.assert_called_once_with(1)
    assert "SetFsmId(damp): code=0" in capsys.readouterr().err


def test_g1_sdk_rejected_command_fails_closed():
    """G1が拒否した速度指令をwalk成功として続行しない。"""
    from unittest.mock import MagicMock

    import pytest

    from inference.desktop.lower_policy.actuators.g1_sdk import (
        G1SDKCommandError,
        G1SDKWalkActuator,
    )

    client = MagicMock()
    client.SetVelocity.return_value = 3203
    actuator = G1SDKWalkActuator(client=client)

    with pytest.raises(G1SDKCommandError, match="code=3203"):
        actuator.set_velocity(0.03, 0.0, 0.0)


def test_events_accumulate_in_order():
    actuator = MockWalkActuator()
    actuator.stand_up()
    actuator.set_velocity(0.1, 0.0, 0.0)
    actuator.set_velocity(0.0, 0.0, 0.0)
    actuator.damp()
    assert actuator.events == [
        ("stand_up",),
        ("set_velocity", 0.1, 0.0, 0.0),
        ("set_velocity", 0.0, 0.0, 0.0),
        ("damp",),
    ]
