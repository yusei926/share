from __future__ import annotations

import pytest

from inference.desktop.xr.check_g1_regular_mode import validate_regular_mode_status


def test_startup_regular_check_requires_mode_zero() -> None:
    validate_regular_mode_status(
        fsm_id=501,
        fsm_mode=0,
        expected_fsm_id=501,
        allowed_fsm_modes=(0,),
    )
    with pytest.raises(RuntimeError, match=r"actual=\(501,1\)"):
        validate_regular_mode_status(
            fsm_id=501,
            fsm_mode=1,
            expected_fsm_id=501,
            allowed_fsm_modes=(0,),
        )


def test_arm_sdk_active_regular_check_accepts_only_modes_zero_or_one() -> None:
    for mode in (0, 1):
        validate_regular_mode_status(
            fsm_id=501,
            fsm_mode=mode,
            expected_fsm_id=501,
            allowed_fsm_modes=(0, 1),
        )
    for fsm_id, mode in ((500, 0), (501, 2), (1, 1)):
        with pytest.raises(RuntimeError):
            validate_regular_mode_status(
                fsm_id=fsm_id,
                fsm_mode=mode,
                expected_fsm_id=501,
                allowed_fsm_modes=(0, 1),
            )
