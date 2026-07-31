"""run_g1_walk_to_table.py の CLI helper (argparse type) の unit tests。

script 全体の main() は Mock 経路で smoke してあるが、`--vx` の validation を
明示的に契約テストしておく (実機で危険な値が通ることを防ぐガード)。
"""

from __future__ import annotations

import argparse

import pytest

from inference.desktop.lower_policy.scripts.run_g1_walk_to_table import (
    _DURATION_MAX,
    _VX_MAX,
    _bounded_duration,
    _clamped_vx,
)
from inference.desktop.lower_policy.skills.move_to_table import DEFAULT_WALK_VX


def test_clamped_vx_accepts_default():
    assert _clamped_vx(str(DEFAULT_WALK_VX)) == DEFAULT_WALK_VX


def test_clamped_vx_accepts_lower_boundary():
    assert _clamped_vx("0") == 0.0


def test_clamped_vx_accepts_upper_boundary():
    assert _clamped_vx(str(_VX_MAX)) == _VX_MAX


def test_clamped_vx_rejects_too_high():
    with pytest.raises(argparse.ArgumentTypeError, match="must be in"):
        _clamped_vx("5.0")


def test_clamped_vx_rejects_negative():
    with pytest.raises(argparse.ArgumentTypeError, match="must be in"):
        _clamped_vx("-0.1")


def test_bounded_duration_accepts_diagnostic_default():
    assert _bounded_duration("1") == 1.0


@pytest.mark.parametrize("value", ["0", "0.099", str(_DURATION_MAX + 0.1)])
def test_bounded_duration_rejects_unsafe_value(value):
    with pytest.raises(argparse.ArgumentTypeError, match="duration must be"):
        _bounded_duration(value)
