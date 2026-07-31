from __future__ import annotations

import json
from collections import deque
import os
import pty
import sys
import time
from pathlib import Path

import pytest

from data.flip_table_data_augmentation.teleop.keyboard import (
    FOOT_PEDAL_SCHEMA_VERSION,
    FootPedalBinding,
    FootPedalReader,
    KeyReader,
    _INPUT_EVENT,
    _decode_input_events,
    load_foot_pedal_binding,
    write_foot_pedal_binding,
)


def test_foot_pedal_binding_requires_distinct_r_s_q_codes() -> None:
    binding = FootPedalBinding(
        device=Path("/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"),
        action_to_key_code={"r": 30, "s": 31, "q": 32},
    )
    assert binding.key_code_to_action == {30: "r", 31: "s", 32: "q"}

    with pytest.raises(ValueError, match="distinct"):
        FootPedalBinding(
            device=Path("/dev/input/event25"),
            action_to_key_code={"r": 30, "s": 30, "q": 32},
        )


def test_foot_pedal_binding_round_trip_is_restrictive(tmp_path) -> None:
    path = tmp_path / "pedal.json"
    binding = FootPedalBinding(
        device=tmp_path / "event-kbd",
        action_to_key_code={"r": 40, "s": 41, "q": 42},
    )
    write_foot_pedal_binding(path, binding)

    assert path.stat().st_mode & 0o777 == 0o600
    assert load_foot_pedal_binding(path) == binding
    assert json.loads(path.read_text())["schema_version"] == FOOT_PEDAL_SCHEMA_VERSION


def test_foot_pedal_binding_rejects_unknown_or_incomplete_actions(tmp_path) -> None:
    path = tmp_path / "pedal.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": FOOT_PEDAL_SCHEMA_VERSION,
                "device": "/dev/input/event25",
                "action_to_key_code": {"r": 30, "s": 31, "d": 32},
            }
        )
    )
    with pytest.raises(ValueError, match="exactly r, s, q"):
        load_foot_pedal_binding(path)


def test_decode_input_events_keeps_partial_tail_and_event_values() -> None:
    key_press = _INPUT_EVENT.pack(1, 2, 1, 30, 1)
    key_release = _INPUT_EVENT.pack(1, 3, 1, 30, 0)
    partial = b"tail"

    tail, events = _decode_input_events(key_press + key_release + partial)

    assert events == ((1, 30, 1), (1, 30, 0))
    assert tail == partial


def test_foot_pedal_reader_queues_every_press_from_one_kernel_read() -> None:
    reader = FootPedalReader.__new__(FootPedalReader)
    reader._pending_key_codes = deque()
    reads = [((1, 30, 1), (1, 31, 1), (1, 30, 0)), ()]
    reader._read_events = lambda: reads.pop(0)

    assert reader.poll_key_code() == 30
    assert reader.poll_key_code() == 31
    assert reader.poll_key_code() is None


def test_keyboard_reader_wakes_and_joins_without_buffered_stdio_thread(
    monkeypatch,
) -> None:
    master_fd, slave_fd = pty.openpty()
    slave = os.fdopen(slave_fd, "r", encoding="utf-8", buffering=1)
    monkeypatch.setattr(sys, "stdin", slave)
    reader = KeyReader()
    try:
        os.write(master_fd, b"r")
        deadline = time.monotonic() + 1.0
        value = None
        while value is None and time.monotonic() < deadline:
            value = reader.poll()
            time.sleep(0.01)
        assert value == "r"
    finally:
        reader.close()
        os.close(master_fd)
        slave.close()
    assert reader._thread.is_alive() is False
