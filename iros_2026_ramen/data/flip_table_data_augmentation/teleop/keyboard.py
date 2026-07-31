"""Terminal and dedicated USB foot-pedal controls for AVP teleoperation.

The foot pedal is deliberately consumed as a *raw* Linux input device.  It is
not remapped into global keyboard events: a udev rule makes the PCsensor device
invisible to the desktop outside a teleop session, and :class:`FootPedalReader`
uses an exclusive grab only while a session is running.  This avoids a pedal
press activating an unrelated desktop shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import select
import sys
import termios
import threading
import time
import tty
from typing import Any, Mapping


VALID_KEYS = frozenset("rsdq")
FOOT_PEDAL_ACTIONS = ("r", "s", "q")
FOOT_PEDAL_SCHEMA_VERSION = "team_ramen_avp_footswitch/v1"
DEFAULT_FOOT_PEDAL_DEVICE = Path(
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
)
DEFAULT_FOOT_PEDAL_CONFIG = (
    Path.home() / ".config" / "iros_2026_ramen" / "avp_footswitch.json"
)

# Linux ``struct input_event`` on the supported 64-bit Ubuntu hosts:
# ``struct timeval`` + type:u16 + code:u16 + value:s32.
_INPUT_EVENT = __import__("struct").Struct("@llHHi")
_EV_KEY = 0x01
_EVIOCGRAB = 0x40044590


@dataclass(frozen=True)
class FootPedalBinding:
    """Physical PCsensor key codes assigned to the three safe AVP controls."""

    device: Path
    action_to_key_code: Mapping[str, int]

    def __post_init__(self) -> None:
        actions = dict(self.action_to_key_code)
        if set(actions) != set(FOOT_PEDAL_ACTIONS):
            raise ValueError(
                "foot-pedal binding must define exactly r, s, q actions"
            )
        codes = tuple(actions.values())
        if (
            any(not isinstance(code, int) or code <= 0 for code in codes)
            or len(set(codes)) != len(codes)
        ):
            raise ValueError(
                "foot-pedal key codes must be distinct positive Linux key codes"
            )

    @property
    def key_code_to_action(self) -> dict[int, str]:
        return {int(code): action for action, code in self.action_to_key_code.items()}


def load_foot_pedal_binding(path: str | Path) -> FootPedalBinding:
    """Load a fail-closed physical pedal mapping written by the calibrator."""

    config_path = Path(path).expanduser()
    try:
        value: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "foot-pedal mapping is missing; run "
            "python -m data.flip_table_data_augmentation.teleop.pedal_setup"
        ) from exc
    if not isinstance(value, dict) or value.get("schema_version") != FOOT_PEDAL_SCHEMA_VERSION:
        raise ValueError("unsupported foot-pedal mapping file")
    device = value.get("device")
    action_to_key_code = value.get("action_to_key_code")
    if not isinstance(device, str) or not device:
        raise ValueError("foot-pedal mapping has no device path")
    if not isinstance(action_to_key_code, dict):
        raise ValueError("foot-pedal mapping has no action_to_key_code object")
    return FootPedalBinding(
        device=Path(device),
        action_to_key_code={str(action): code for action, code in action_to_key_code.items()},
    )


def write_foot_pedal_binding(path: str | Path, binding: FootPedalBinding) -> None:
    """Persist a calibrated mapping outside the repository with restrictive mode."""

    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FOOT_PEDAL_SCHEMA_VERSION,
        "device": str(binding.device),
        "action_to_key_code": dict(binding.action_to_key_code),
    }
    descriptor = os.open(
        config_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    # ``fdopen`` takes ownership of the descriptor and closes it on every
    # exit path, including a write failure.
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(config_path, 0o600)


def _decode_input_events(payload: bytes) -> tuple[bytes, tuple[tuple[int, int, int], ...]]:
    """Return complete ``(type, code, value)`` events and an incomplete tail."""

    complete_bytes = len(payload) - (len(payload) % _INPUT_EVENT.size)
    events = tuple(
        _INPUT_EVENT.unpack_from(payload, offset)[2:]
        for offset in range(0, complete_bytes, _INPUT_EVENT.size)
    )
    return payload[complete_bytes:], events


class FootPedalReader:
    """Read calibrated PCsensor pedal presses only while teleoperation runs.

    Opening the device performs ``EVIOCGRAB``.  The kernel therefore does not
    deliver the pedal's factory keyboard shortcuts to another process during a
    teleop session.  Releases and key-repeat values are ignored: one physical
    press yields at most one operator control event.
    """

    def __init__(self, binding: FootPedalBinding) -> None:
        self._binding = binding
        self._fd = os.open(
            str(binding.device.expanduser()),
            os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC,
        )
        self._buffer = b""
        self._pending_key_codes: deque[int] = deque()
        self._closed = False
        try:
            fcntl.ioctl(self._fd, _EVIOCGRAB, 1)
        except OSError as exc:
            os.close(self._fd)
            self._closed = True
            raise RuntimeError(
                f"cannot exclusively read foot pedal {binding.device}: {exc}. "
                "Install the PCsensor udev rule and reconnect the pedal."
            ) from exc

    @property
    def device(self) -> Path:
        return self._binding.device

    def _read_events(self) -> tuple[tuple[int, int, int], ...]:
        while True:
            try:
                chunk = os.read(self._fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                raise RuntimeError("foot pedal input device disconnected")
            self._buffer += chunk
        self._buffer, events = _decode_input_events(self._buffer)
        return events

    def poll_key_code(self) -> int | None:
        """Return one press code, ignoring releases/repeats and unrelated events."""

        if self._pending_key_codes:
            return self._pending_key_codes.popleft()
        for event_type, code, value in self._read_events():
            if event_type == _EV_KEY and value == 1:
                self._pending_key_codes.append(code)
        return (
            self._pending_key_codes.popleft()
            if self._pending_key_codes
            else None
        )

    def wait_for_key_code(self, timeout_s: float) -> int:
        """Wait for one non-repeating key press; used only by offline setup."""

        deadline = time.monotonic() + timeout_s
        while True:
            code = self.poll_key_code()
            if code is not None:
                return code
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for a foot-pedal press")
            select.select((self._fd,), (), (), min(remaining, 0.25))

    def poll(self) -> str | None:
        code = self.poll_key_code()
        if code is None:
            return None
        return self._binding.key_code_to_action.get(code)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.ioctl(self._fd, _EVIOCGRAB, 0)
        finally:
            os.close(self._fd)

    def __enter__(self) -> "FootPedalReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class KeyReader:
    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("teleoperation requires an interactive TTY")
        self._queue: SimpleQueue[str] = SimpleQueue()
        self._stop = threading.Event()
        self._fd = sys.stdin.fileno()
        self._original = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(
            target=self._run,
            name="teleop-keyboard",
            daemon=False,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select((self._fd,), (), (), 0.1)
            if not readable:
                continue
            try:
                payload = os.read(self._fd, 1)
            except OSError:
                if self._stop.is_set():
                    return
                raise
            if not payload:
                return
            value = payload.decode(errors="ignore").lower()
            if value in VALID_KEYS:
                self._queue.put(value)

    def poll(self) -> str | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original)
        finally:
            if self._thread.is_alive():
                raise RuntimeError("teleop keyboard reader did not stop")

    def __enter__(self) -> "KeyReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
