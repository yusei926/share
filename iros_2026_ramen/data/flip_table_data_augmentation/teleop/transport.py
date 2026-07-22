"""Bounded length-prefixed JSON transport with explicit binary fields."""

from __future__ import annotations

import base64
import json
import socket
import struct
import threading
from typing import Any, Mapping


MAX_MESSAGE_BYTES = 32 * 1024 * 1024
_BINARY_KEY = "__team_ramen_bytes_b64__"


def _encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {_BINARY_KEY: base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {_BINARY_KEY}:
            try:
                return base64.b64decode(value[_BINARY_KEY], validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("invalid base64 binary field") from exc
        return {str(key): _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def encode_message(value: Mapping[str, Any]) -> bytes:
    payload = json.dumps(
        _encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message size {len(payload)} is outside the transport limit")
    return struct.pack("!I", len(payload)) + payload


def decode_message(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("message payload is empty or oversized")
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("transport message must be a JSON object")
    return _decode(value)


class FramedSocket:
    def __init__(self, connection: socket.socket) -> None:
        self.connection = connection
        self._send_lock = threading.Lock()

    def send(self, value: Mapping[str, Any]) -> None:
        packet = encode_message(value)
        with self._send_lock:
            self.connection.sendall(packet)

    def _read_exact(self, count: int) -> bytes:
        parts = bytearray()
        while len(parts) < count:
            chunk = self.connection.recv(count - len(parts))
            if not chunk:
                raise EOFError("teleoperation transport disconnected")
            parts.extend(chunk)
        return bytes(parts)

    def receive(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        previous = self.connection.gettimeout()
        self.connection.settimeout(timeout_s)
        try:
            size = struct.unpack("!I", self._read_exact(4))[0]
            if size <= 0 or size > MAX_MESSAGE_BYTES:
                raise ValueError(f"invalid teleoperation message size: {size}")
            return decode_message(self._read_exact(size))
        finally:
            self.connection.settimeout(previous)

    def close(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()
