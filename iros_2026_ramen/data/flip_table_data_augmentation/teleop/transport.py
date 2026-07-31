"""Bounded length-prefixed JSON transport with native binary attachments.

Camera JPEGs are already compressed binary.  Base64 encoding them into JSON
increases every stereo packet by roughly one third and adds an unnecessary
copy/encode/decode on both ends.  That was especially costly for local Isaac
Sim AVP teleoperation, where the simulator and operator are connected through
localhost.  The wire format therefore keeps a small JSON metadata document and
appends byte attachments verbatim in the same bounded frame.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
from typing import Any, Mapping


MAX_MESSAGE_BYTES = 32 * 1024 * 1024
_BINARY_KEY = "__team_ramen_binary_attachment__"


def _encode(value: Any, attachments: list[bytes]) -> Any:
    if isinstance(value, bytes):
        index = len(attachments)
        attachments.append(value)
        return {_BINARY_KEY: [index, len(value)]}
    if isinstance(value, Mapping):
        return {str(key): _encode(item, attachments) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item, attachments) for item in value]
    return value


def _decode(value: Any, attachments: tuple[bytes, ...]) -> Any:
    if isinstance(value, dict):
        if set(value) == {_BINARY_KEY}:
            descriptor = value[_BINARY_KEY]
            try:
                index, size = descriptor
                if (
                    not isinstance(index, int)
                    or not isinstance(size, int)
                    or index < 0
                    or size < 0
                    or index >= len(attachments)
                    or len(attachments[index]) != size
                ):
                    raise ValueError
                return attachments[index]
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid binary attachment descriptor") from exc
        return {str(key): _decode(item, attachments) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, attachments) for item in value]
    return value


def encode_message(value: Mapping[str, Any]) -> bytes:
    attachments: list[bytes] = []
    metadata = json.dumps(
        _encode(value, attachments), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(metadata) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message metadata size {len(metadata)} exceeds the transport limit")
    payload = struct.pack("!I", len(metadata)) + metadata + b"".join(attachments)
    if not payload or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message size {len(payload)} is outside the transport limit")
    return struct.pack("!I", len(payload)) + payload


def decode_message(payload: bytes) -> dict[str, Any]:
    if len(payload) < 4 or len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("message payload is empty or oversized")
    metadata_size = struct.unpack("!I", payload[:4])[0]
    if metadata_size <= 0 or metadata_size > len(payload) - 4:
        raise ValueError("invalid transport metadata size")
    value = json.loads(payload[4 : 4 + metadata_size].decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("transport message must be a JSON object")
    # Attachment descriptors include their exact sizes, so reconstructing the
    # ordered binary tail is deterministic and rejects truncation/trailing
    # bytes rather than silently accepting a corrupt camera frame.
    tail = payload[4 + metadata_size :]
    descriptors: list[tuple[int, int]] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            if set(item) == {_BINARY_KEY}:
                descriptor = item[_BINARY_KEY]
                if (
                    not isinstance(descriptor, list)
                    or len(descriptor) != 2
                    or not all(isinstance(value, int) for value in descriptor)
                ):
                    raise ValueError("invalid binary attachment descriptor")
                descriptors.append((descriptor[0], descriptor[1]))
                return
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    if sorted(index for index, _size in descriptors) != list(range(len(descriptors))):
        raise ValueError("binary attachment indices must be contiguous")
    attachments: list[bytes] = []
    offset = 0
    for _index, size in sorted(descriptors):
        if size < 0 or offset + size > len(tail):
            raise ValueError("binary attachment exceeds transport payload")
        attachments.append(tail[offset : offset + size])
        offset += size
    if offset != len(tail):
        raise ValueError("transport payload has trailing binary bytes")
    return _decode(value, tuple(attachments))


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
