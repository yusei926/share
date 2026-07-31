"""Read-only network diagnostics for the physical TeleImager route."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Mapping


_INTERFACE_COUNTERS = (
    "rx_bytes",
    "rx_packets",
    "rx_dropped",
    "rx_errors",
    "rx_missed_errors",
    "tx_bytes",
    "tx_packets",
    "tx_dropped",
    "tx_errors",
)


def route_interface(destination: str) -> str | None:
    try:
        result = subprocess.run(
            ["ip", "-json", "route", "get", destination],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        rows = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    device = rows[0].get("dev")
    return device if isinstance(device, str) and device else None


def _tcp_retransmitted_segments() -> int | None:
    try:
        lines = Path("/proc/net/snmp").read_text(encoding="ascii").splitlines()
    except OSError:
        return None
    for index in range(len(lines) - 1):
        header = lines[index].split()
        values = lines[index + 1].split()
        if (
            header
            and values
            and header[0] == "Tcp:"
            and values[0] == "Tcp:"
            and "RetransSegs" in header
            and len(header) == len(values)
        ):
            try:
                return int(values[header.index("RetransSegs")])
            except ValueError:
                return None
    return None


def network_snapshot(interface: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if interface:
        root = Path("/sys/class/net") / interface / "statistics"
        for name in _INTERFACE_COUNTERS:
            try:
                result[name] = int((root / name).read_text(encoding="ascii").strip())
            except (OSError, ValueError):
                continue
    retransmitted = _tcp_retransmitted_segments()
    if retransmitted is not None:
        result["tcp_retransmitted_segments"] = retransmitted
    return result


def counter_delta(
    baseline: Mapping[str, int], current: Mapping[str, int]
) -> dict[str, int]:
    return {
        key: max(0, int(value) - int(baseline.get(key, value)))
        for key, value in current.items()
    }
