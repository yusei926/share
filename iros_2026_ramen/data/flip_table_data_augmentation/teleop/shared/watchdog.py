"""Backend-neutral command freshness states."""

from __future__ import annotations

from enum import Enum


class WatchdogState(str, Enum):
    ACTIVE = "active"
    HOLD = "hold"
    STOP = "stop"
