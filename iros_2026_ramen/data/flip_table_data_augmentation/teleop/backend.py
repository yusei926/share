"""Backend-neutral interface used by the shared operator state machine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import ArmHandTarget, TeleopObservation


class TransientObservationError(RuntimeError):
    """A recoverable sensor outage which pauses rather than kills a session."""


class TeleopBackend(ABC):
    @abstractmethod
    def observe(self, timeout_s: float) -> TeleopObservation:
        """Return the newest synchronized camera/proprioception sample."""

    @abstractmethod
    def apply(self, target: ArmHandTarget) -> None:
        """Submit one backend-neutral arm/hand target."""

    @abstractmethod
    def close(self) -> None:
        """Hold safely and release runtime resources."""
