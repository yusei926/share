"""Pinned TeleImager client adapter for physical-camera consumers.

The upstream ``TeleImage`` API is intentionally contained here.  Revision
7dc9aa1 returns an object with explicit ``fps``, ``jpg`` and ``bgr`` fields;
older tuple-unpacking call sites silently used the wrong contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ReceivedTeleImage:
    bgr: np.ndarray | None
    jpg: bytes | None
    fps: float
    received_monotonic_ns: int


@dataclass(frozen=True)
class LatestCameraSample:
    """One unique JPEG transition observed from an upstream latest-value stream."""

    role: str
    jpg: bytes
    source_fps: float
    jpeg_generation: int
    first_observed_monotonic_ns: int
    transition_hz: float

    def age_ms(self, now_ns: int) -> float:
        return max(0.0, (now_ns - self.first_observed_monotonic_ns) / 1.0e6)


class LatestCameraTracker:
    """Retain a short history of unique latest-value JPEG transitions.

    The official subscriber intentionally exposes only its newest value.
    Polling it at the same nominal 30 Hz as the camera aliases two clocks and
    periodically skips a source frame.  The real backend polls at 120 Hz and
    keeps this ring for diagnostics and one-to-one bundle matching.
    """

    def __init__(
        self,
        role: str,
        getter: Callable[[], Any],
        *,
        history_size: int = 16,
    ) -> None:
        if not role:
            raise ValueError("camera role must be non-empty")
        if history_size < 8:
            raise ValueError("camera transition history must hold at least 8 frames")
        self.role = role
        self._getter = getter
        self._sample: LatestCameraSample | None = None
        self._history: deque[LatestCameraSample] = deque(maxlen=history_size)
        self._transition_times_ns: deque[int] = deque(maxlen=301)

    @property
    def sample(self) -> LatestCameraSample | None:
        return self._sample

    @property
    def history(self) -> tuple[LatestCameraSample, ...]:
        return tuple(self._history)

    def samples_after(self, jpeg_generation: int) -> tuple[LatestCameraSample, ...]:
        return tuple(
            sample
            for sample in self._history
            if sample.jpeg_generation > jpeg_generation
        )

    def poll(self) -> tuple[LatestCameraSample | None, bool]:
        # The real backend intentionally requests JPEG-only TeleImage values.
        # Do not touch the upstream ``.bgr`` property in this path: the pinned
        # implementation warns on every access when decoding is disabled.
        received = receive_teleimage(self._getter, include_bgr=False)
        jpg = received.jpg
        if jpg is None:
            return self._sample, False
        if self._sample is not None and jpg == self._sample.jpg:
            return self._sample, False
        now_ns = received.received_monotonic_ns
        self._transition_times_ns.append(now_ns)
        transition_hz = 0.0
        if len(self._transition_times_ns) >= 2:
            elapsed_s = (
                self._transition_times_ns[-1] - self._transition_times_ns[0]
            ) / 1.0e9
            if elapsed_s > 0.0:
                transition_hz = (len(self._transition_times_ns) - 1) / elapsed_s
        generation = 1 if self._sample is None else self._sample.jpeg_generation + 1
        self._sample = LatestCameraSample(
            role=self.role,
            jpg=jpg,
            source_fps=received.fps,
            jpeg_generation=generation,
            first_observed_monotonic_ns=now_ns,
            transition_hz=transition_hz,
        )
        self._history.append(self._sample)
        return self._sample, True


def create_image_client(host: str, *, request_bgr: bool = False) -> Any:
    """Create the pinned client.

    The physical backend consumes one coherent upstream artifact (JPEG) and
    decodes only the stereo head locally.  This avoids pairing an asynchronous
    BGR decoder output with a newer JPEG. Read-only camera checkers may opt in
    to upstream BGR decoding explicitly.
    """

    from ..upstream_compat import install_logging_mp_compat

    install_logging_mp_compat()
    from teleimager.image_client import ImageClient

    return ImageClient(host=host, request_bgr=request_bgr)


def receive_teleimage(
    getter: Callable[[], Any],
    *,
    include_bgr: bool = True,
) -> ReceivedTeleImage:
    """Read one pinned ``TeleImage`` without relying on iterable ordering."""

    value = getter()
    received_ns = time.monotonic_ns()
    required_fields = ("bgr", "jpg", "fps") if include_bgr else ("jpg", "fps")
    missing = [name for name in required_fields if not hasattr(value, name)]
    if missing:
        raise RuntimeError(
            "unsupported TeleImager API; expected TeleImage fields "
            f"{'/'.join(required_fields)}, missing={missing}"
        )
    bgr = value.bgr if include_bgr else None
    if bgr is not None:
        bgr = np.asarray(bgr)
    jpg = value.jpg
    if jpg is not None and not isinstance(jpg, bytes):
        jpg = bytes(jpg)
    return ReceivedTeleImage(
        bgr=bgr,
        jpg=jpg,
        fps=float(value.fps),
        received_monotonic_ns=received_ns,
    )
