"""Timing helpers shared by the desktop recorder and simulator bridge."""

from __future__ import annotations

from collections.abc import Sequence


def bounded_delay_steps(
    capture_times_ns: Sequence[int],
    requested_steps: int,
    *,
    maximum_age_ns: int,
) -> int:
    """Limit a frame-count delay by its actual elapsed capture time.

    Rendering can run slightly below the nominal camera rate. Two frames are
    therefore not always two nominal camera periods. Returning fewer delayed
    frames preserves the configured real-time sensor-latency bound.
    """

    if requested_steps < 0:
        raise ValueError("requested delay steps must be non-negative")
    if maximum_age_ns < 0:
        raise ValueError("maximum camera age must be non-negative")
    if not capture_times_ns:
        raise ValueError("at least one capture timestamp is required")
    if any(timestamp <= 0 for timestamp in capture_times_ns):
        raise ValueError("capture timestamps must be positive")
    if any(
        later <= earlier
        for earlier, later in zip(capture_times_ns, capture_times_ns[1:])
    ):
        raise ValueError("capture timestamps must be strictly increasing")

    latest = capture_times_ns[-1]
    available_steps = min(requested_steps, len(capture_times_ns) - 1)
    for steps in range(available_steps, -1, -1):
        if latest - capture_times_ns[-1 - steps] <= maximum_age_ns:
            return steps
    return 0
