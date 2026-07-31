#!/usr/bin/env python3
"""Read-only preflight for the Orin lossless camera recorder."""

from __future__ import annotations

import argparse

from data.flip_table_data_augmentation.teleop.real.lossless_camera import (
    CAMERA_CONTROL_PORT,
    RecorderControlClient,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=CAMERA_CONTROL_PORT)
    parser.add_argument("--maximum-clock-uncertainty-ms", type=float, default=2.0)
    args = parser.parse_args()

    client = RecorderControlClient(args.host, port=args.port)
    status = client.status()
    if status.get("active"):
        raise RuntimeError(
            f"Orin camera recorder already has an active session: {status}"
        )
    if status.get("failed"):
        raise RuntimeError(f"Orin camera recorder is unhealthy: {status}")
    free_bytes = int(status.get("disk_free_bytes", 0))
    required_bytes = int(status.get("minimum_recording_free_bytes", 1))
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Orin camera recorder has insufficient free space: "
            f"free={free_bytes},required={required_bytes}"
        )
    mapping = client.synchronize()
    uncertainty_ms = mapping.uncertainty_ns / 1.0e6
    if uncertainty_ms > args.maximum_clock_uncertainty_ms:
        raise RuntimeError(
            "Desktop/Orin monotonic clock mapping is too uncertain: "
            f"{uncertainty_ms:.3f} ms"
        )
    print(
        "lossless-camera-recorder-ok "
        f"clock_uncertainty_ms={uncertainty_ms:.3f} "
        f"offset_ms={mapping.desktop_to_orin_offset_ns / 1.0e6:.3f} "
        f"queue_capacity={status.get('queue_capacity')} "
        f"disk_free_gib={free_bytes / 1024**3:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
