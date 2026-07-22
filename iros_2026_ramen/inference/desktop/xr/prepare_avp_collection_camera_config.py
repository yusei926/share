#!/usr/bin/env python3
"""Generate a role-explicit TeleImager config for stereo head + two D405s."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _serial(value: str) -> str:
    result = value.strip()
    if not result.isdigit():
        raise argparse.ArgumentTypeError("a RealSense serial must contain only digits")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-video-id", required=True, type=int)
    parser.add_argument("--left-d405-serial", required=True, type=_serial)
    parser.add_argument("--right-d405-serial", required=True, type=_serial)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _camera(*, port: int, shape: list[int], binocular: bool) -> dict[str, object]:
    return {
        "enable_zmq": True,
        "zmq_port": port,
        "enable_webrtc": False,
        "webrtc_port": port + 4500,
        "webrtc_codec": "h264",
        "image_shape": shape,
        "binocular": binocular,
        "fps": 30,
    }


def main() -> int:
    args = parse_args()
    if args.head_video_id < 0:
        raise ValueError("--head-video-id must be non-negative")
    if args.left_d405_serial == args.right_d405_serial:
        raise ValueError("left and right D405 serials must be distinct")

    head = _camera(port=55555, shape=[480, 1280], binocular=True)
    head.update(
        {
            "type": "opencv",
            "video_id": args.head_video_id,
            "serial_number": None,
            "physical_path": None,
        }
    )
    config: dict[str, object] = {"head_camera": head}
    for role, port, serial in (
        ("left_wrist_camera", 55556, args.left_d405_serial),
        ("right_wrist_camera", 55557, args.right_d405_serial),
    ):
        camera = _camera(port=port, shape=[480, 640], binocular=False)
        camera.update(
            {
                "type": "realsense",
                "video_id": None,
                "serial_number": serial,
                "physical_path": None,
            }
        )
        config[role] = camera

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
