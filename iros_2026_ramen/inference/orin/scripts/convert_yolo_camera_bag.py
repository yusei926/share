#!/usr/bin/env python3
"""Orinで記録した3つのCompressedImage topicを4本のYOLO確認用MP4へ変換する。

headのpacked stereo streamはhead_left/head_rightへ分割し、左右D405 RGBと合わせて
4本を生成する。bag内のheader timestampを ``frames.csv`` に、各streamの実効fpsを
``manifest.json`` に保存する。これは録画後のoffline変換なので、実機camera topicの
受信を阻害しない。

Example (container内)::

    python3 /scripts/convert_yolo_camera_bag.py \
        /recordings/yolo_fourcam_20260719_150000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


HEAD_TOPIC = "/head/camera/color/image_raw/compressed"
WRIST_LEFT_TOPIC = "/wrist_left/camera/color/image_raw/compressed"
WRIST_RIGHT_TOPIC = "/wrist_right/camera/color/image_raw/compressed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="rosbag2 directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="MP4 output directory (default: <bag>/mp4)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="MP4 timebase FPS (default: 30)")
    args = parser.parse_args()
    if not args.bag.is_dir():
        parser.error(f"bag directory not found: {args.bag}")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    return args


def _source_t_ns(msg: Any) -> int:
    stamp = msg.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _effective_fps(info: dict[str, Any]) -> float | None:
    count = info["frames_written"]
    first_t = info["first_source_t_ns"]
    last_t = info["last_source_t_ns"]
    if count < 2 or last_t <= first_t:
        return None
    return (count - 1) * 1_000_000_000 / (last_t - first_t)


def main() -> int:
    args = _parse_args()
    out_dir = args.out_dir or args.bag / "mp4"
    if out_dir.exists():
        raise FileExistsError(f"output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    import cv2
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from sensor_msgs.msg import CompressedImage

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    writers: dict[str, Any] = {}
    stream_info: dict[str, dict[str, Any]] = {}
    csv_file = (out_dir / "frames.csv").open("w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=("stream", "output_frame_index", "source_t_ns"),
    )
    csv_writer.writeheader()

    def write_frame(name: str, frame: Any, source_t: int) -> None:
        if name not in writers:
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_dir / f"{name}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"failed to open VideoWriter for {name}")
            writers[name] = writer
            stream_info[name] = {
                "video": f"{name}.mp4",
                "shape_bgr": [height, width, int(frame.shape[2])],
                "frames_written": 0,
                "first_source_t_ns": source_t,
                "last_source_t_ns": source_t,
                "effective_source_fps": None,
            }
        info = stream_info[name]
        writers[name].write(frame)
        index = info["frames_written"]
        info["frames_written"] += 1
        info["last_source_t_ns"] = source_t
        csv_writer.writerow(
            {"stream": name, "output_frame_index": index, "source_t_ns": source_t}
        )

    try:
        while reader.has_next():
            topic, data, _bag_timestamp = reader.read_next()
            if topic not in {HEAD_TOPIC, WRIST_LEFT_TOPIC, WRIST_RIGHT_TOPIC}:
                continue
            msg = deserialize_message(data, CompressedImage)
            image = cv2.imdecode(
                np.frombuffer(bytes(msg.data), np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise RuntimeError(f"failed to decode CompressedImage from {topic}")
            t_ns = _source_t_ns(msg)
            if topic == HEAD_TOPIC:
                width = image.shape[1]
                if width < 2 or width % 2:
                    raise RuntimeError(f"invalid packed head width: {width}")
                split = width // 2
                write_frame("head_left", image[:, :split], t_ns)
                write_frame("head_right", image[:, split:], t_ns)
            elif topic == WRIST_LEFT_TOPIC:
                write_frame("wrist_left", image, t_ns)
            else:
                write_frame("wrist_right", image, t_ns)
    finally:
        csv_file.close()
        for writer in writers.values():
            writer.release()

    for info in stream_info.values():
        info["effective_source_fps"] = _effective_fps(info)
    manifest = {
        "format": "yolo-four-camera-mp4/v1",
        "source_bag": str(args.bag),
        "mp4_timebase_fps": args.fps,
        "codec": "mp4v",
        "streams": stream_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, info in stream_info.items():
        print(
            f"[done] {name}: frames={info['frames_written']} "
            f"effective_source_fps={info['effective_source_fps']}"
        )
    print(f"[done] output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
