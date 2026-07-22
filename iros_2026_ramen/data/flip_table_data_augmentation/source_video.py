"""Exact, single-pass decoding of episode frames from shared LeRobot video shards."""

from __future__ import annotations

from collections.abc import Iterable

import av
import numpy as np

from .source_dataset import VideoSlice


def source_frame_indices(frame_count: int, stride: int) -> tuple[int, ...]:
    if frame_count <= 0 or stride <= 0:
        raise ValueError("frame_count and stride must be positive")
    values = list(range(0, frame_count, stride))
    if values[-1] != frame_count - 1:
        values.append(frame_count - 1)
    return tuple(values)


def decode_video_slice_rgb(
    video: VideoSlice,
    frame_indices: Iterable[int],
    *,
    fps: int,
    frame_count: int,
) -> dict[int, np.ndarray]:
    """Decode selected RGB frames once and verify each PTS against episode metadata."""

    frames = tuple(frame_indices)
    if not frames or tuple(sorted(set(frames))) != frames:
        raise ValueError("frame_indices must be non-empty, sorted, and unique")
    if frames[0] < 0 or frames[-1] >= frame_count or fps <= 0:
        raise ValueError("frame_indices or fps are outside the episode contract")
    targets = tuple(video.timestamp_for_frame(index, fps, frame_count) for index in frames)
    tolerance_s = 0.5 / fps + 1.0e-6
    decoded: dict[int, np.ndarray] = {}
    with av.open(str(video.path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"video shard has no video stream: {video.path}")
        stream = container.streams.video[0]
        seek_s = max(0.0, targets[0] - 1.0)
        container.seek(int(seek_s * av.time_base), stream=None, backward=True, any_frame=False)
        target_index = 0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < targets[target_index] - tolerance_s:
                continue
            if abs(timestamp - targets[target_index]) > tolerance_s:
                raise ValueError(
                    f"{video.feature} frame {frames[target_index]} has no aligned video PTS: "
                    f"target={targets[target_index]:.9f}, decoded={timestamp:.9f}"
                )
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError(
                    f"{video.feature} decoded {rgb.shape}/{rgb.dtype}, expected uint8 (480,640,3)"
                )
            decoded[frames[target_index]] = rgb
            target_index += 1
            if target_index == len(frames):
                break
    missing = [index for index in frames if index not in decoded]
    if missing:
        raise ValueError(f"video shard ended before selected frames were decoded: {missing[:10]}")
    return decoded
