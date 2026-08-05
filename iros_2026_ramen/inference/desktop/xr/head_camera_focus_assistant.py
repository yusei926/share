#!/usr/bin/env python3
"""Read-only manual-focus assistant for the physical G1 head stereo camera.

The operator draws an arbitrary target bounding box in each eye.  Only pixels
inside those boxes are scored, so unrelated background texture cannot bias the
focus decision.  This process imports no Unitree SDK and sends no robot command.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.flip_table_data_augmentation.teleop.real.teleimager import (  # noqa: E402
    create_image_client,
    receive_teleimage,
)


EYE_WIDTH = 640
EYE_HEIGHT = 480
STEREO_WIDTH = EYE_WIDTH * 2
PLOT_HEIGHT = 180
TRACE_LENGTH = 240
MIN_ROI_PIXELS = 32


@dataclass(frozen=True)
class NormalizedRoi:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = np.asarray((self.x, self.y, self.width, self.height), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("ROI values must be finite")
        if self.width < 0.05 or self.height < 0.05:
            raise ValueError("ROI width and height must be at least 0.05")
        if self.x < 0.0 or self.y < 0.0:
            raise ValueError("ROI origin must be non-negative")
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError("ROI must fit inside one 640x480 eye image")

    def bounds(self, image: np.ndarray) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        x0 = int(round(self.x * width))
        y0 = int(round(self.y * height))
        x1 = int(round((self.x + self.width) * width))
        y1 = int(round((self.y + self.height) * height))
        return x0, y0, x1, y1

    def scaled(self, factor: float) -> "NormalizedRoi":
        center_x = self.x + self.width / 2
        center_y = self.y + self.height / 2
        width = float(np.clip(self.width * factor, 0.15, 0.90))
        height = float(np.clip(self.height * factor, 0.15, 0.90))
        center_x = float(np.clip(center_x, width / 2, 1 - width / 2))
        center_y = float(np.clip(center_y, height / 2, 1 - height / 2))
        return NormalizedRoi(center_x - width / 2, center_y - height / 2, width, height)


def roi_from_drag(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    minimum_pixels: int = MIN_ROI_PIXELS,
) -> NormalizedRoi:
    """Convert one-eye mouse coordinates into a clipped normalized box."""

    x0, x1 = sorted((int(start[0]), int(end[0])))
    y0, y1 = sorted((int(start[1]), int(end[1])))
    x0 = int(np.clip(x0, 0, EYE_WIDTH - 1))
    x1 = int(np.clip(x1, 0, EYE_WIDTH))
    y0 = int(np.clip(y0, 0, EYE_HEIGHT - 1))
    y1 = int(np.clip(y1, 0, EYE_HEIGHT))
    if x1 - x0 < minimum_pixels or y1 - y0 < minimum_pixels:
        raise ValueError(
            f"target bounding box must be at least {minimum_pixels}x{minimum_pixels} pixels"
        )
    return NormalizedRoi(
        x=x0 / EYE_WIDTH,
        y=y0 / EYE_HEIGHT,
        width=(x1 - x0) / EYE_WIDTH,
        height=(y1 - y0) / EYE_HEIGHT,
    )


@dataclass(frozen=True)
class FocusMeasurement:
    score: float
    laplacian_variance: float
    tenengrad: float
    brightness: float
    contrast: float
    clipped_fraction: float
    roi_gray_small: np.ndarray


@dataclass(frozen=True)
class FocusDisplayState:
    live_score: float
    best_score: float
    relative_percent: float
    motion: float
    stable: bool
    reliable: bool
    label: str
    new_best: bool


class FocusTracker:
    """Smooth one eye's score and retain only stable task-space peaks."""

    def __init__(
        self,
        *,
        smoothing_frames: int = 7,
        stable_frames: int = 3,
        motion_threshold: float = 4.0,
    ) -> None:
        if smoothing_frames < 1 or smoothing_frames % 2 == 0:
            raise ValueError("smoothing_frames must be a positive odd number")
        if stable_frames < 1:
            raise ValueError("stable_frames must be positive")
        self._scores: deque[float] = deque(maxlen=smoothing_frames)
        self.trace: deque[float] = deque(maxlen=TRACE_LENGTH)
        self._stable_frames_required = stable_frames
        self._motion_threshold = motion_threshold
        self._stable_run = 0
        self._previous_gray: np.ndarray | None = None
        self.best_score = 0.0

    def reset(self) -> None:
        self._scores.clear()
        self.trace.clear()
        self._stable_run = 0
        self._previous_gray = None
        self.best_score = 0.0

    def update(self, measurement: FocusMeasurement) -> FocusDisplayState:
        current_gray = measurement.roi_gray_small.astype(np.float32, copy=False)
        motion = float("inf")
        if self._previous_gray is not None and self._previous_gray.shape == current_gray.shape:
            motion = float(np.mean(np.abs(current_gray - self._previous_gray)))
        self._previous_gray = current_gray.copy()

        reliable = measurement.contrast >= 12.0 and measurement.clipped_fraction <= 0.35
        stable_now = np.isfinite(motion) and motion <= self._motion_threshold
        self._stable_run = self._stable_run + 1 if stable_now else 0
        stable = self._stable_run >= self._stable_frames_required

        self._scores.append(measurement.score)
        live_score = float(np.median(np.asarray(self._scores, dtype=np.float64)))
        self.trace.append(live_score)
        new_best = False
        if stable and reliable and live_score > self.best_score:
            self.best_score = live_score
            new_best = True
        relative = 0.0 if self.best_score <= 0.0 else 100.0 * live_score / self.best_score

        if measurement.contrast < 12.0:
            label = "LOW CONTRAST - PLACE TEXT/EDGES IN ROI"
        elif measurement.clipped_fraction > 0.35:
            label = "EXPOSURE CLIPPED - ADJUST LIGHTING"
        elif not stable:
            label = "HOLD CAMERA/TARGET STILL"
        elif relative >= 97.0:
            label = "PEAK ZONE"
        elif relative >= 88.0:
            label = "CLOSE TO BEST"
        else:
            label = "TURN FOCUS SLOWLY"
        return FocusDisplayState(
            live_score=live_score,
            best_score=self.best_score,
            relative_percent=relative,
            motion=motion,
            stable=stable,
            reliable=reliable,
            label=label,
            new_best=new_best,
        )


def parse_roi(value: str) -> NormalizedRoi:
    try:
        parts = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must have four comma-separated values")
    try:
        return NormalizedRoi(*parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def split_stereo_bgr(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frame.shape != (EYE_HEIGHT, STEREO_WIDTH, 3):
        raise ValueError(f"head stereo must be 1280x480 BGR, got {frame.shape!r}")
    return frame[:, :EYE_WIDTH], frame[:, EYE_WIDTH:]


def measure_focus(image_bgr: np.ndarray, roi: NormalizedRoi) -> FocusMeasurement:
    if image_bgr.shape != (EYE_HEIGHT, EYE_WIDTH, 3):
        raise ValueError(f"one eye image must be 640x480 BGR, got {image_bgr.shape!r}")
    x0, y0, x1, y1 = roi.bounds(image_bgr)
    gray = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if gray.size < 256:
        raise ValueError("target ROI is too small")
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian_variance = float(np.var(laplacian))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    # Laplacian variance is intentionally the primary score: while the robot
    # and target stay fixed, it peaks near best optical focus and is easy to
    # explain.  Tenengrad is displayed only as a secondary diagnostic.
    score = max(0.0, laplacian_variance)
    clipped = np.logical_or(gray <= 5, gray >= 250)
    small = cv2.resize(gray, (96, 64), interpolation=cv2.INTER_AREA)
    return FocusMeasurement(
        score=score,
        laplacian_variance=laplacian_variance,
        tenengrad=tenengrad,
        brightness=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        clipped_fraction=float(np.mean(clipped)),
        roi_gray_small=small,
    )


def _color_for_state(state: FocusDisplayState) -> tuple[int, int, int]:
    if not state.reliable or not state.stable:
        return (0, 180, 255)
    if state.relative_percent >= 97.0:
        return (60, 220, 60)
    if state.relative_percent >= 88.0:
        return (0, 220, 255)
    return (40, 80, 240)


def _draw_eye_overlay(
    canvas: np.ndarray,
    *,
    eye_name: str,
    eye_offset_x: int,
    roi: NormalizedRoi | None,
    measurement: FocusMeasurement | None,
    state: FocusDisplayState | None,
) -> None:
    panel_x = eye_offset_x + 10
    cv2.rectangle(canvas, (panel_x, 8), (eye_offset_x + 625, 112), (20, 20, 20), -1)
    if roi is None or measurement is None or state is None:
        cv2.putText(
            canvas,
            f"{eye_name}: DRAG A TARGET BOUNDING BOX",
            (panel_x + 8, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Only the selected pixels will be scored",
            (panel_x + 8, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        return

    x0, y0, x1, y1 = roi.bounds(canvas[:EYE_HEIGHT, eye_offset_x : eye_offset_x + EYE_WIDTH])
    x0 += eye_offset_x
    x1 += eye_offset_x
    color = _color_for_state(state)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 3)
    cv2.putText(
        canvas,
        f"{eye_name} TARGET ROI",
        (x0 + 8, max(24, y0 + 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{eye_name}: LIVE {state.live_score:8.1f}   BEST {state.best_score:8.1f}   {state.relative_percent:5.1f}%",
        (panel_x + 8, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        state.label,
        (panel_x + 8, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"brightness={measurement.brightness:.0f} contrast={measurement.contrast:.1f} "
        f"motion={state.motion if np.isfinite(state.motion) else 0.0:.1f}",
        (panel_x + 8, 93),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )


def _draw_traces(
    canvas: np.ndarray,
    trackers: dict[str, FocusTracker],
    *,
    source_fps: float,
    observed_hz: float,
) -> None:
    top = EYE_HEIGHT
    cv2.rectangle(canvas, (0, top), (STEREO_WIDTH, top + PLOT_HEIGHT), (18, 18, 18), -1)
    cv2.putText(
        canvas,
        "Drag a box around the SAME target in each eye. Sweep each lens, then return to its stable peak.",
        (18, top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "right-click clear eye | c clear all | r reset peak | [ ] resize | s save | q/ESC quit",
        (18, top + 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    plot_left, plot_right = 18, STEREO_WIDTH - 18
    plot_top, plot_bottom = top + 76, top + PLOT_HEIGHT - 12
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (80, 80, 80), 1)
    traces = {name: np.asarray(tracker.trace, dtype=np.float64) for name, tracker in trackers.items()}
    values = np.concatenate([value for value in traces.values() if value.size], axis=0) if any(
        value.size for value in traces.values()
    ) else np.asarray([1.0])
    maximum = max(1.0, float(np.percentile(values, 98)) * 1.10)
    colors = {"left": (80, 230, 80), "right": (255, 170, 60)}
    for name, trace in traces.items():
        if trace.size < 2:
            continue
        xs = np.linspace(plot_left, plot_right, trace.size).astype(np.int32)
        ys = (plot_bottom - np.clip(trace / maximum, 0.0, 1.0) * (plot_bottom - plot_top)).astype(np.int32)
        points = np.column_stack((xs, ys)).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, colors[name], 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"LEFT=green RIGHT=orange   source={source_fps:.1f} fps observed={observed_hz:.1f} Hz",
        (plot_left + 8, plot_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )


def compose_focus_view(
    stereo_bgr: np.ndarray,
    rois: dict[str, NormalizedRoi | None],
    measurements: dict[str, FocusMeasurement],
    states: dict[str, FocusDisplayState],
    trackers: dict[str, FocusTracker],
    *,
    source_fps: float,
    observed_hz: float,
    drag_preview: tuple[str, tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray:
    canvas = np.zeros((EYE_HEIGHT + PLOT_HEIGHT, STEREO_WIDTH, 3), dtype=np.uint8)
    canvas[:EYE_HEIGHT] = stereo_bgr
    for eye_name, offset in (("left", 0), ("right", EYE_WIDTH)):
        _draw_eye_overlay(
            canvas,
            eye_name=eye_name.upper(),
            eye_offset_x=offset,
            roi=rois[eye_name],
            measurement=measurements.get(eye_name),
            state=states.get(eye_name),
        )
    if drag_preview is not None:
        eye, start, end = drag_preview
        offset = 0 if eye == "left" else EYE_WIDTH
        x0, x1 = sorted((start[0], end[0]))
        y0, y1 = sorted((start[1], end[1]))
        cv2.rectangle(
            canvas,
            (int(np.clip(x0, 0, EYE_WIDTH)) + offset, int(np.clip(y0, 0, EYE_HEIGHT))),
            (int(np.clip(x1, 0, EYE_WIDTH)) + offset, int(np.clip(y1, 0, EYE_HEIGHT))),
            (255, 255, 0),
            2,
        )
    _draw_traces(canvas, trackers, source_fps=source_fps, observed_hz=observed_hz)
    return canvas


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("G1_IMAGE_SERVER_IP"),
        help="Orin TeleImager IPv4 (default: G1_IMAGE_SERVER_IP)",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="optional initial per-eye target ROI as normalized x,y,width,height",
    )
    parser.add_argument("--motion-threshold", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Downloads" / "head_camera_focus",
    )
    args = parser.parse_args(argv)
    if not args.host:
        parser.error("--host or G1_IMAGE_SERVER_IP is required")
    if args.motion_threshold <= 0:
        parser.error("--motion-threshold must be positive")
    return args


def _save_snapshot(
    output_dir: Path,
    view: np.ndarray,
    states: dict[str, FocusDisplayState],
    rois: dict[str, NormalizedRoi | None],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = datetime.now().astimezone().strftime("focus_%Y%m%d_%H%M%S")
    image_path = output_dir / f"{stem}.jpg"
    json_path = output_dir / f"{stem}.json"
    if not cv2.imwrite(str(image_path), view, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"failed to save {image_path}")
    payload = {
        "schema_version": "head_camera_manual_focus_v1",
        "captured_at": datetime.now().astimezone().isoformat(),
        "scores": {
            eye: {
                "live": state.live_score,
                "best": state.best_score,
                "relative_percent": state.relative_percent,
            }
            for eye, state in states.items()
        },
        "target_roi": {
            eye: {
                "x": roi.x,
                "y": roi.y,
                "width": roi.width,
                "height": roi.height,
            }
            for eye, roi in rois.items()
            if roi is not None
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return image_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise SystemExit("graphical DISPLAY/WAYLAND_DISPLAY is required")

    client = create_image_client(args.host, request_bgr=False)
    config = client.get_cam_config().get("head_camera", {})
    if config.get("binocular") is not True or tuple(config.get("image_shape", ())) != (
        EYE_HEIGHT,
        STEREO_WIDTH,
    ):
        raise SystemExit(f"expected binocular 1280x480 head camera, got {config!r}")

    rois: dict[str, NormalizedRoi | None] = {"left": args.roi, "right": args.roi}
    trackers = {
        eye: FocusTracker(motion_threshold=args.motion_threshold)
        for eye in ("left", "right")
    }
    window = "G1 Head Camera - Target ROI Focus Assistant"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    drag: dict[str, object | None] = {"eye": None, "start": None, "current": None}

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        if event in {cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN} and not (
            0 <= y < EYE_HEIGHT
        ):
            return
        if drag["eye"] is None and not (0 <= y < EYE_HEIGHT):
            return
        eye = "left" if x < EYE_WIDTH else "right"
        local_x = x if eye == "left" else x - EYE_WIDTH
        local = (int(np.clip(local_x, 0, EYE_WIDTH)), int(np.clip(y, 0, EYE_HEIGHT)))
        if event == cv2.EVENT_RBUTTONDOWN:
            rois[eye] = None
            trackers[eye].reset()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update({"eye": eye, "start": local, "current": local})
            return
        if event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if drag["eye"] == eye:
                drag["current"] = local
            return
        if event == cv2.EVENT_LBUTTONUP and drag["eye"] is not None:
            selected_eye = str(drag["eye"])
            offset = 0 if selected_eye == "left" else EYE_WIDTH
            release = (
                int(np.clip(x - offset, 0, EYE_WIDTH)),
                int(np.clip(y, 0, EYE_HEIGHT)),
            )
            try:
                rois[selected_eye] = roi_from_drag(
                    drag["start"],  # type: ignore[arg-type]
                    release,
                )
                trackers[selected_eye].reset()
            except ValueError as exc:
                print(f"ROI selection ignored: {exc}", flush=True)
            finally:
                drag.update({"eye": None, "start": None, "current": None})

    cv2.setMouseCallback(window, on_mouse)
    previous_jpg: bytes | None = None
    transition_times: deque[int] = deque(maxlen=301)
    latest_view: np.ndarray | None = None
    latest_states: dict[str, FocusDisplayState] = {}
    print(
        "Read-only focus assistant started. Drag a bounding box around the desired "
        "focus target in each eye, then sweep each focus ring slowly.",
        flush=True,
    )
    try:
        while True:
            received = receive_teleimage(client.get_head_frame, include_bgr=False)
            if received.jpg is None or received.jpg == previous_jpg:
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    return 0
                time.sleep(0.002)
                continue
            previous_jpg = received.jpg
            transition_times.append(received.received_monotonic_ns)
            encoded = np.frombuffer(received.jpg, dtype=np.uint8)
            stereo_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if stereo_bgr is None:
                raise RuntimeError("head JPEG decode failed")
            left, right = split_stereo_bgr(stereo_bgr)
            images = {"left": left, "right": right}
            measurements = {
                eye: measure_focus(images[eye], roi)
                for eye, roi in rois.items()
                if roi is not None
            }
            latest_states = {
                eye: trackers[eye].update(measurement)
                for eye, measurement in measurements.items()
            }
            observed_hz = 0.0
            if len(transition_times) >= 2:
                elapsed = (transition_times[-1] - transition_times[0]) / 1.0e9
                if elapsed > 0:
                    observed_hz = (len(transition_times) - 1) / elapsed
            latest_view = compose_focus_view(
                stereo_bgr,
                rois,
                measurements,
                latest_states,
                trackers,
                source_fps=received.fps,
                observed_hz=observed_hz,
                drag_preview=(
                    str(drag["eye"]),
                    drag["start"],  # type: ignore[arg-type]
                    drag["current"],  # type: ignore[arg-type]
                )
                if drag["eye"] is not None
                else None,
            )
            cv2.imshow(window, latest_view)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                return 0
            if key == ord("r"):
                for tracker in trackers.values():
                    tracker.reset()
            elif key == ord("c"):
                rois = {eye: None for eye in rois}
                for tracker in trackers.values():
                    tracker.reset()
            elif key in {ord("["), ord("]")}:
                factor = 0.85 if key == ord("[") else 1.15
                rois = {
                    eye: None if roi is None else roi.scaled(factor)
                    for eye, roi in rois.items()
                }
                for tracker in trackers.values():
                    tracker.reset()
            elif key == ord("s") and latest_view is not None:
                path = _save_snapshot(args.output_dir, latest_view, latest_states, rois)
                print(f"Saved focus evidence: {path}", flush=True)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        cv2.destroyAllWindows()
        close = getattr(client, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    raise SystemExit(main())
