"""Process-isolated, latest-only four-camera Desktop preview.

The preview is deliberately a diagnostic sink.  JPEG decoding, HUD drawing,
and GUI event handling happen in a separate process and can never delay robot
control, model inference, or dataset recording.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from typing import Any

import numpy as np

from .operator_view import compose_real_desktop_view


def environment_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "true" if default else "false").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean")


def enable_camera_preview_for_policy_runner() -> None:
    """Enable the monitor for direct real-policy runners unless opted out."""

    os.environ.setdefault("IROS_REAL_EVAL_DESKTOP_PREVIEW", "true")


def decode_camera_jpeg(payload: bytes) -> np.ndarray:
    import cv2

    value = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if value is None or value.shape != (480, 640, 3):
        raise ValueError("camera JPEG does not decode to 640x480 RGB")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _offer_latest(updates: Any, value: Any) -> None:
    """Replace a stale queued paint request without ever blocking the caller."""

    try:
        updates.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        updates.get_nowait()
    except queue.Empty:
        pass
    try:
        updates.put_nowait(value)
    except queue.Full:
        # The GUI may have raced us and inserted/consumed an item.  A future
        # observation will retry; display loss must never affect control.
        pass


def _desktop_preview_worker(updates: Any, window_title: str) -> None:
    """Run OpenCV GUI in an isolated process; never participate in control."""

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print(
            "Desktop camera monitor disabled because no graphical display is set.",
            flush=True,
        )
        return
    import cv2

    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1200, 900)
        while True:
            update = updates.get()
            if update is None:
                return
            images = {
                role: decode_camera_jpeg(payload)
                for role, payload in update["camera_jpeg"].items()
            }
            view = compose_real_desktop_view(
                images["head_left"],
                images["head_right"],
                images["left_wrist"],
                images["right_wrist"],
                np.asarray(update["arm_joint_position_rad"], dtype=np.float64),
                str(update["status"]),
            )
            # Composition uses RGB; OpenCV windows consume BGR.
            cv2.imshow(window_title, view[..., ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                return
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                return
    except BaseException as exc:  # noqa: BLE001
        # A Desktop/Qt failure must not stop inference, recording, or safety.
        print(f"Desktop camera monitor stopped: {type(exc).__name__}: {exc}", flush=True)
    finally:
        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass


class DesktopPreviewProcess:
    """Non-blocking latest-frame publisher for a four-camera monitor."""

    def __init__(
        self,
        *,
        window_title: str = "IROS 2026 RAMEN - Teleoperation Monitor",
    ) -> None:
        context = mp.get_context("spawn")
        self._updates = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_desktop_preview_worker,
            args=(self._updates, window_title),
            name="iros-four-camera-monitor",
            daemon=True,
        )
        self._process.start()

    def submit(
        self,
        camera_jpeg: dict[str, bytes],
        arm_joint_position_rad: list[float],
        status: str,
    ) -> None:
        if not self._process.is_alive():
            return
        _offer_latest(
            self._updates,
            {
                "camera_jpeg": dict(camera_jpeg),
                "arm_joint_position_rad": list(arm_joint_position_rad),
                "status": status,
            },
        )

    def close(self) -> None:
        if self._process.is_alive():
            _offer_latest(self._updates, None)
            self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        self._updates.close()
