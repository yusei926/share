"""Compose the stereo Apple Vision Pro operator views.

The robot's head cameras remain a true left/right stereo pair.  For real
teleoperation only, the D405 wrist cameras and measured arm angles are added
as compact *monocular HUD panels* at the outer edges of both eyes. Duplicating
these diagnostic panels in both eyes avoids inventing a false wrist-camera
stereo baseline while keeping the head image geometrically correct in the
middle.
"""

from __future__ import annotations

import numpy as np


HEAD_SHAPE = (480, 640, 3)
ARM_JOINT_COUNT = 14
HUD_PANEL_WIDTH = 160
HUD_PANEL_HEIGHT = 480
REAL_EYE_WIDTH = HUD_PANEL_WIDTH + HEAD_SHAPE[1] + HUD_PANEL_WIDTH
REAL_STEREO_SHAPE = (HEAD_SHAPE[0], REAL_EYE_WIDTH * 2, 3)
REAL_DESKTOP_SHAPE = (HEAD_SHAPE[0] * 2, HEAD_SHAPE[1] * 2, 3)

_LEFT_JOINT_LABELS = ("L SP", "L SR", "L SY", "L EL", "L WR", "L WP", "L WY")
_RIGHT_JOINT_LABELS = ("R SP", "R SR", "R SY", "R EL", "R WR", "R WP", "R WY")


def _status_color(status: str) -> tuple[int, int, int]:
    return {
        "TRACKING": (80, 255, 120),
        "HANDS READY": (80, 255, 120),
        "ANCHORING": (80, 220, 255),
        "PRESS R": (80, 165, 255),
        "HANDS WAIT": (255, 190, 80),
    }.get(status, (80, 180, 255))


def _rgb(value: np.ndarray, shape: tuple[int, int, int], label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.uint8:
        raise ValueError(f"{label} must be uint8 {shape}, got {result.dtype} {result.shape}")
    return result


def compose_head_stereo_view(
    head_left: np.ndarray,
    head_right: np.ndarray,
) -> np.ndarray:
    """Return the unmodified 1280x480 side-by-side stereo head view."""

    left = _rgb(head_left, HEAD_SHAPE, "head_left")
    right = _rgb(head_right, HEAD_SHAPE, "head_right")
    return np.concatenate((left, right), axis=1)


def _arm_joint_angles_deg(arm_joint_position_rad: np.ndarray) -> np.ndarray:
    values = np.asarray(arm_joint_position_rad, dtype=np.float64)
    if values.shape != (ARM_JOINT_COUNT,) or not np.isfinite(values).all():
        raise ValueError("arm_joint_position_rad must be finite 14-D")
    return np.rad2deg(values)


def _hud_panel(
    wrist_rgb: np.ndarray,
    *,
    side: str,
    angles_deg: np.ndarray,
    hand_status: str,
) -> np.ndarray:
    """Return a labelled compact diagnostic panel for one arm."""

    import cv2

    wrist = _rgb(wrist_rgb, HEAD_SHAPE, f"{side}_wrist")
    panel = np.zeros((HUD_PANEL_HEIGHT, HUD_PANEL_WIDTH, 3), dtype=np.uint8)
    # Keep the native 4:3 wrist aspect ratio.  The black margins deliberately
    # separate an auxiliary monocular image from the central stereo image.
    if not hand_status:
        raise ValueError("hand_status must be non-empty")
    thumbnail_height = 108
    thumbnail_width = 144
    thumbnail = cv2.resize(
        wrist,
        (thumbnail_width, thumbnail_height),
        interpolation=cv2.INTER_AREA,
    )
    x = (HUD_PANEL_WIDTH - thumbnail_width) // 2
    y = 48
    panel[y : y + thumbnail_height, x : x + thumbnail_width] = thumbnail
    labels = _LEFT_JOINT_LABELS if side == "left" else _RIGHT_JOINT_LABELS
    title = "LEFT WRIST" if side == "left" else "RIGHT WRIST"
    cv2.putText(
        panel,
        title,
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (80, 220, 255),
        1,
        cv2.LINE_AA,
    )
    status_color = _status_color(hand_status)
    cv2.putText(
        panel,
        hand_status,
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        status_color,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(panel, (5, 43), (154, 160), (80, 220, 255), 1)
    cv2.putText(
        panel,
        "measured arm (deg)",
        (8, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    for index, (label, angle) in enumerate(zip(labels, angles_deg, strict=True)):
        text = f"{label} {angle:+5.1f}"
        cv2.putText(
            panel,
            text,
            (8, 212 + 32 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return panel


def compose_real_operator_stereo_view(
    head_left: np.ndarray,
    head_right: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    arm_joint_position_rad: np.ndarray,
    hand_status: str,
) -> np.ndarray:
    """Compose head stereo with real-only wrist and measured-angle HUDs.

    The same left/right panels are placed around both eyes.  Thus the central
    640x480 area remains the unmodified corresponding head image for that eye,
    while the operator can consult each wrist view and live joint angles
    without leaving the AVP display.
    """

    left = _rgb(head_left, HEAD_SHAPE, "head_left")
    right = _rgb(head_right, HEAD_SHAPE, "head_right")
    angles_deg = _arm_joint_angles_deg(arm_joint_position_rad)
    left_panel = _hud_panel(
        left_wrist,
        side="left",
        angles_deg=angles_deg[:7],
        hand_status=hand_status,
    )
    right_panel = _hud_panel(
        right_wrist,
        side="right",
        angles_deg=angles_deg[7:],
        hand_status=hand_status,
    )
    left_eye = np.concatenate((left_panel, left, right_panel), axis=1)
    right_eye = np.concatenate((left_panel, right, right_panel), axis=1)
    return np.concatenate((left_eye, right_eye), axis=1)


def _desktop_camera_tile(
    rgb: np.ndarray,
    *,
    title: str,
    status: str | None = None,
    joint_labels: tuple[str, ...] = (),
    joint_angles_deg: np.ndarray | None = None,
) -> np.ndarray:
    """Label one native-resolution Desktop monitor tile."""

    import cv2

    tile = _rgb(rgb, HEAD_SHAPE, title).copy()
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (639, 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, tile, 0.38, 0.0, tile)
    cv2.putText(
        tile,
        title,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 230, 100),
        2,
        cv2.LINE_AA,
    )
    if status is not None:
        color = _status_color(status)
        cv2.putText(
            tile,
            status,
            (430, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            color,
            1,
            cv2.LINE_AA,
        )
    if joint_labels:
        if joint_angles_deg is None or joint_angles_deg.shape != (7,):
            raise ValueError("Desktop wrist tile requires seven joint angles")
        overlay = tile.copy()
        cv2.rectangle(overlay, (8, 48), (238, 285), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, tile, 0.32, 0.0, tile)
        cv2.putText(
            tile,
            "measured joints (deg)",
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        for index, (label, angle) in enumerate(
            zip(joint_labels, joint_angles_deg, strict=True)
        ):
            cv2.putText(
                tile,
                f"{label} {angle:+6.1f}",
                (18, 101 + 28 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (250, 250, 250),
                1,
                cv2.LINE_AA,
            )
    return tile


def compose_real_desktop_view(
    head_left: np.ndarray,
    head_right: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    arm_joint_position_rad: np.ndarray,
    hand_status: str,
) -> np.ndarray:
    """Return a readable 2x2 Desktop monitor for all four real cameras."""

    if not hand_status:
        raise ValueError("hand_status must be non-empty")
    angles_deg = _arm_joint_angles_deg(arm_joint_position_rad)
    top = np.concatenate(
        (
            _desktop_camera_tile(
                head_left,
                title="HEAD LEFT",
                status=hand_status,
            ),
            _desktop_camera_tile(
                head_right,
                title="HEAD RIGHT",
                status=hand_status,
            ),
        ),
        axis=1,
    )
    bottom = np.concatenate(
        (
            _desktop_camera_tile(
                left_wrist,
                title="LEFT WRIST",
                joint_labels=_LEFT_JOINT_LABELS,
                joint_angles_deg=angles_deg[:7],
            ),
            _desktop_camera_tile(
                right_wrist,
                title="RIGHT WRIST",
                joint_labels=_RIGHT_JOINT_LABELS,
                joint_angles_deg=angles_deg[7:],
            ),
        ),
        axis=1,
    )
    result = np.concatenate((top, bottom), axis=0)
    if result.shape != REAL_DESKTOP_SHAPE:
        raise RuntimeError(f"unexpected Desktop monitor shape: {result.shape}")
    return result
