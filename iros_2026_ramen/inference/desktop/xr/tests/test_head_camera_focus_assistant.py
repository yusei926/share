from __future__ import annotations

import cv2
import numpy as np
import pytest

from inference.desktop.xr.head_camera_focus_assistant import (
    EYE_HEIGHT,
    EYE_WIDTH,
    FocusMeasurement,
    FocusTracker,
    NormalizedRoi,
    compose_focus_view,
    measure_focus,
    parse_roi,
    roi_from_drag,
    split_stereo_bgr,
)


TARGET_ROI = NormalizedRoi(0.20, 0.45, 0.60, 0.45)


def _target_checkerboard() -> np.ndarray:
    image = np.full((EYE_HEIGHT, EYE_WIDTH, 3), 110, dtype=np.uint8)
    x0, y0, x1, y1 = TARGET_ROI.bounds(image)
    tile = 12
    yy, xx = np.indices((y1 - y0, x1 - x0))
    pattern = (((xx // tile) + (yy // tile)) % 2 * 220 + 15).astype(np.uint8)
    image[y0:y1, x0:x1] = pattern[..., None]
    return image


def _measurement(score: float, gray: np.ndarray, *, contrast: float = 40.0) -> FocusMeasurement:
    return FocusMeasurement(
        score=score,
        laplacian_variance=score,
        tenengrad=score,
        brightness=100.0,
        contrast=contrast,
        clipped_fraction=0.0,
        roi_gray_small=gray,
    )


def test_human_drag_defines_arbitrary_target_box() -> None:
    roi = roi_from_drag((500, 400), (100, 120))
    assert roi == NormalizedRoi(
        100 / EYE_WIDTH,
        120 / EYE_HEIGHT,
        400 / EYE_WIDTH,
        280 / EYE_HEIGHT,
    )
    with pytest.raises(ValueError, match="at least"):
        roi_from_drag((100, 100), (110, 110))
    assert roi_from_drag((0, 0), (32, 32)).width == pytest.approx(0.05)


def test_focus_measurement_falls_when_selected_target_is_blurred() -> None:
    sharp = _target_checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4.0)
    sharp_score = measure_focus(sharp, TARGET_ROI).score
    blurred_score = measure_focus(blurred, TARGET_ROI).score
    assert sharp_score > blurred_score * 10.0


def test_background_edges_do_not_affect_selected_box_score() -> None:
    base = _target_checkerboard()
    changed = base.copy()
    _x0, y0, _x1, _y1 = TARGET_ROI.bounds(base)
    background = changed[:y0]
    yy, xx = np.indices(background.shape[:2])
    background[:] = ((((xx // 3) + (yy // 3)) % 2) * 255)[..., None]
    assert measure_focus(base, TARGET_ROI).score == pytest.approx(
        measure_focus(changed, TARGET_ROI).score
    )


def test_tracker_updates_best_only_after_stable_frames() -> None:
    tracker = FocusTracker(smoothing_frames=3, stable_frames=2, motion_threshold=1.0)
    gray = np.full((64, 96), 100, dtype=np.uint8)
    first = tracker.update(_measurement(10.0, gray))
    second = tracker.update(_measurement(20.0, gray))
    third = tracker.update(_measurement(30.0, gray))
    assert first.best_score == 0.0
    assert second.best_score == 0.0
    assert third.best_score == pytest.approx(20.0)
    moving = tracker.update(_measurement(100.0, np.full_like(gray, 150)))
    assert not moving.stable
    assert moving.best_score == pytest.approx(20.0)


def test_split_stereo_and_roi_parser_contract() -> None:
    packed = np.zeros((EYE_HEIGHT, EYE_WIDTH * 2, 3), dtype=np.uint8)
    packed[:, EYE_WIDTH:] = 255
    left, right = split_stereo_bgr(packed)
    assert int(left.max()) == 0
    assert int(right.min()) == 255
    assert parse_roi("0.2,0.45,0.6,0.45") == TARGET_ROI
    with pytest.raises(ValueError):
        split_stereo_bgr(np.zeros((480, 640, 3), dtype=np.uint8))


def test_gui_can_render_before_operator_selects_boxes() -> None:
    packed = np.zeros((EYE_HEIGHT, EYE_WIDTH * 2, 3), dtype=np.uint8)
    trackers = {"left": FocusTracker(), "right": FocusTracker()}
    view = compose_focus_view(
        packed,
        {"left": None, "right": None},
        {},
        {},
        trackers,
        source_fps=30.0,
        observed_hz=29.9,
        drag_preview=("left", (50, 70), (300, 350)),
    )
    assert view.shape == (EYE_HEIGHT + 180, EYE_WIDTH * 2, 3)
    assert int(view.max()) > 0
