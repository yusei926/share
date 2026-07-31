"""Pinned Unitree v1.5 AVP input and G1 29-DoF IK adapter."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import importlib
from multiprocessing import Array, Lock, Value
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .config import TeleopConfig
from .operator_view import REAL_STEREO_SHAPE
from .upstream_compat import install_logging_mp_compat


_HAND_STREAM_STALE_NS = 200_000_000
_HAND_STREAM_READY_EVENT_COUNT = 3
_HAND_INVALID_LOG_INTERVAL_NS = 3_000_000_000
_ANCHOR_STABLE_EVENT_COUNT = 5
_ANCHOR_MAX_TRANSLATION_M = 0.015
_ANCHOR_MAX_ROTATION_RAD = np.deg2rad(8.0)
# A pose discontinuity (for example, a WebXR relocalization) must not become a
# robot-arm jump. Require a fresh explicit anchor instead.
_TRACK_MAX_FRAME_TRANSLATION_M = 0.060
_TRACK_MAX_FRAME_ROTATION_RAD = np.deg2rad(20.0)
_XR_DISPLAY_MODE_ENV = "FLIP_TABLE_TELEOP_XR_DISPLAY_MODE"
_XR_DISPLAY_MODES = frozenset(("ego", "immersive"))
_COHERENT_HAND_FRAME_WAIT_S = 0.050
_HAND_DIAGNOSTIC_KEYS = (
    "missing_left_pose",
    "missing_right_pose",
    "invalid_left_wrist",
    "invalid_right_wrist",
    "invalid_left_pinch",
    "invalid_right_pinch",
    "invalid_left_unused_skeleton",
    "invalid_right_unused_skeleton",
)


class IncoherentBilateralHandFrame(RuntimeError):
    """A WebXR hand update was being written while the control loop read it.

    This is an expected shared-memory race, not a fatal AVP runtime error.  A
    caller must hold the robot and require a new explicit anchor rather than
    terminate the display/control process.
    """


def _xr_display_mode(value: str | None = None) -> str:
    """Select a robot stereo view that preserves hand tracking."""

    mode = (value if value is not None else os.environ.get(_XR_DISPLAY_MODE_ENV, "ego"))
    mode = mode.strip().lower()
    if mode not in _XR_DISPLAY_MODES:
        choices = ", ".join(sorted(_XR_DISPLAY_MODES))
        raise ValueError(f"{_XR_DISPLAY_MODE_ENV} must be one of: {choices}")
    return mode


def _valid_webxr_wrist_payload(pose: np.ndarray) -> bool:
    """Match upstream ``safe_mat_update`` for the control-critical wrist."""

    if pose.shape != (400,) or not np.isfinite(pose[:16]).all():
        return False
    wrist = pose[:16].reshape(4, 4)
    determinant = float(np.linalg.det(wrist))
    return bool(np.isfinite(determinant) and not np.isclose(determinant, 0.0, atol=1e-6))


def _unused_skeleton_invalid(pose: np.ndarray) -> bool:
    """Diagnose unused finger joints without rejecting wrist/Dex1 control."""

    unused = pose[16:]
    if not np.isfinite(unused).all():
        return True
    matrices = pose.reshape(25, 4, 4)[1:]
    determinants = np.linalg.det(matrices)
    return bool(
        not np.isfinite(determinants).all()
        or np.any(np.isclose(determinants, 0.0, atol=1e-6))
    )


def _hand_payload_assessment(
    event: object,
) -> tuple[str | None, tuple[str, ...]]:
    """Classify an unusable WebXR event without modifying the live stream.

    Official TeleVuer silently ignores incomplete events and retains its last
    complete hand sample.  We pre-validate for bilateral atomicity, but follow
    that same last-valid-sample behavior instead of turning one malformed
    browser event into an immediate disconnect.
    """

    value = getattr(event, "value", None)
    if not isinstance(value, dict):
        return "missing_left_pose", ("missing_left_pose", "missing_right_pose")
    missing = tuple(
        f"missing_{side}_pose"
        for side in ("left", "right")
        if side not in value
    )
    if missing:
        return missing[0], missing
    poses: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        try:
            pose = np.asarray(value[side], dtype=np.float64)
        except (TypeError, ValueError):
            return f"invalid_{side}_wrist", (f"invalid_{side}_wrist",)
        if not _valid_webxr_wrist_payload(pose):
            return f"invalid_{side}_wrist", (f"invalid_{side}_wrist",)
        poses[side] = pose
    for side in ("left", "right"):
        state = value.get(f"{side}State")
        if not isinstance(state, dict):
            return f"invalid_{side}_pinch", (f"invalid_{side}_pinch",)
        try:
            pinch = float(state.get("pinchValue", np.nan))
        except (TypeError, ValueError):
            return f"invalid_{side}_pinch", (f"invalid_{side}_pinch",)
        if not np.isfinite(pinch):
            return f"invalid_{side}_pinch", (f"invalid_{side}_pinch",)
    diagnostics = tuple(
        f"invalid_{side}_unused_skeleton"
        for side, pose in poses.items()
        if _unused_skeleton_invalid(pose)
    )
    return None, diagnostics


def _hand_payload_issue(event: object) -> str | None:
    """Backward-compatible critical rejection reason used by tests/tools."""

    issue, _diagnostics = _hand_payload_assessment(event)
    if issue is None:
        return None
    if "missing_" in issue:
        return "missing_bilateral_pose"
    if "_wrist" in issue:
        return "invalid_wrist_matrix"
    return "invalid_pinch_state"


def _git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_xr_runtime(root: str | Path, config: TeleopConfig) -> Path:
    path = Path(root).expanduser().resolve()
    if _git_revision(path) != config.runtime.xr_revision:
        raise RuntimeError(f"xr_teleoperate must be pinned to {config.runtime.xr_revision}")
    televuer = path / "teleop" / "televuer"
    if _git_revision(televuer) != config.runtime.televuer_revision:
        raise RuntimeError(f"TeleVuer must be pinned to {config.runtime.televuer_revision}")
    return path


def _install_heartbeat_patch() -> type:
    module = importlib.import_module("televuer.televuer")
    tele_vuer = module.TeleVuer
    if getattr(tele_vuer, "_team_ramen_heartbeat_patch", False):
        return tele_vuer
    original_init = tele_vuer.__init__
    original_hand = tele_vuer.on_hand_move
    original_binocular_stream = tele_vuer.main_image_binocular_zmq
    original_binocular_ego_stream = getattr(
        tele_vuer, "main_image_binocular_zmq_ego", None
    )

    def patched_init(self, *args, **kwargs):
        # A HAND_MOVE event contains both wrists and both hand states.  Upstream
        # stores those values in several independently locked shared objects;
        # guard the complete upstream write/read transaction with one outer
        # process-shared lock so the control side always sees one event.
        self.team_ramen_hand_frame_lock = Lock()
        self.team_ramen_hand_heartbeat_ns = Value("q", 0, lock=True)
        self.team_ramen_hand_tracking_hz = Value("d", 0.0, lock=True)
        self.team_ramen_hand_event_count = Value("q", 0, lock=True)
        self.team_ramen_hand_frame_generation = Value("q", 0, lock=True)
        self.team_ramen_hand_window_start_ns = Value("q", 0, lock=True)
        self.team_ramen_hand_window_event_count = Value("q", 0, lock=True)
        self.team_ramen_hand_invalid_event_count = Value("q", 0, lock=True)
        self.team_ramen_hand_missing_pose_count = Value("q", 0, lock=True)
        self.team_ramen_hand_invalid_wrist_count = Value("q", 0, lock=True)
        self.team_ramen_hand_invalid_pinch_count = Value("q", 0, lock=True)
        self.team_ramen_hand_invalid_unused_skeleton_count = Value(
            "q", 0, lock=True
        )
        self.team_ramen_hand_diagnostic_counts = Array(
            "q", len(_HAND_DIAGNOSTIC_KEYS), lock=True
        )
        self.team_ramen_hand_last_invalid_log_ns = Value("q", 0, lock=True)
        self.team_ramen_session_heartbeat_ns = Value("q", 0, lock=True)
        result = original_init(self, *args, **kwargs)
        if hasattr(self, "img2display"):
            self.img2display.fill(0)
        return result

    async def patched_hand(self, event, session, fps=60):
        issue, diagnostic_details = _hand_payload_assessment(event)
        if issue is None and diagnostic_details:
            with self.team_ramen_hand_invalid_unused_skeleton_count.get_lock():
                self.team_ramen_hand_invalid_unused_skeleton_count.value += len(
                    diagnostic_details
                )
            with self.team_ramen_hand_diagnostic_counts.get_lock():
                for name in diagnostic_details:
                    index = _HAND_DIAGNOSTIC_KEYS.index(name)
                    self.team_ramen_hand_diagnostic_counts[index] += 1
        if issue is not None:
            # Match official TeleVuer: discard this event and preserve the last
            # complete sample. Liveness is based only on the monotonic age of
            # the last valid bilateral event, so isolated browser glitches do
            # not create a false HOLD.
            with self.team_ramen_hand_invalid_event_count.get_lock():
                self.team_ramen_hand_invalid_event_count.value += 1
                invalid_total = int(self.team_ramen_hand_invalid_event_count.value)
            if "missing_" in issue:
                counter = self.team_ramen_hand_missing_pose_count
                public_issue = "missing_bilateral_pose"
            elif "_wrist" in issue:
                counter = self.team_ramen_hand_invalid_wrist_count
                public_issue = "invalid_wrist_matrix"
            else:
                counter = self.team_ramen_hand_invalid_pinch_count
                public_issue = "invalid_pinch_state"
            with counter.get_lock():
                counter.value += 1
            with self.team_ramen_hand_diagnostic_counts.get_lock():
                for name in diagnostic_details:
                    index = _HAND_DIAGNOSTIC_KEYS.index(name)
                    self.team_ramen_hand_diagnostic_counts[index] += 1
            now_ns = time.monotonic_ns()
            with self.team_ramen_hand_last_invalid_log_ns.get_lock():
                last_log_ns = int(self.team_ramen_hand_last_invalid_log_ns.value)
                log_due = (
                    last_log_ns <= 0
                    or now_ns - last_log_ns >= _HAND_INVALID_LOG_INTERVAL_NS
                )
                if log_due:
                    self.team_ramen_hand_last_invalid_log_ns.value = now_ns
            if log_due:
                print(
                    "AVP dropped an incomplete hand event; preserving the last "
                    "valid bilateral sample "
                    f"(reason={public_issue}, detail={issue}, total={invalid_total}).",
                    flush=True,
                )
            return None

        frame_generation = self.team_ramen_hand_frame_generation
        with self.team_ramen_hand_frame_lock:
            with frame_generation.get_lock():
                frame_generation.value += 1
            try:
                # Preserve the complete official HAND_MOVE implementation,
                # including all hand joints and motion_data_ready. The outer
                # lock only makes its bilateral shared-memory update atomic to
                # our consumer; it does not replace or reinterpret upstream.
                result = await original_hand(self, event, session, fps=fps)
            finally:
                with frame_generation.get_lock():
                    frame_generation.value += 1

        now_ns = time.monotonic_ns()
        with self.team_ramen_hand_heartbeat_ns.get_lock():
            previous_heartbeat_ns = self.team_ramen_hand_heartbeat_ns.value
            self.team_ramen_hand_heartbeat_ns.value = now_ns
        with self.team_ramen_hand_event_count.get_lock():
            self.team_ramen_hand_event_count.value += 1
        recovered = (
            previous_heartbeat_ns > 0
            and now_ns - previous_heartbeat_ns >= _HAND_STREAM_STALE_NS
        )
        with self.team_ramen_hand_window_start_ns.get_lock():
            first_ns = self.team_ramen_hand_window_start_ns.value
            if first_ns <= 0 or recovered:
                first_ns = now_ns
                self.team_ramen_hand_window_start_ns.value = first_ns
        with self.team_ramen_hand_window_event_count.get_lock():
            if recovered or self.team_ramen_hand_window_event_count.value <= 0:
                window_event_count = 1
            else:
                window_event_count = self.team_ramen_hand_window_event_count.value + 1
            self.team_ramen_hand_window_event_count.value = window_event_count
        measured_hz = 0.0
        if window_event_count > 1 and now_ns > first_ns:
            measured_hz = (window_event_count - 1) * 1.0e9 / (now_ns - first_ns)
        with self.team_ramen_hand_tracking_hz.get_lock():
            self.team_ramen_hand_tracking_hz.value = measured_hz
        return result

    async def patched_binocular_stream(
        self,
        session,
        *,
        original_stream,
        image_height: float,
        image_distance: float,
    ):
        with self.team_ramen_hand_heartbeat_ns.get_lock():
            self.team_ramen_hand_heartbeat_ns.value = 0
        with self.team_ramen_hand_tracking_hz.get_lock():
            self.team_ramen_hand_tracking_hz.value = 0.0
        with self.team_ramen_hand_event_count.get_lock():
            self.team_ramen_hand_event_count.value = 0
        with self.team_ramen_hand_frame_generation.get_lock():
            self.team_ramen_hand_frame_generation.value = 0
        with self.team_ramen_hand_window_start_ns.get_lock():
            self.team_ramen_hand_window_start_ns.value = 0
        with self.team_ramen_hand_window_event_count.get_lock():
            self.team_ramen_hand_window_event_count.value = 0
        for counter_name in (
            "team_ramen_hand_invalid_event_count",
            "team_ramen_hand_missing_pose_count",
            "team_ramen_hand_invalid_wrist_count",
            "team_ramen_hand_invalid_pinch_count",
            "team_ramen_hand_invalid_unused_skeleton_count",
            "team_ramen_hand_last_invalid_log_ns",
        ):
            counter = getattr(self, counter_name)
            with counter.get_lock():
                counter.value = 0
        with self.team_ramen_hand_diagnostic_counts.get_lock():
            for index in range(len(_HAND_DIAGNOSTIC_KEYS)):
                self.team_ramen_hand_diagnostic_counts[index] = 0

        image_background = getattr(module, "ImageBackground", None)
        hands = getattr(module, "Hands", None)
        if image_background is None or hands is None:
            try:
                return await original_stream(self, session)
            except AssertionError as exc:
                disconnected = session.CURRENT_WS_ID not in session.vuer.ws
                if disconnected and str(exc) == "Websocket session is missing.":
                    return None
                raise

        def upsert_hands() -> None:
            session.upsert(
                hands(
                    stream=True,
                    key="hands",
                    hideLeft=True,
                    hideRight=True,
                ),
                to="bgChildren",
            )

        async def update_session_heartbeat() -> None:
            while session.CURRENT_WS_ID in session.vuer.ws:
                now_ns = time.monotonic_ns()
                with self.team_ramen_session_heartbeat_ns.get_lock():
                    self.team_ramen_session_heartbeat_ns.value = now_ns
                await asyncio.sleep(0.05)

        upsert_hands()
        heartbeat_task = asyncio.create_task(update_session_heartbeat())
        try:
            period_s = 1.0 / float(self.display_fps)
            deadline = time.monotonic()
            while True:
                session.upsert(
                    [
                        image_background(
                            self.img2display[:, : self.img_width],
                            aspect=self.aspect_ratio,
                            height=image_height,
                            distanceToCamera=image_distance,
                            layers=1,
                            format="jpeg",
                            quality=80,
                            key="background-left",
                            interpolate=True,
                        ),
                        image_background(
                            self.img2display[:, self.img_width :],
                            aspect=self.aspect_ratio,
                            height=image_height,
                            distanceToCamera=image_distance,
                            layers=2,
                            format="jpeg",
                            quality=80,
                            key="background-right",
                            interpolate=True,
                        ),
                    ],
                    to="bgChildren",
                )
                deadline += period_s
                now = time.monotonic()
                if deadline < now - period_s:
                    deadline = now
                await asyncio.sleep(max(0.0, deadline - now))
        except AssertionError as exc:
            disconnected = session.CURRENT_WS_ID not in session.vuer.ws
            if disconnected and str(exc) == "Websocket session is missing.":
                return None
            raise
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task

    async def patched_immersive_stream(self, session):
        return await patched_binocular_stream(
            self,
            session,
            original_stream=original_binocular_stream,
            image_height=1.0,
            image_distance=1.0,
        )

    async def patched_ego_stream(self, session):
        return await patched_binocular_stream(
            self,
            session,
            original_stream=original_binocular_ego_stream,
            # Keep Unitree's official ego/pass-through mode, but make the
            # first-person panel comfortably readable when it contains the
            # real-only wrist and joint-angle HUDs.
            image_height=0.82,
            image_distance=1.60,
        )

    tele_vuer.__init__ = patched_init
    tele_vuer.on_hand_move = patched_hand
    tele_vuer.main_image_binocular_zmq = patched_immersive_stream
    if original_binocular_ego_stream is not None:
        tele_vuer.main_image_binocular_zmq_ego = patched_ego_stream
    tele_vuer._team_ramen_heartbeat_patch = True
    return tele_vuer


class XrInput:
    _DEX1_PINCH_CLOSED_CM = 5.0
    _DEX1_PINCH_OPEN_CM = 7.0

    def __init__(
        self,
        root: str | Path,
        config: TeleopConfig,
        *,
        real_operator_hud: bool = False,
    ) -> None:
        path = verify_xr_runtime(root, config)
        for import_root in (str(path), str(path / "teleop" / "televuer" / "src")):
            if import_root not in sys.path:
                sys.path.insert(0, import_root)
        install_logging_mp_compat()
        _install_heartbeat_patch()
        from televuer import TeleVuerWrapper
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK

        raw_preview_hz = os.environ.get(
            "FLIP_TABLE_TELEOP_PREVIEW_HZ",
            "30",
        )
        try:
            preview_hz = float(raw_preview_hz)
        except ValueError as exc:
            raise ValueError("FLIP_TABLE_TELEOP_PREVIEW_HZ must be a number in [5,30]") from exc
        if not np.isfinite(preview_hz) or not 5.0 <= preview_hz <= 30.0:
            raise ValueError("FLIP_TABLE_TELEOP_PREVIEW_HZ must be in [5,30]")
        self._frame_shape = REAL_STEREO_SHAPE if real_operator_hud else (480, 1280, 3)
        self.wrapper = TeleVuerWrapper(
            use_hand_tracking=True,
            binocular=True,
            img_shape=self._frame_shape[:2],
            # Display FPS is independent from the 30 Hz offline dataset
            # render clock.  The caller always submits the latest stereo pair.
            display_fps=preview_hz,
            display_mode=_xr_display_mode(),
            arm_reference_mode="head_yaw",
            zmq=True,
            webrtc=False,
        )
        self.ik = G1_29_ArmIK()
        self._tracking_generation = 0
        self._pending_tracking_generation = 0
        self._pending_hand_event_count = -1
        self._pending_wrist_samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._last_avp_wrist_poses: tuple[np.ndarray, np.ndarray] | None = None

    @classmethod
    def _wrist_pair_has_discontinuity(
        cls,
        previous: tuple[np.ndarray, np.ndarray] | None,
        current: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        if previous is None:
            return False
        for older, newer in zip(previous, current, strict=True):
            if np.linalg.norm(newer[:3, 3] - older[:3, 3]) > _TRACK_MAX_FRAME_TRANSLATION_M:
                return True
            if cls._rotation_distance_rad(older, newer) > _TRACK_MAX_FRAME_ROTATION_RAD:
                return True
        return False

    @staticmethod
    def _heartbeat_age_s(now_ns: int, heartbeat_ns: int) -> float | None:
        if heartbeat_ns <= 0:
            return None
        # The heartbeat is updated by TeleVuer's worker process. It can advance
        # between the caller sampling ``now_ns`` and this process reading the
        # shared value. Treat that sub-frame race as age zero; interpreting it
        # as a missing heartbeat causes a false safety stop that requires a
        # manual re-anchor even though hand tracking never disconnected.
        return max(0.0, (now_ns - heartbeat_ns) / 1.0e9)

    def liveness(
        self, now_ns: int
    ) -> dict[str, float | int | dict[str, int] | None]:
        tvuer = self.wrapper.tvuer

        def shared_value(name: str, default: float | int = 0) -> float | int:
            shared = getattr(tvuer, name, None)
            return default if shared is None else shared.value

        details_array = getattr(
            tvuer, "team_ramen_hand_diagnostic_counts", None
        )
        if details_array is None:
            details = {name: 0 for name in _HAND_DIAGNOSTIC_KEYS}
        else:
            with details_array.get_lock():
                details = {
                    name: int(details_array[index])
                    for index, name in enumerate(_HAND_DIAGNOSTIC_KEYS)
                }
        return {
            "session_age_s": self._heartbeat_age_s(
                now_ns, int(tvuer.team_ramen_session_heartbeat_ns.value)
            ),
            "hand_age_s": self._heartbeat_age_s(
                now_ns, int(tvuer.team_ramen_hand_heartbeat_ns.value)
            ),
            "hand_tracking_hz": float(tvuer.team_ramen_hand_tracking_hz.value),
            "hand_event_count": int(tvuer.team_ramen_hand_event_count.value),
            "hand_contiguous_event_count": int(
                tvuer.team_ramen_hand_window_event_count.value
            ),
            "hand_invalid_event_count": int(
                shared_value("team_ramen_hand_invalid_event_count")
            ),
            "hand_missing_pose_count": int(
                shared_value("team_ramen_hand_missing_pose_count")
            ),
            "hand_invalid_wrist_count": int(
                shared_value("team_ramen_hand_invalid_wrist_count")
            ),
            "hand_invalid_pinch_count": int(
                shared_value("team_ramen_hand_invalid_pinch_count")
            ),
            "hand_invalid_unused_skeleton_count": int(
                shared_value("team_ramen_hand_invalid_unused_skeleton_count")
            ),
            "hand_invalid_details": details,
        }

    def connected(
        self,
        now_ns: int,
        session_maximum_age_s: float,
        hand_maximum_age_s: float,
    ) -> bool:
        liveness = self.liveness(now_ns)
        session_age = liveness["session_age_s"]
        hand_age = liveness["hand_age_s"]
        contiguous_events = liveness["hand_contiguous_event_count"]
        return bool(
            session_age is not None
            and hand_age is not None
            and isinstance(contiguous_events, int)
            and contiguous_events >= _HAND_STREAM_READY_EVENT_COUNT
            and session_age <= session_maximum_age_s
            and hand_age <= hand_maximum_age_s
        )

    def render(self, stereo_rgb: np.ndarray) -> None:
        if stereo_rgb.shape != self._frame_shape or stereo_rgb.dtype != np.uint8:
            raise ValueError(
                f"AVP stereo frame must be uint8 {self._frame_shape}, got "
                f"{stereo_rgb.dtype} {stereo_rgb.shape}"
            )
        # Upstream converts BGR to RGB in its shared-memory writer.
        self.wrapper.render_to_xr(stereo_rgb[..., ::-1].copy())

    @classmethod
    def _dex1_opening_from_pinch(cls, pinch_cm: np.ndarray) -> np.ndarray:
        value = np.asarray(pinch_cm, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("AVP pinch distance must be finite left/right centimetres")
        return np.clip(
            (value - cls._DEX1_PINCH_CLOSED_CM)
            / (cls._DEX1_PINCH_OPEN_CM - cls._DEX1_PINCH_CLOSED_CM),
            0.0,
            1.0,
        )

    def _reset_ik_history(self, arm_q: np.ndarray) -> None:
        """Reset all pinned upstream IK state to the measured arm posture."""

        smooth_filter = getattr(self.ik, "smooth_filter", None)
        if smooth_filter is None or not all(
            hasattr(smooth_filter, name) for name in ("_data_queue", "_filtered_data")
        ):
            raise RuntimeError("pinned G1_29_ArmIK smoothing state is unavailable")
        seed = np.asarray(arm_q, dtype=np.float64).copy()
        self.ik.init_data = seed.copy()
        smooth_filter._data_queue = [seed.copy()]
        smooth_filter._filtered_data = seed.copy()

    def _tele_data_snapshot(self) -> Any:
        frame_lock = getattr(self.wrapper.tvuer, "team_ramen_hand_frame_lock", None)
        if frame_lock is not None:
            if not frame_lock.acquire(timeout=_COHERENT_HAND_FRAME_WAIT_S):
                raise IncoherentBilateralHandFrame(
                    "timed out waiting for one bilateral AVP hand event"
                )
            try:
                return self.wrapper.get_tele_data()
            finally:
                frame_lock.release()

        # Compatibility fallback for tests/minimal stand-ins. Production uses
        # the process-shared outer lock above.
        generation = self.wrapper.tvuer.team_ramen_hand_frame_generation
        deadline = time.monotonic() + _COHERENT_HAND_FRAME_WAIT_S
        while time.monotonic() < deadline:
            before = int(generation.value)
            if before % 2:
                # A complete bilateral shared-memory update can briefly hold
                # the odd generation while it copies both wrists and all hand
                # scalar fields.  Yield for a bounded fraction of one 30 Hz
                # control period instead of treating a normal writer/read race
                # as an operator-process failure.
                time.sleep(0.0005)
                continue
            data = self.wrapper.get_tele_data()
            after = int(generation.value)
            if before == after and after % 2 == 0:
                return data
            time.sleep(0.0005)
        raise IncoherentBilateralHandFrame(
            "could not obtain a coherent bilateral AVP hand frame"
        )

    @staticmethod
    def _rotation_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
        delta = second[:3, :3] @ first[:3, :3].T
        cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.arccos(cosine))

    @classmethod
    def _wrist_pair_is_stable(
        cls,
        reference: tuple[np.ndarray, np.ndarray],
        current: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        for reference_pose, current_pose in zip(reference, current, strict=True):
            translation = np.linalg.norm(
                current_pose[:3, 3] - reference_pose[:3, 3]
            )
            rotation = cls._rotation_distance_rad(reference_pose, current_pose)
            if (
                translation > _ANCHOR_MAX_TRANSLATION_M
                or rotation > _ANCHOR_MAX_ROTATION_RAD
            ):
                return False
        return True

    def disarm(self) -> None:
        """Discard active and pending anchors after any tracking outage."""

        self._tracking_generation = 0
        self._pending_tracking_generation = 0
        self._pending_hand_event_count = -1
        self._pending_wrist_samples = []
        self._last_avp_wrist_poses = None

    def _stable_anchor_candidate(
        self,
        tracking_generation: int,
        hand_event_count: int,
        wrist_poses: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if tracking_generation != self._pending_tracking_generation:
            self._pending_tracking_generation = tracking_generation
            self._pending_hand_event_count = -1
            self._pending_wrist_samples = []

        # The operator loop can poll faster than WebXR emits HAND_MOVE events.
        # Count only distinct bilateral samples toward the stability gate.
        if hand_event_count == self._pending_hand_event_count:
            return None
        if hand_event_count < self._pending_hand_event_count:
            self._pending_wrist_samples = []
        self._pending_hand_event_count = hand_event_count

        sample = tuple(pose.copy() for pose in wrist_poses)
        if self._pending_wrist_samples and not self._wrist_pair_is_stable(
            self._pending_wrist_samples[0], sample
        ):
            self._pending_wrist_samples = []
        self._pending_wrist_samples.append(sample)
        if len(self._pending_wrist_samples) < _ANCHOR_STABLE_EVENT_COUNT:
            return None
        return self._pending_wrist_samples[-1]

    def target(
        self,
        current_arm_q: np.ndarray,
        current_arm_dq: np.ndarray,
        current_dex1_opening: np.ndarray,
        tracking_generation: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        data = self._tele_data_snapshot()
        arm_q = np.asarray(current_arm_q, dtype=np.float64)
        arm_dq = np.asarray(current_arm_dq, dtype=np.float64)
        hand = np.asarray(current_dex1_opening, dtype=np.float64)
        if arm_q.shape != (14,) or arm_dq.shape != (14,):
            raise ValueError("G1 IK current arm state must be 14-D")
        if not np.isfinite(arm_q).all() or not np.isfinite(arm_dq).all():
            raise ValueError("G1 IK current arm state contains NaN or Inf")
        if hand.shape != (2,) or not np.isfinite(hand).all():
            raise ValueError("Dex1 current opening state must be finite 2-D")
        if not isinstance(tracking_generation, int) or tracking_generation <= 0:
            raise ValueError("tracking_generation must be a positive integer")
        left = np.asarray(data.left_wrist_pose, dtype=np.float64)
        right = np.asarray(data.right_wrist_pose, dtype=np.float64)
        if left.shape != (4, 4) or right.shape != (4, 4) or not (
            np.isfinite(left).all() and np.isfinite(right).all()
        ):
            raise ValueError("AVP wrist poses are invalid")

        reanchored = tracking_generation != self._tracking_generation
        if reanchored:
            hand_event_count = int(
                self.wrapper.tvuer.team_ramen_hand_event_count.value
            )
            candidate = self._stable_anchor_candidate(
                tracking_generation,
                hand_event_count,
                (left, right),
            )
            if candidate is None:
                return None
            self._tracking_generation = tracking_generation
            self._pending_tracking_generation = 0
            self._pending_hand_event_count = -1
            self._pending_wrist_samples = []
            self._last_avp_wrist_poses = tuple(pose.copy() for pose in candidate)
            # Upstream keeps both an IK warm start and four historical solutions.
            # Reusing either after a tracking outage blends the old branch into
            # the first newly acquired pose and can cause a command jump. Prime
            # both with the measured posture and hold exactly one target. From
            # the next hand frame onward, pass TeleVuer's absolute wrist poses
            # directly to the official G1_29 IK, matching upstream behavior.
            self._reset_ik_history(arm_q)
            return (
                arm_q.copy(),
                np.zeros(14, dtype=np.float64),
                np.clip(hand, 0.0, 1.0),
            )
        current_pair = (left, right)
        if self._wrist_pair_has_discontinuity(self._last_avp_wrist_poses, current_pair):
            # Never silently change the mapping. The session catches this
            # expected tracking fault, latches HOLD, and requires an explicit
            # r press after the hands are visible again.
            raise IncoherentBilateralHandFrame(
                "AVP wrist pose jumped; explicit re-anchor required"
            )
        self._last_avp_wrist_poses = tuple(pose.copy() for pose in current_pair)
        solved, feedforward = self.ik.solve_ik(left, right, arm_q, arm_dq)
        solved = np.asarray(solved, dtype=np.float64)
        feedforward = np.asarray(feedforward, dtype=np.float64)
        if solved.shape != (14,) or not np.isfinite(solved).all():
            raise ValueError("official G1_29 IK returned an invalid target")
        if feedforward.shape != (14,) or not np.isfinite(feedforward).all():
            raise ValueError(
                "official G1_29 IK returned invalid feedforward torque"
            )
        # TeleVuer reports thumb-index distance in metres and upstream scales it
        # by 100 before mapping 5..7 cm onto the Dex1 stroke.
        pinch_cm = np.asarray(
            (data.left_hand_pinchValue, data.right_hand_pinchValue), dtype=np.float64
        )
        opening = self._dex1_opening_from_pinch(pinch_cm)
        return solved, feedforward, opening

    def close(self) -> None:
        self.wrapper.close()
