"""Official G1-29 arm gravity feed-forward for physical arm targets."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np


EXPECTED_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
# This is a fail-closed validation bound, not a replacement for the official
# RNEA output. All three pre-motion poses are below 7 Nm with the pinned model.
MAX_ABS_GRAVITY_TORQUE_NM = 15.0


def find_pinned_g1_model_cache() -> Path:
    """Locate the model cache inside the wrapper-verified xr_teleoperate tree."""

    candidates: list[Path] = []
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry).resolve()
        candidates.extend(
            (
                root / "teleop/g1_29_model_cache.pkl",
                root / "g1_29_model_cache.pkl",
            )
        )
    existing = tuple(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise RuntimeError(
            "expected exactly one pinned xr_teleoperate G1-29 model cache, "
            f"found {[str(path) for path in existing]}"
        )
    return existing[0]


class OfficialG1ArmGravityCompensator:
    """Compute the same Pinocchio RNEA term returned by official G1 arm IK."""

    def __init__(self, cache_path: Path | None = None) -> None:
        import pinocchio as pin

        self._pin = pin
        self.cache_path = (
            find_pinned_g1_model_cache()
            if cache_path is None
            else Path(cache_path).resolve()
        )
        payload = self.cache_path.read_bytes()
        cache = pickle.loads(payload)  # noqa: S301 - pinned official local artifact
        model = cache.get("reduced_model")
        if model is None or model.nq != 14 or model.nv != 14:
            raise RuntimeError("pinned G1 reduced model must have nq=nv=14")
        names = tuple(str(name) for name in model.names[1:])
        if names != EXPECTED_ARM_JOINT_NAMES:
            raise RuntimeError(f"unexpected G1 reduced-model joint order: {names}")
        self._model = model
        self._data = model.createData()
        self.cache_sha256 = hashlib.sha256(payload).hexdigest()

    def torque_nm(self, arm_position_rad: np.ndarray) -> np.ndarray:
        q = np.asarray(arm_position_rad, dtype=np.float64)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("gravity compensation input must be finite arms[14]")
        torque = np.asarray(
            self._pin.rnea(
                self._model,
                self._data,
                q,
                np.zeros(self._model.nv),
                np.zeros(self._model.nv),
            ),
            dtype=np.float64,
        )
        if torque.shape != (14,) or not np.isfinite(torque).all():
            raise RuntimeError("official G1 RNEA returned an invalid torque vector")
        maximum = float(np.max(np.abs(torque)))
        if maximum > MAX_ABS_GRAVITY_TORQUE_NM:
            raise RuntimeError(
                "official G1 gravity feed-forward exceeded the deployment bound "
                f"({maximum:.3f}>{MAX_ABS_GRAVITY_TORQUE_NM:.3f} Nm)"
            )
        return torque
