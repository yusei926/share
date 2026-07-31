"""Pure NumPy adapters from model-native contracts to a safe 16-D boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from inference.desktop.upper_policy.act_pick_leg_contract import (
    ACTION_MAX as ACT_ACTION_MAX,
    ACTION_MIN as ACT_ACTION_MIN,
    CAMERA_KEYS as ACT_CAMERA_KEYS,
    CAMERA_ROLES as ACT_CAMERA_ROLES,
    canonical_action as canonical_act_action,
    compose_model_state as compose_act_state,
)
from inference.desktop.upper_policy.groot_pick_leg_contract import (
    DEX1_DATASET_OPEN_VALUE,
    compose_model_state as compose_pick_state,
    extract_executable_action as extract_pick_action,
)
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    source_euler_xyz_pose_to_xyz_rot6d,
)
from .registry import ModelSpec


CANONICAL_ACTION_DIM = 16
BODY_DIM = 29
EEF_XYZ_EULER_DIM = 12


@dataclass(frozen=True)
class CanonicalObservation:
    body_joint_position_rad: np.ndarray
    dex1_opening_fraction: np.ndarray
    camera_jpeg: Mapping[str, bytes]
    eef_xyz_euler: np.ndarray | None = None

    def __post_init__(self) -> None:
        body = _vector(self.body_joint_position_rad, BODY_DIM, "body_joint_position_rad")
        dex1 = _vector(self.dex1_opening_fraction, 2, "dex1_opening_fraction")
        if np.any((dex1 < 0.0) | (dex1 > 1.0)):
            raise ValueError("Dex1 opening fractions must be in [0,1]")
        if self.eef_xyz_euler is not None:
            eef = _vector(self.eef_xyz_euler, EEF_XYZ_EULER_DIM, "eef_xyz_euler")
            eef.setflags(write=False)
            object.__setattr__(self, "eef_xyz_euler", eef)
        if not isinstance(self.camera_jpeg, Mapping):
            raise ValueError("camera_jpeg must be a mapping")
        body.setflags(write=False)
        dex1.setflags(write=False)
        object.__setattr__(self, "body_joint_position_rad", body)
        object.__setattr__(self, "dex1_opening_fraction", dex1)


class FamilyAdapter:
    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def validate_observation(self, observation: CanonicalObservation) -> None:
        missing = set(self.spec.camera_roles) - set(observation.camera_jpeg)
        if missing:
            raise ValueError(f"missing camera roles: {sorted(missing)}")
        if any(not bytes(observation.camera_jpeg[role]) for role in self.spec.camera_roles):
            raise ValueError("camera JPEG payloads must be non-empty")

    def model_state(self, observation: CanonicalObservation) -> np.ndarray:
        raise NotImplementedError

    def canonical_action(
        self,
        native_chunk: Sequence[Sequence[float]],
        observation: CanonicalObservation,
    ) -> np.ndarray:
        raise NotImplementedError

    def synthetic_native_action(self) -> np.ndarray:
        """Return a deterministic contract probe; never load model weights."""
        raise NotImplementedError

    def offline_request(
        self,
        observation: CanonicalObservation,
        state: np.ndarray,
    ) -> dict[str, Any]:
        """Build the trusted local worker request for this exact family."""
        raise NotImplementedError

    def _finalize(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float64)
        expected = (self.spec.model_action_horizon, CANONICAL_ACTION_DIM)
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"canonical action must be finite {expected}, got {value.shape}")
        if np.any((value[:, 14:] < 0.0) | (value[:, 14:] > 1.0)):
            raise ValueError("canonical Dex1 opening fractions must be in [0,1]")
        return value


class GrootAbsoluteJointAdapter(FamilyAdapter):
    def model_state(self, observation: CanonicalObservation) -> np.ndarray:
        self.validate_observation(observation)
        return compose_pick_state(
            observation.body_joint_position_rad,
            observation.dex1_opening_fraction,
        )

    def canonical_action(
        self,
        native_chunk: Sequence[Sequence[float]],
        observation: CanonicalObservation,
    ) -> np.ndarray:
        del observation
        executable = extract_pick_action(native_chunk).astype(np.float64)
        executable[:, 14:] /= DEX1_DATASET_OPEN_VALUE
        return self._finalize(executable)

    def synthetic_native_action(self) -> np.ndarray:
        native = np.zeros((self.spec.model_action_horizon, 38), dtype=np.float64)
        native[:, 36:] = [1.125, 3.375]
        return native

    def offline_request(
        self,
        observation: CanonicalObservation,
        state: np.ndarray,
    ) -> dict[str, Any]:
        roles = ("head_left", "head_right", "left_wrist", "right_wrist")
        keys = tuple(f"observation.images.cam_{index}" for index in range(4))
        return {
            "type": "predict",
            "request_id": 1,
            "state": state.tolist(),
            "cameras": {
                key: observation.camera_jpeg[role]
                for key, role in zip(keys, roles, strict=True)
            },
            "task": self.spec.task,
        }


class ActAbsoluteJoint16Adapter(FamilyAdapter):
    """Adapter for ACT policies trained directly on G1 arms14 + Dex1 2."""

    def model_state(self, observation: CanonicalObservation) -> np.ndarray:
        self.validate_observation(observation)
        return compose_act_state(
            observation.body_joint_position_rad,
            observation.dex1_opening_fraction,
        )

    def canonical_action(
        self,
        native_chunk: Sequence[Sequence[float]],
        observation: CanonicalObservation,
    ) -> np.ndarray:
        del observation
        return self._finalize(canonical_act_action(native_chunk))

    def synthetic_native_action(self) -> np.ndarray:
        midpoint = (ACT_ACTION_MIN + ACT_ACTION_MAX) / 2.0
        return np.repeat(
            midpoint[None, :], self.spec.model_action_horizon, axis=0
        )

    def offline_request(
        self,
        observation: CanonicalObservation,
        state: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "type": "predict",
            "request_id": 1,
            "state": state.tolist(),
            "cameras": {
                key: observation.camera_jpeg[role]
                for key, role in zip(
                    ACT_CAMERA_KEYS, ACT_CAMERA_ROLES, strict=True
                )
            },
            "task": self.spec.task,
        }


class GrootRelativeEefAdapter(FamilyAdapter):
    def model_state(self, observation: CanonicalObservation) -> np.ndarray:
        self.validate_observation(observation)
        if observation.eef_xyz_euler is None:
            raise ValueError("relative-EEF model requires root-frame EEF XYZ+Euler")
        result = np.zeros(49, dtype=np.float32)
        result[0:9] = source_euler_xyz_pose_to_xyz_rot6d(
            observation.eef_xyz_euler[0:6].tolist()
        )
        result[9:18] = source_euler_xyz_pose_to_xyz_rot6d(
            observation.eef_xyz_euler[6:12].tolist()
        )
        physical_hands = (
            observation.dex1_opening_fraction * DEX1_DATASET_OPEN_VALUE
        )
        result[18] = -physical_hands[0] / 3.0
        result[25] = physical_hands[1] / 3.0
        result[32:39] = observation.body_joint_position_rad[15:22]
        result[39:46] = observation.body_joint_position_rad[22:29]
        result[46:49] = observation.body_joint_position_rad[12:15]
        if not np.isfinite(result).all():
            raise RuntimeError("relative-EEF state conversion violated 49-D contract")
        return result

    def canonical_action(
        self,
        native_chunk: Sequence[Sequence[float]],
        observation: CanonicalObservation,
    ) -> np.ndarray:
        del observation
        logical = np.asarray(native_chunk, dtype=np.float64)
        expected = (self.spec.model_action_horizon, 53)
        if logical.shape != expected or not np.isfinite(logical).all():
            raise ValueError(f"coarse-insert decoded action must be finite {expected}")
        # This older checkpoint predates the generalized Dex1 hand-synergy
        # representation. Its model card and serialized processor define one
        # physical scalar per hand: left=-3*a[18], right=3*a[25].
        physical = np.empty((logical.shape[0], 16), dtype=np.float64)
        physical[:, :7] = logical[:, 32:39]
        physical[:, 7:14] = logical[:, 39:46]
        physical[:, 14] = np.clip(-3.0 * logical[:, 18], 0.0, 4.5)
        physical[:, 15] = np.clip(3.0 * logical[:, 25], 0.0, 4.5)
        physical[:, 14:] /= DEX1_DATASET_OPEN_VALUE
        return self._finalize(physical)

    def synthetic_native_action(self) -> np.ndarray:
        native = np.zeros((self.spec.model_action_horizon, 53), dtype=np.float64)
        native[:, 18] = -0.375
        native[:, 25] = 1.125
        return native

    def offline_request(
        self,
        observation: CanonicalObservation,
        state: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "type": "predict",
            "request_id": 1,
            "state": state.tolist(),
            "cameras": {
                f"observation.images.{role}": observation.camera_jpeg[role]
                for role in ("head_left", "left_wrist", "right_wrist")
            },
            "task": self.spec.task,
        }


class DiffusionChunkRelativeAdapter(FamilyAdapter):
    def model_state(self, observation: CanonicalObservation) -> np.ndarray:
        self.validate_observation(observation)
        result = np.concatenate(
            (
                observation.body_joint_position_rad[12:15],
                observation.body_joint_position_rad[15:29],
                observation.dex1_opening_fraction * DEX1_DATASET_OPEN_VALUE,
            )
        )
        if result.shape != (19,):
            raise RuntimeError("Diffusion state conversion violated 19-D contract")
        return result.astype(np.float32)

    def canonical_action(
        self,
        native_chunk: Sequence[Sequence[float]],
        observation: CanonicalObservation,
    ) -> np.ndarray:
        native = np.asarray(native_chunk, dtype=np.float64)
        expected = (self.spec.model_action_horizon, 16)
        if native.shape != expected or not np.isfinite(native).all():
            raise ValueError(f"chunk-relative action must be finite {expected}")
        result = native.copy()
        # Arm deltas are anchored exactly once to the measured arm state at the
        # beginning of this chunk. They are never recursively integrated.
        result[:, :14] += observation.body_joint_position_rad[15:29]
        result[:, 14:] /= DEX1_DATASET_OPEN_VALUE
        return self._finalize(result)

    def synthetic_native_action(self) -> np.ndarray:
        native = np.zeros((self.spec.model_action_horizon, 16), dtype=np.float64)
        native[:, 14:] = [1.125, 3.375]
        return native

    def offline_request(
        self,
        observation: CanonicalObservation,
        state: np.ndarray,
    ) -> dict[str, Any]:
        return {
            "type": "predict",
            "request_id": 1,
            "state_history": [state.tolist(), state.tolist()],
            "camera_history": {
                f"observation.images.{role}": [
                    observation.camera_jpeg[role],
                    observation.camera_jpeg[role],
                ]
                for role in ("head_left", "left_wrist", "right_wrist")
            },
        }


def adapter_for(spec: ModelSpec) -> FamilyAdapter:
    adapters = {
        "act_absolute_joint16_v1": ActAbsoluteJoint16Adapter,
        "groot_absolute_joint_v1": GrootAbsoluteJointAdapter,
        "groot_relative_eef_v1": GrootRelativeEefAdapter,
        "diffusion_chunk_relative_v1": DiffusionChunkRelativeAdapter,
    }
    try:
        return adapters[spec.family](spec)
    except KeyError as exc:
        raise ValueError(f"no adapter for family {spec.family!r}") from exc


def _vector(values: Sequence[float], width: int, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}], got {result.shape}")
    return result
