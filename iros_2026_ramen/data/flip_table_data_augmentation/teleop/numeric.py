"""Convert raw teleoperation traces into the six immutable numeric features."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..fk_audit import G1_BODY_JOINT_ORDER
from ..io_utils import sha256_file
from ..source_contract import NUMERIC_FEATURES
from .raw_episode import RAW_EPISODE_SCHEMA_VERSION


NUMERIC_CONVERSION_SCHEMA_VERSION = "team_ramen_flip_table_teleop_numeric/v1"
HAND_COMMAND_OPEN = 4.5
ARM_SLICE = slice(15, 29)
EEF_FRAME_NAMES = {
    "left": "left_wrist_yaw_link",
    "right": "right_wrist_yaw_link",
}
TOOL_TRANSLATION_M = np.asarray((0.05, 0.0, 0.0), dtype=np.float64)
SIM_TELEOP_EPISODE_INDEX_OFFSET = 1_000_000


def sim_teleop_source_index(episode_id: str) -> int:
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("teleoperation episode ID must be non-empty")
    # A 48-bit deterministic namespace is stable across separate export runs
    # and remains well within signed int64/HDF5 integer limits.
    return SIM_TELEOP_EPISODE_INDEX_OFFSET + int(
        hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:12], 16
    )


def _finite_vector(value: Any, width: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite [{width}]")
    return result


def compose_robot_q(root_pose_xyzw: Any, body_joint_position_rad: Any) -> np.ndarray:
    root = _finite_vector(root_pose_xyzw, 7, "root pose")
    body = _finite_vector(body_joint_position_rad, 29, "body joint position")
    if not math.isclose(float(np.dot(root[3:], root[3:])), 1.0, abs_tol=1.0e-4):
        raise ValueError("root quaternion must be unit length")
    return np.concatenate((root, body)).astype(np.float32)


def desired_body_q(current_body_q: Any, arm_target_rad: Any) -> np.ndarray:
    current = _finite_vector(current_body_q, 29, "current body joint position").copy()
    current[ARM_SLICE] = _finite_vector(arm_target_rad, 14, "arm target")
    return current


def demo_hand_value(opening_fraction: Any) -> np.ndarray:
    opening = _finite_vector(opening_fraction, 2, "Dex1 opening fraction")
    if np.any((opening < 0.0) | (opening > 1.0)):
        raise ValueError("Dex1 opening fraction lies outside [0,1]")
    return (opening * HAND_COMMAND_OPEN).astype(np.float32)


class G1EefForwardKinematics:
    """Pinned root-frame FK for the same wrist/tool labels as the source data."""

    def __init__(self, urdf_path: str | Path) -> None:
        import pinocchio as pin

        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        wrapper = pin.RobotWrapper.BuildFromURDF(str(path), package_dirs=[str(path.parent)])
        self.pin = pin
        self.model = wrapper.model
        self.data = wrapper.data
        missing_joints = [
            name for name in G1_BODY_JOINT_ORDER if not self.model.existJointName(name)
        ]
        missing_frames = [
            name for name in EEF_FRAME_NAMES.values() if not self.model.existFrame(name)
        ]
        if missing_joints or missing_frames:
            raise ValueError(
                f"URDF contract mismatch: missing_joints={missing_joints}, "
                f"missing_frames={missing_frames}"
            )
        self.joint_indices = np.asarray(
            [
                self.model.joints[self.model.getJointId(name)].idx_q
                for name in G1_BODY_JOINT_ORDER
            ],
            dtype=np.int64,
        )
        self.frame_ids = {
            side: int(self.model.getFrameId(name))
            for side, name in EEF_FRAME_NAMES.items()
        }
        self.urdf_path = path

    def __call__(self, body_joint_position_rad: Any) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        body = _finite_vector(body_joint_position_rad, 29, "FK body joint position")
        q = np.zeros(self.model.nq, dtype=np.float64)
        q[self.joint_indices] = body
        self.pin.framesForwardKinematics(self.model, self.data, q)
        values: list[float] = []
        for side in ("left", "right"):
            placement = self.data.oMf[self.frame_ids[side]]
            rotation = np.asarray(placement.rotation, dtype=np.float64)
            position = (
                np.asarray(placement.translation, dtype=np.float64)
                + rotation @ TOOL_TRANSLATION_M
            )
            euler = Rotation.from_matrix(rotation).as_euler("xyz")
            values.extend(position.tolist())
            values.extend(euler.tolist())
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (12,) or not np.isfinite(result).all():
            raise RuntimeError("G1 EEF FK produced an invalid value")
        return result


def numeric_features(
    frame: Mapping[str, Any],
    *,
    fk: G1EefForwardKinematics,
) -> dict[str, np.ndarray]:
    body = _finite_vector(frame["body_joint_position_rad"], 29, "body joint position")
    desired_body = desired_body_q(body, frame["commanded_arm_target_rad"])
    root = _finite_vector(frame["root_pose_xyzw"], 7, "root pose")
    result = {
        "observation.state.ee_state": fk(body),
        "observation.state.hand_state": demo_hand_value(frame["dex1_opening_state"]),
        "observation.state.robot_q_current": compose_robot_q(root, body),
        "action.ee_action": fk(desired_body),
        "action.hand_cmd": demo_hand_value(frame["commanded_dex1_opening_target"]),
        "action.robot_q_desired": compose_robot_q(root, desired_body),
    }
    if set(result) != set(NUMERIC_FEATURES):
        raise RuntimeError("numeric conversion does not match the source feature contract")
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        value = result[key]
        if value.shape != (width,) or value.dtype != np.float32:
            raise RuntimeError(f"converted {key} does not have float32[{width}]")
    return result


def _load_frames(path: Path, expected_count: int) -> tuple[dict[str, Any], ...]:
    frames = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict) or value.get("frame_index") != len(frames):
                raise ValueError("raw frame indices must be contiguous from zero")
            frames.append(value)
    if len(frames) != expected_count:
        raise ValueError("raw frame count differs from the episode manifest")
    return tuple(frames)


def convert_raw_episode(
    episode_root: str | Path,
    *,
    urdf_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write an atomic Parquet file containing policy-compatible numeric data."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(episode_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    trace_path = root / "frames.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RAW_EPISODE_SCHEMA_VERSION:
        raise ValueError("unsupported raw teleoperation episode")
    if manifest.get("success") is not True:
        raise ValueError("only successful teleoperation episodes may be converted")
    if manifest.get("privileged_policy_features") != []:
        raise ValueError("raw manifest contains privileged policy features")
    frame_count = int(manifest["frame_count"])
    frames = _load_frames(trace_path, frame_count)
    fk = G1EefForwardKinematics(urdf_path)
    converted = [numeric_features(frame, fk=fk) for frame in frames]

    fields = []
    arrays = []
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        fields.append(pa.field(key, pa.list_(pa.float32(), width)))
        arrays.append(
            pa.array([row[key].tolist() for row in converted], type=fields[-1].type)
        )
    fields.extend(
        (
            pa.field("timestamp", pa.float32()),
            pa.field("frame_index", pa.int64()),
        )
    )
    arrays.extend(
        (
            pa.array(
                [index / float(manifest["fps"]) for index in range(frame_count)],
                type=pa.float32(),
            ),
            pa.array(range(frame_count), type=pa.int64()),
        )
    )
    table = pa.Table.from_arrays(arrays, schema=pa.schema(fields))
    output = (
        root / "numeric.parquet"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False)
    os.replace(temporary, output)

    report = {
        "schema_version": NUMERIC_CONVERSION_SCHEMA_VERSION,
        "episode_id": manifest["episode_id"],
        "frame_count": frame_count,
        "fps": manifest["fps"],
        "raw_manifest_sha256": sha256_file(manifest_path),
        "raw_trace_sha256": sha256_file(trace_path),
        "urdf_sha256": sha256_file(fk.urdf_path),
        "urdf_path": str(fk.urdf_path),
        "numeric_feature_keys": list(NUMERIC_FEATURES),
        "action_alignment": (
            "camera/state at frame t -> desired target commanded from that same frame; "
            "servo-delayed applied targets remain diagnostic-only"
        ),
        "policy_camera_keys": manifest["policy_camera_keys"],
        "privileged_policy_features": [],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    report_path = output.with_suffix(".manifest.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
