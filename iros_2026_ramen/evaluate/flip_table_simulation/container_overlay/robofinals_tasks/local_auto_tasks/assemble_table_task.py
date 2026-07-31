"""Flip-table evaluation overlay for the organizer IKEA AssembleTableTask.

This file is bind-mounted over the container's original
`robofinals_tasks/local_auto_tasks/assemble_table_task.py`.

It preserves the registered class name `AssembleTableTask` so the existing
`Local-Task-AssembleTableTask` Gym registration continues to work, but changes
the reset state and success condition for flip_table evaluation.
"""

from __future__ import annotations

from dataclasses import MISSING
import json
import math
import os
from pathlib import Path

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

import robofinals.core.mdp as mdp
from .auto_condition_success_task import AutoConditionSuccessTask
from robofinals.utils import object_utils as OU
from robofinals.utils.isaac_data_compat import as_torch, sim_quat_raw_to_xyzw_torch


FLIP_TABLE_DATASET_INITIAL_UPPER_BODY_JOINT_POS = {
    # Median of frame_index=0 over Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1.
    # This keeps camera observations close to the real flip-table collection pose
    # instead of the organizer's near-neutral G1 reset pose.
    "waist_yaw_joint": 0.0682,
    "waist_roll_joint": -0.0216,
    "waist_pitch_joint": 0.0484,
    "left_shoulder_pitch_joint": -0.0066,
    "left_shoulder_roll_joint": 0.1687,
    "left_shoulder_yaw_joint": 0.2016,
    "left_elbow_joint": -0.1206,
    "left_wrist_roll_joint": -0.1366,
    "left_wrist_pitch_joint": 0.1505,
    "left_wrist_yaw_joint": -0.2500,
    "right_shoulder_pitch_joint": -0.0502,
    "right_shoulder_roll_joint": -0.2394,
    "right_shoulder_yaw_joint": 0.0639,
    "right_elbow_joint": -0.1060,
    "right_wrist_roll_joint": 0.2859,
    "right_wrist_pitch_joint": 0.2715,
    "right_wrist_yaw_joint": 0.6012,
}

FLIP_TABLE_DATASET_INITIAL_DEX1_FINGER_JOINT_POS = {
    # Convert frame_index=0 median hand_state [3.6998, 4.4868] from the
    # dataset's 0.0=closed, 4.5=open convention to Dex1 prismatic joint
    # positions using OPEN_POS=0.0245 and CLOSE_POS=-0.02.
    "left_dex1_finger_joint_1": 0.01659,
    "left_dex1_finger_joint_2": 0.01659,
    "right_dex1_finger_joint_1": 0.02437,
    "right_dex1_finger_joint_2": 0.02437,
}

FLIP_TABLE_UPPER_BODY_ACTION_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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

# This follows the dataset's ``robot_q_*[7:]`` ordering exactly.  It is used
# only by the explicit full-body controller-identification replay; normal
# evaluation and deployable policies retain organizer WBC ownership of legs,
# waist, and floating base.
FLIP_TABLE_FULL_BODY_ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    *FLIP_TABLE_UPPER_BODY_ACTION_JOINT_NAMES,
)

# Per-joint reset perturbations around the real dataset's initial-pose median.
# The ranges are deliberately modest: they vary the camera and reach posture
# without turning the reset into an arbitrary or self-colliding configuration.
FLIP_TABLE_UPPER_BODY_INITIAL_POSE_RANGES_RAD = {
    "waist_yaw_joint": 0.08,
    "waist_roll_joint": 0.04,
    "waist_pitch_joint": 0.06,
    "left_shoulder_pitch_joint": 0.12,
    "left_shoulder_roll_joint": 0.10,
    "left_shoulder_yaw_joint": 0.10,
    "left_elbow_joint": 0.12,
    "left_wrist_roll_joint": 0.10,
    "left_wrist_pitch_joint": 0.08,
    "left_wrist_yaw_joint": 0.10,
    "right_shoulder_pitch_joint": 0.12,
    "right_shoulder_roll_joint": 0.10,
    "right_shoulder_yaw_joint": 0.10,
    "right_elbow_joint": 0.12,
    "right_wrist_roll_joint": 0.10,
    "right_wrist_pitch_joint": 0.08,
    "right_wrist_yaw_joint": 0.10,
}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    result = float(default if value is None or value == "" else value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    result = int(default if value is None or value == "" else value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _calibration_table_pose_candidates() -> tuple[dict[str, object], ...] | None:
    """Read explicit workbench-local pose candidates for offline calibration.

    This is deliberately an evaluation-only reset aid.  The candidates are
    assigned by environment index once during reset; they are never exposed in
    observations or used to alter an episode after physics starts.
    """

    raw = os.environ.get("FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON", "").strip()
    if not raw:
        return None
    try:
        candidates = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON must be valid JSON") from exc
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON must be a non-empty list")
    if len(candidates) > 64:
        raise ValueError("at most 64 calibration table-pose candidates are supported")

    normalized: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"calibration table-pose candidate {index} must be an object")
        offset = candidate.get("offset_local_m", (0.0, 0.0, 0.0))
        yaw = candidate.get("yaw_rad", 0.0)
        if not isinstance(offset, (list, tuple)) or len(offset) != 3:
            raise ValueError(
                f"calibration table-pose candidate {index}.offset_local_m must have three values"
            )
        try:
            offset_values = tuple(float(value) for value in offset)
            yaw_value = float(yaw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration table-pose candidate {index} must be numeric") from exc
        if not all(math.isfinite(value) for value in (*offset_values, yaw_value)):
            raise ValueError(f"calibration table-pose candidate {index} must be finite")
        camera_offset = candidate.get("head_stereo_offset_local_m", (0.0, 0.0, 0.0))
        camera_rpy_deg = candidate.get("head_stereo_rotation_rpy_deg", (0.0, 0.0, 0.0))
        robot_root = candidate.get("robot_root_pos_local_m")
        robot_yaw = candidate.get("robot_root_yaw_rad")
        if not isinstance(camera_offset, (list, tuple)) or len(camera_offset) != 3:
            raise ValueError(
                f"calibration table-pose candidate {index}.head_stereo_offset_local_m must have three values"
            )
        if not isinstance(camera_rpy_deg, (list, tuple)) or len(camera_rpy_deg) != 3:
            raise ValueError(
                f"calibration table-pose candidate {index}.head_stereo_rotation_rpy_deg must have three values"
            )
        try:
            camera_offset_values = tuple(float(value) for value in camera_offset)
            camera_rpy_values = tuple(float(value) for value in camera_rpy_deg)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"calibration camera candidate {index} must be numeric") from exc
        if not all(math.isfinite(value) for value in (*camera_offset_values, *camera_rpy_values)):
            raise ValueError(f"calibration camera candidate {index} must be finite")
        if (robot_root is None) != (robot_yaw is None):
            raise ValueError(
                f"calibration candidate {index} must provide robot_root_pos_local_m and "
                "robot_root_yaw_rad together"
            )
        if robot_root is not None:
            if not isinstance(robot_root, (list, tuple)) or len(robot_root) != 3:
                raise ValueError(
                    f"calibration candidate {index}.robot_root_pos_local_m must have three values"
                )
            try:
                robot_root_values = tuple(float(value) for value in robot_root)
                robot_yaw_value = float(robot_yaw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"calibration robot candidate {index} must be numeric") from exc
            if not all(math.isfinite(value) for value in (*robot_root_values, robot_yaw_value)):
                raise ValueError(f"calibration robot candidate {index} must be finite")
        else:
            robot_root_values = None
            robot_yaw_value = None
        normalized.append(
            {
                "offset_local_m": offset_values,
                "yaw_rad": yaw_value,
                "head_stereo_offset_local_m": camera_offset_values,
                "head_stereo_rotation_rpy_deg": camera_rpy_values,
                "robot_root_pos_local_m": robot_root_values,
                "robot_root_yaw_rad": robot_yaw_value,
                "label": str(candidate.get("label", f"candidate_{index:03d}")),
            }
        )
    return tuple(normalized)


def _verbose_reset_logs() -> bool:
    return _env_bool("FLIP_TABLE_VERBOSE_RESET_LOGS", True)


def _env_range(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be '<low>,<high>', got {value!r}")
    if not all(math.isfinite(part) for part in parts) or parts[0] > parts[1]:
        raise ValueError(f"{name} must be a finite ordered range, got {value!r}")
    return parts[0], parts[1]


def _handle_randomization_failure(feature: str, exc: Exception) -> None:
    if _env_bool("FLIP_TABLE_STRICT_DOMAIN_RANDOMIZATION", True):
        raise RuntimeError(f"{feature} domain randomization failed") from exc
    print(f"[FlipTableEvalTask] {feature} randomization skipped: {exc}", flush=True)


def _rl_randomization_level() -> float:
    """Return the curriculum width in [0, 1]; regular evaluation uses full DR."""

    value = _env_float("FLIP_TABLE_RL_RANDOMIZATION_LEVEL", 1.0)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"FLIP_TABLE_RL_RANDOMIZATION_LEVEL must be in [0,1], got {value}")
    return value


def _curriculum_range(
    full_range: tuple[float, float],
    *,
    minimum_fraction: float = 0.10,
) -> tuple[float, float]:
    """Narrow a full randomization interval around its midpoint for early RL."""

    low, high = full_range
    midpoint = 0.5 * (low + high)
    fraction = max(minimum_fraction, _rl_randomization_level())
    half_width = 0.5 * (high - low) * fraction
    return midpoint - half_width, midpoint + half_width


def _curriculum_choices(values: tuple) -> tuple:
    """Expose progressively more discrete appearance choices."""

    count = max(1, min(len(values), math.ceil(len(values) * _rl_randomization_level())))
    return values[:count]


def _env_choices(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    choices = tuple(part.strip() for part in value.split(",") if part.strip())
    return choices or default


def _env_named_float_map(name: str) -> dict[str, float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return {}
    result: dict[str, float] = {}
    for item in value.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{name} entries must be '<name>=<float>', got {part!r}")
        key, raw = part.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{name} contains an empty joint name in {part!r}")
        parsed = float(raw.strip())
        if not math.isfinite(parsed):
            raise ValueError(f"{name} value for {key!r} must be finite")
        result[key] = parsed
    return result


def _env_tuple_or_none(name: str, expected_len: int) -> tuple[float, ...] | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated floats, got {value!r}")
    if not all(math.isfinite(part) for part in parts):
        raise ValueError(f"{name} must contain only finite values")
    return parts


def _env_float_vector(name: str, expected_len: int) -> tuple[float, ...] | None:
    """Read an explicit diagnostic action vector without silently truncating it."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    parts = tuple(float(part.strip()) for part in value.split(","))
    if len(parts) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} comma-separated floats, got {value!r}")
    if not all(math.isfinite(part) for part in parts):
        raise ValueError(f"{name} must contain only finite values")
    return parts


def _env_axis_xy(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    aliases = {
        "+x": (1.0, 0.0),
        "x": (1.0, 0.0),
        "-x": (-1.0, 0.0),
        "+y": (0.0, 1.0),
        "y": (0.0, 1.0),
        "-y": (0.0, -1.0),
    }
    normalized = value.strip().lower().replace(" ", "")
    if normalized in aliases:
        return aliases[normalized]
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be '+x', '-x', '+y', '-y', or '<x>,<y>', got {value!r}")
    norm = math.hypot(parts[0], parts[1])
    if norm < 1.0e-6:
        raise ValueError(f"{name} must not be a zero vector")
    return parts[0] / norm, parts[1] / norm


def _yaw_quat_xyzw(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack([zeros, zeros, torch.sin(half), torch.cos(half)], dim=-1)


def _rpy_quat_xyzw(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert intrinsic XYZ roll/pitch/yaw angles to scalar-last quaternions."""

    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    return torch.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dim=-1,
    )


def _yaw_from_quat_xyzw(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., :3] *= -1.0
    return result


def _normalize_xy(v: torch.Tensor) -> torch.Tensor:
    xy = v[..., :2]
    norm = torch.linalg.norm(xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    return xy / norm


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if high == low:
        return torch.full(shape, low, device=device, dtype=dtype)
    return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)


def _surface_values_for_average_pairs(
    hand_white: float,
    white_workbench: float,
    workbench_hand: float,
) -> tuple[float, float, float]:
    """Recover hand/white/workbench values from three pairwise averages."""

    return (
        hand_white + workbench_hand - white_workbench,
        hand_white + white_workbench - workbench_hand,
        white_workbench + workbench_hand - hand_white,
    )


G1_MATERIAL_COLORS: dict[str, tuple[float, float, float]] = {
    # Values from the official G1_GRIPPER.usd material table.  This fallback
    # is opt-in only; normal evaluation keeps the organizer USD bindings.
    "dark": (0.2, 0.2, 0.2),
    "white": (0.7, 0.7, 0.7),
    "dex1_base": (0.41176, 0.41176, 0.41176),
    "dex1_finger_1": (0.79216, 0.81961, 0.93333),
    "dex1_finger_2": (0.89804, 0.91765, 0.92941),
}

G1_DARK_LINK_TOKENS = (
    "pelvis",
    "hip_pitch_link",
    "ankle_roll_link",
    "head_link",
    "logo_link",
    "d435_link",
    "camera",
)

ROOM_BACKGROUND_PALETTES: tuple[dict[str, tuple[float, float, float] | str], ...] = (
    {
        "name": "neutral_lab",
        "floor": (0.46, 0.49, 0.50),
        "tile": (0.68, 0.70, 0.70),
        "wall": (0.78, 0.81, 0.80),
        "accent": (0.50, 0.57, 0.62),
    },
    {
        "name": "warm_room",
        "floor": (0.48, 0.36, 0.24),
        "tile": (0.64, 0.52, 0.38),
        "wall": (0.82, 0.77, 0.68),
        "accent": (0.54, 0.62, 0.52),
    },
    {
        "name": "warehouse",
        "floor": (0.35, 0.36, 0.34),
        "tile": (0.53, 0.54, 0.50),
        "wall": (0.62, 0.64, 0.62),
        "accent": (0.43, 0.45, 0.48),
    },
    {
        "name": "training_room",
        "floor": (0.28, 0.34, 0.32),
        "tile": (0.45, 0.53, 0.50),
        "wall": (0.72, 0.74, 0.70),
        "accent": (0.60, 0.50, 0.46),
    },
    {
        "name": "light_tile_room",
        "floor": (0.66, 0.66, 0.62),
        "tile": (0.42, 0.43, 0.42),
        "wall": (0.86, 0.86, 0.82),
        "accent": (0.58, 0.64, 0.70),
    },
)

ROOM_FLOOR_PATTERNS: tuple[str, ...] = ("grid", "checker", "planks", "border")
ROOM_WALL_PATTERNS: tuple[str, ...] = (
    "plain",
    "baseboard",
    "horizontal_stripes",
    "vertical_panels",
    "wainscot",
)

ROOM_FLOOR_MATERIALS: tuple[tuple[str, str, float, float], ...] = (
    ("oak_wood", "oak_wood.png", 0.58, 1.15),
    ("rough_concrete", "rough_concrete.png", 0.90, 1.35),
    ("ceramic_tile", "ceramic_tile.png", 0.42, 1.00),
    ("industrial_vinyl", "industrial_vinyl.png", 0.70, 1.20),
)

ROOM_WALL_MATERIALS: tuple[tuple[str, str, float, float], ...] = (
    ("painted_plaster", "painted_plaster.png", 0.82, 1.40),
    ("rough_concrete", "rough_concrete.png", 0.90, 1.25),
    ("red_brick", "red_brick.png", 0.88, 0.90),
    ("oak_panels", "oak_wood.png", 0.62, 1.10),
)

ROOM_PROP_ASSETS: tuple[tuple[str, float], ...] = (
    ("Chair", 0.45),
    ("Desk", 0.75),
    ("Shelf", 0.60),
    ("Cabinet", 0.58),
    ("Crates", 0.55),
    ("Plant", 0.45),
)


def _g1_material_key_for_path(path: str) -> str | None:
    lowered = path.lower()
    if "/robot/" not in lowered:
        return None
    if "/joints/" in lowered or lowered.endswith("_joint"):
        return None
    if "/collisions" in lowered:
        return None
    if "/visuals" not in lowered and "camera" not in lowered:
        return None
    if "pelvis_contour_link" in lowered:
        return "white"
    if "dex1_base_link" in lowered:
        return "dex1_base"
    if "dex1_finger_link_1" in lowered:
        return "dex1_finger_1"
    if "dex1_finger_link_2" in lowered:
        return "dex1_finger_2"
    if any(token in lowered for token in G1_DARK_LINK_TOKENS):
        return "dark"
    if "_link" in lowered or "/visuals" in lowered or "/collisions" in lowered:
        return "white"
    return None


def _flip_table_eval_reset_placeholder(env, env_ids=None) -> None:
    """Replaced by the task instance with bound methods in __init__."""


def _reset_scene_to_default_and_targets(env, env_ids) -> None:
    """Run the organizer reset and clear commands left by the prior episode."""
    mdp.reset_scene_to_default(env, env_ids)
    for articulation in env.scene.articulations.values():
        data = getattr(articulation, "data", None)
        if data is None:
            continue
        joint_pos = getattr(data, "default_joint_pos", None)
        joint_vel = getattr(data, "default_joint_vel", None)
        if joint_pos is not None and hasattr(articulation, "set_joint_position_target"):
            articulation.set_joint_position_target(
                as_torch(joint_pos)[env_ids].clone(),
                env_ids=env_ids,
            )
        if joint_vel is not None and hasattr(articulation, "set_joint_velocity_target"):
            articulation.set_joint_velocity_target(
                as_torch(joint_vel)[env_ids].clone(),
                env_ids=env_ids,
            )


@configclass
class FlipTableEvalEventCfg:
    init_task: EventTerm = MISSING
    flip_table_prepare_robot_reset = EventTerm(func=_flip_table_eval_reset_placeholder, mode="reset")
    reset_all = EventTerm(func=_reset_scene_to_default_and_targets, mode="reset")
    flip_table_eval_reset = EventTerm(func=_flip_table_eval_reset_placeholder, mode="reset")


class AssembleTableTask(AutoConditionSuccessTask):
    """Flip-table task using the organizer IKEA scene and registration sites."""

    task_name: str = "FlipTableEvalTask"

    leg_reg_int_sites: tuple[tuple[str, str, str], ...] = (
        ("Leg001_Leg001", "Leg001/Leg001", "Leg001/Leg001/Sites/reg_int1"),
        ("Leg001_01_Leg001", "Leg001_01/Leg001", "Leg001_01/Leg001/Sites/reg_int1"),
        ("Leg001_03_Leg001", "Leg001_03/Leg001", "Leg001_03/Leg001/Sites/reg_int1"),
        ("Leg001_06_Leg001", "Leg001_06/Leg001", "Leg001_06/Leg001/Sites/reg_int1"),
    )
    table_reg_int_sites: tuple[tuple[str, str, str], ...] = (
        ("Table001_Table001_01", "Table001/Table001_01", "Table001/Table001_01/Sites/reg_int1"),
        ("Table001_Table001_01", "Table001/Table001_01", "Table001/Table001_01/Sites/reg_int2"),
        ("Table001_Table001_01", "Table001/Table001_01", "Table001/Table001_01/Sites/reg_int3"),
        ("Table001_Table001_01", "Table001/Table001_01", "Table001/Table001_01/Sites/reg_int4"),
    )
    workbench_prim_path: str = "Table278"

    gripper_leg_distance_threshold: float = 0.25
    _start_success_check_count: int = 20

    def __init__(self):
        super().__init__()
        self._sim_body_mode = os.environ.get(
            "FLIP_TABLE_SIM_BODY_MODE", "balanced_wbc"
        ).strip().lower()
        if self._sim_body_mode not in {
            "balanced_wbc",
            "fixed_diagnostic",
            "full_body_diagnostic",
        }:
            raise ValueError(
                f"unsupported FLIP_TABLE_SIM_BODY_MODE={self._sim_body_mode!r}"
            )
        if self._sim_body_mode == "balanced_wbc":
            forbidden = [
                name
                for name in (
                    "FLIP_TABLE_LOCK_LOWER_BODY",
                    "FLIP_TABLE_LOCK_ROBOT_ROOT",
                    "FLIP_TABLE_FIX_ROOT_LINK",
                    "FLIP_TABLE_REQUIRE_WAIST_LOCK",
                )
                if _env_bool(name, False)
            ]
            if forbidden:
                raise RuntimeError(
                    "balanced_wbc forbids root/lower-body/waist locks: "
                    + ", ".join(forbidden)
                )
        self.events_cfg = FlipTableEvalEventCfg(init_task=self.events_cfg.init_task)
        self.events_cfg.flip_table_prepare_robot_reset = EventTerm(
            func=self._prepare_robot_default_pose_for_reset,
            mode="reset",
        )
        self.events_cfg.flip_table_eval_reset = EventTerm(func=self._reset_flip_table_scene, mode="reset")
        self.resample_robot_placement_on_reset = False
        self._success_debug_step = 0
        self._runtime_contact_debug_printed = False
        self._runtime_table_dynamics_debug_printed = False
        self._runtime_table_joint_debug_printed = False
        self._initial_table_normal = torch.zeros(
            (self.context.num_envs, 3),
            dtype=torch.float32,
            device=self.context.device,
        )
        self._initial_table_pos = torch.zeros(
            (self.context.num_envs, 3),
            dtype=torch.float32,
            device=self.context.device,
        )
        self._stable_success_streak = torch.zeros(
            self.context.num_envs,
            dtype=torch.long,
            device=self.context.device,
        )
        self._stable_success_result = torch.zeros(
            self.context.num_envs,
            dtype=torch.bool,
            device=self.context.device,
        )
        self._stable_success_previous_candidate = torch.zeros_like(
            self._stable_success_result
        )
        self._assembled_table_joints_created = False
        self._base_table_pos_local = None
        self._base_table_quat = None
        self._base_leg_positions_local = None
        self._base_leg_quats = None
        self._base_workbench_pos_local = None
        self._base_workbench_quat = None
        self._lower_body_lock_joint_ids = None
        self._lower_body_lock_joint_names = ()
        self._lower_body_lock_joint_pos = None
        self._lower_body_lock_root_pose = None
        self._lower_body_lock_logged = False
        self._robot_visual_materials_applied = False
        self._contact_material_collision_shape_counts = None
        self._policy_camera_mount_defaults = {}
        self._object_mass_defaults = {}
        self._upper_body_joint_property_defaults = None
        # ``IdealPDActuator`` keeps its controller gains on the actuator
        # object, separately from the simulator joint-drive data.  Preserve a
        # per-actuator copy so reset-time identification can restore a clean
        # baseline before applying an episode-fixed scale.
        self._upper_body_explicit_actuator_property_defaults = None
        self._teleop_randomization_samples: dict[int, dict[str, object]] = {}

    def _teleop_randomization(self, env_id: int) -> dict[str, object]:
        """Return the diagnostic-only randomization record for one environment."""

        return self._teleop_randomization_samples.setdefault(
            int(env_id),
            {
                "profile_level": _rl_randomization_level(),
                "policy_uses_privileged_values": False,
            },
        )

    def _sample_contact_material_parameters(self, env) -> dict[str, dict[str, object]]:
        pair_ranges = {
            "hand_white": {
                "static_friction": _env_range(
                    "FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE", _curriculum_range((0.65, 0.95))
                ),
                "dynamic_friction": _env_range(
                    "FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE", _curriculum_range((0.48, 0.64))
                ),
                "restitution": _env_range(
                    "FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE", _curriculum_range((0.02, 0.08))
                ),
            },
            "white_workbench": {
                "static_friction": _env_range(
                    "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE", _curriculum_range((0.50, 0.75))
                ),
                "dynamic_friction": _env_range(
                    "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE", _curriculum_range((0.35, 0.46))
                ),
                "restitution": _env_range(
                    "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE", _curriculum_range((0.01, 0.05))
                ),
            },
            "workbench_hand": {
                "static_friction": _env_range(
                    "FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE", _curriculum_range((0.60, 0.90))
                ),
                "dynamic_friction": _env_range(
                    "FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE", _curriculum_range((0.42, 0.56))
                ),
                "restitution": _env_range(
                    "FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE", _curriculum_range((0.02, 0.08))
                ),
            },
        }
        pair_names = ("hand_white", "white_workbench", "workbench_hand")
        properties = ("static_friction", "dynamic_friction", "restitution")
        for pair_name in pair_names:
            for property_name in properties:
                low, high = pair_ranges[pair_name][property_name]
                if low < 0.0 or high < low:
                    raise ValueError(
                        f"Invalid contact range for {pair_name}.{property_name}: {(low, high)}"
                    )

        for _attempt in range(128):
            pair_values: dict[str, dict[str, float]] = {name: {} for name in pair_names}
            for pair_name in pair_names:
                dynamic_low, dynamic_high = pair_ranges[pair_name]["dynamic_friction"]
                dynamic = float(_uniform((1,), dynamic_low, dynamic_high, device=env.device)[0])
                static_low, static_high = pair_ranges[pair_name]["static_friction"]
                static_low = max(static_low, dynamic + 0.04)
                if static_low > static_high:
                    break
                pair_values[pair_name]["dynamic_friction"] = dynamic
                pair_values[pair_name]["static_friction"] = float(
                    _uniform((1,), static_low, static_high, device=env.device)[0]
                )
                restitution_low, restitution_high = pair_ranges[pair_name]["restitution"]
                pair_values[pair_name]["restitution"] = float(
                    _uniform((1,), restitution_low, restitution_high, device=env.device)[0]
                )
            else:
                surfaces: dict[str, dict[str, float]] = {"hand": {}, "white": {}, "workbench": {}}
                for property_name in properties:
                    hand, white, workbench = _surface_values_for_average_pairs(
                        pair_values["hand_white"][property_name],
                        pair_values["white_workbench"][property_name],
                        pair_values["workbench_hand"][property_name],
                    )
                    for surface_name, value in zip(("hand", "white", "workbench"), (hand, white, workbench)):
                        surfaces[surface_name][property_name] = value

                valid = True
                for surface_values in surfaces.values():
                    static = surface_values["static_friction"]
                    dynamic = surface_values["dynamic_friction"]
                    restitution = surface_values["restitution"]
                    valid &= 0.15 <= static <= 1.20
                    valid &= 0.10 <= dynamic <= 1.00
                    valid &= static >= dynamic + 0.02
                    valid &= 0.0 <= restitution <= 0.20
                if valid:
                    return {"pairs": pair_values, "surfaces": surfaces}

        raise RuntimeError("Could not sample a physically consistent contact-material triplet")

    def _randomize_contact_materials(self, env, env_ids: torch.Tensor) -> None:
        if not _env_bool("FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS", True):
            return

        try:
            from pxr import Usd, UsdPhysics, UsdShade

            stage = env.sim.stage
            samples = []
            for env_id in env_ids.tolist():
                parameters = self._sample_contact_material_parameters(env)
                material_suffixes = {
                    "hand": "Robot/Looks/flip_table_contact_hand",
                    # The prepared scene authors these shared materials beneath
                    # /World/Looks. They are not duplicated below Scene when
                    # Isaac Lab clones the single-environment task.
                    "white": "Looks/flip_table_contact_white",
                    "workbench": "Looks/flip_table_contact_workbench",
                }
                for surface_name, values in parameters["surfaces"].items():
                    material_prim = self._find_prim_by_suffix(
                        env,
                        material_suffixes[surface_name],
                        env_id=env_id,
                    )
                    if material_prim is None:
                        raise RuntimeError(f"Missing pre-bound {surface_name} contact material")
                    for attribute_name, value in (
                        ("physics:staticFriction", values["static_friction"]),
                        ("physics:dynamicFriction", values["dynamic_friction"]),
                        ("physics:restitution", values["restitution"]),
                    ):
                        attribute = material_prim.GetAttribute(attribute_name)
                        if not attribute:
                            raise RuntimeError(
                                f"Missing {attribute_name} on contact material {material_prim.GetPath()}"
                            )
                        attribute.Set(float(value))
                    material_prim.GetAttribute("physxMaterial:frictionCombineMode").Set("average")
                    material_prim.GetAttribute("physxMaterial:restitutionCombineMode").Set("average")

                target_suffixes = {
                    "hand": (
                        "Robot/left_wrist_yaw_link/collisions",
                        "Robot/right_wrist_yaw_link/collisions",
                        "Robot/left_dex1_finger_link_1/collisions",
                        "Robot/left_dex1_finger_link_2/collisions",
                        "Robot/right_dex1_finger_link_1/collisions",
                        "Robot/right_dex1_finger_link_2/collisions",
                    ),
                    "white": (
                        "Scene/Table001/Table001_01",
                        "Scene/Leg001/Leg001",
                        "Scene/Leg001_01/Leg001",
                        "Scene/Leg001_03/Leg001",
                        "Scene/Leg001_06/Leg001",
                    ),
                    # Collider5 is the 1.80 x 0.75 m workbench top. Colliders1-4
                    # are legs and Collider6 is the narrower underside support.
                    "workbench": (
                        "Scene/Table278/Table278/Collisions/Table278_Collider5",
                    ),
                }
                bound_counts = {}
                missing = []
                for surface_name, suffixes in target_suffixes.items():
                    bound = 0
                    for suffix in suffixes:
                        prim = self._find_prim_by_suffix(env, suffix, env_id=env_id)
                        if prim is None:
                            missing.append(suffix)
                            continue
                        computed = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                        bound_material = computed[0] if computed else None
                        expected_material_name = material_suffixes[surface_name].rsplit("/", 1)[-1]
                        if not bound_material or not str(bound_material.GetPath()).endswith(
                            expected_material_name
                        ):
                            raise RuntimeError(
                                f"Unexpected physics material on {prim.GetPath()}: "
                                f"{bound_material.GetPath() if bound_material else '(none)'}"
                            )
                        bound += 1
                    bound_counts[surface_name] = bound
                if missing:
                    raise RuntimeError("Missing contact-material prims: " + ", ".join(missing))

                if self._contact_material_collision_shape_counts is None:
                    collision_shape_counts = {"hand": 0, "white": 0, "workbench": 0}
                    env_prefix = f"/World/envs/env_{env_id}/"
                    hand_tokens = (
                        "/Robot/left_wrist_yaw_link/collisions/",
                        "/Robot/right_wrist_yaw_link/collisions/",
                        "/Robot/left_dex1_finger_link_1/collisions/",
                        "/Robot/left_dex1_finger_link_2/collisions/",
                        "/Robot/right_dex1_finger_link_1/collisions/",
                        "/Robot/right_dex1_finger_link_2/collisions/",
                    )
                    white_tokens = (
                        "/Scene/Table001/Table001_01/",
                        "/Scene/Leg001/Leg001/",
                        "/Scene/Leg001_01/Leg001/",
                        "/Scene/Leg001_03/Leg001/",
                        "/Scene/Leg001_06/Leg001/",
                    )
                    workbench_top_token = "/Scene/Table278/Table278/Collisions/Table278_Collider5"
                    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
                        path = str(prim.GetPath())
                        if env_prefix not in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
                            continue
                        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                            continue
                        surface_name = None
                        if any(token in path for token in hand_tokens):
                            surface_name = "hand"
                        elif any(token in path for token in white_tokens):
                            surface_name = "white"
                        elif path.endswith(workbench_top_token):
                            surface_name = "workbench"
                        if surface_name is None:
                            continue
                        computed = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                        bound_material = computed[0] if computed else None
                        expected_suffix = material_suffixes[surface_name].rsplit("/", 1)[-1]
                        if not bound_material or not str(bound_material.GetPath()).endswith(expected_suffix):
                            raise RuntimeError(
                                f"Collision shape {path} has unexpected physics material: "
                                f"{bound_material.GetPath() if bound_material else '(none)'}"
                            )
                        collision_shape_counts[surface_name] += 1
                    if collision_shape_counts["hand"] < 8:
                        raise RuntimeError(f"Too few Dex1 collision shapes: {collision_shape_counts}")
                    expected_white_minimum = (
                        13
                        if _env_bool("FLIP_TABLE_SIMPLIFY_WHITE_COLLISION", False)
                        else 20
                    )
                    if collision_shape_counts["white"] < expected_white_minimum:
                        raise RuntimeError(f"Too few white-table collision shapes: {collision_shape_counts}")
                    if collision_shape_counts["workbench"] != 1:
                        raise RuntimeError(f"Wrong workbench-top collision count: {collision_shape_counts}")
                    self._contact_material_collision_shape_counts = collision_shape_counts

                samples.append(
                    {
                        "env": env_id,
                        "combine_mode": "average",
                        "pairs": {
                            pair_name: {key: round(value, 4) for key, value in values.items()}
                            for pair_name, values in parameters["pairs"].items()
                        },
                        "surfaces": {
                            surface_name: {key: round(value, 4) for key, value in values.items()}
                            for surface_name, values in parameters["surfaces"].items()
                        },
                        "bound_counts": bound_counts,
                        "collision_shape_counts": self._contact_material_collision_shape_counts,
                    }
                )
                self._teleop_randomization(env_id)["contact_materials"] = samples[-1]
            if _verbose_reset_logs():
                print(
                    f"[FlipTableEvalTask] contact material randomization: {samples}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[FlipTableEvalTask] contact material randomization failed: {exc}", flush=True)
            raise

    def _randomized_color(
        self,
        base_color: tuple[float, float, float],
        jitter: float,
        *,
        device: torch.device,
    ) -> tuple[float, float, float]:
        base = torch.tensor(base_color, dtype=torch.float32, device=device)
        if jitter > 0.0:
            base += _uniform((3,), -jitter, jitter, device=device)
        return tuple(float(value) for value in torch.clamp(base, 0.02, 0.95).detach().cpu().tolist())

    def _set_cube_transform(self, prim, center: tuple[float, float, float], scale: tuple[float, float, float]) -> None:
        from pxr import Gf, UsdGeom

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale and scale_op is None:
                scale_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if scale_op is None:
            scale_op = xformable.AddScaleOp()
        xformable.SetXformOpOrder([translate_op, scale_op])
        translate_op.Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2])))
        scale_op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))

    def _define_visual_cube(
        self,
        stage,
        path: str,
        center: tuple[float, float, float],
        scale: tuple[float, float, float],
        material,
        color: tuple[float, float, float],
    ) -> None:
        from pxr import Gf, UsdGeom, UsdShade

        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        self._set_cube_transform(prim, center, scale)
        UsdGeom.Imageable(prim).MakeVisible()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    def _define_textured_material_at(
        self,
        stage,
        material_path: str,
        texture_path: Path,
        *,
        roughness: float,
    ):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(stage, material_path)
        surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0.5, 0.5, 0.5, 1.0))
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        return material

    def _define_textured_quad(
        self,
        stage,
        path: str,
        points: tuple[tuple[float, float, float], ...],
        material,
        repeat_u: float,
        repeat_v: float,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        st.Set(
            [
                Gf.Vec2f(0.0, 0.0),
                Gf.Vec2f(float(repeat_u), 0.0),
                Gf.Vec2f(float(repeat_u), float(repeat_v)),
                Gf.Vec2f(0.0, float(repeat_v)),
            ]
        )
        UsdGeom.Imageable(mesh.GetPrim()).MakeVisible()
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def _spawn_room_props(
        self,
        stage,
        env_id: int,
        center_x: float,
        center_y: float,
        room_half_x: float,
        room_half_y: float,
        robot_x: float,
        robot_y: float,
        env,
    ) -> list[dict[str, object]]:
        from pxr import Gf, Sdf, UsdGeom

        if not _env_bool("FLIP_TABLE_RANDOMIZE_ROOM_PROPS", True):
            return []
        asset_root = Path(
            os.environ.get("FLIP_TABLE_ROOM_ASSET_ROOT", "/workspace/flip_table_room_assets")
        )
        asset_file = asset_root / "room_props.usda"
        if not asset_file.is_file():
            print(f"[FlipTableEvalTask] room prop asset missing: {asset_file}", flush=True)
            return []

        count = max(0, int(_env_float("FLIP_TABLE_ROOM_PROP_SLOTS", 10)))
        visible_probability = min(1.0, max(0.0, _env_float("FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY", 0.62)))
        x_range = _env_range("FLIP_TABLE_ROOM_PROP_X_RANGE_M", (-4.8, 4.8))
        y_range = _env_range("FLIP_TABLE_ROOM_PROP_Y_RANGE_M", (-4.8, 4.8))
        yaw_range = _env_range("FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD", (-math.pi, math.pi))
        scale_range = _env_range("FLIP_TABLE_ROOM_PROP_SCALE_RANGE", (0.80, 1.18))
        safe_radius = _env_float("FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M", 2.20)
        min_separation = _env_float("FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M", 0.30)
        wall_clearance = _env_float("FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M", 0.20)
        front_min_distance = _env_float("FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M", 0.50)
        front_half_angle_deg = _env_float("FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG", 80.0)
        if not 0.0 < front_half_angle_deg <= 90.0:
            raise ValueError("FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG must be in (0, 90]")
        front_tan = math.tan(math.radians(front_half_angle_deg))
        heading_x, heading_y = _env_axis_xy("FLIP_TABLE_ROOM_PROP_FRONT_AXIS", (1.0, 0.0))
        prop_spec_by_name = dict(ROOM_PROP_ASSETS)
        prop_names = _env_choices("FLIP_TABLE_ROOM_PROP_ASSETS", tuple(prop_spec_by_name))
        unknown_props = set(prop_names) - set(prop_spec_by_name)
        if unknown_props:
            raise ValueError("Unknown FLIP_TABLE_ROOM_PROP_ASSETS: " + ", ".join(sorted(unknown_props)))
        prop_root = f"/World/envs/env_{env_id}/FlipTableEvalPropPool"
        UsdGeom.Xform.Define(stage, prop_root)
        samples: list[dict[str, object]] = []
        occupied: list[tuple[float, float, float]] = []

        for slot in range(count):
            asset_name = prop_names[int(torch.randint(len(prop_names), (1,), device=env.device)[0].item())]
            footprint = prop_spec_by_name[asset_name]
            visible = bool(float(torch.rand(1, device=env.device)[0]) < visible_probability)
            scale = float(_uniform((1,), scale_range[0], scale_range[1], device=env.device)[0])
            footprint_scaled = footprint * scale
            x_low = max(x_range[0], -room_half_x + footprint_scaled + wall_clearance)
            x_high = min(x_range[1], room_half_x - footprint_scaled - wall_clearance)
            y_low = max(y_range[0], -room_half_y + footprint_scaled + wall_clearance)
            y_high = min(y_range[1], room_half_y - footprint_scaled - wall_clearance)
            x = center_x
            y = center_y
            if x_low > x_high or y_low > y_high:
                visible = False
            for _attempt in range(64):
                if not visible:
                    break
                x = center_x + float(_uniform((1,), x_low, x_high, device=env.device)[0])
                y = center_y + float(_uniform((1,), y_low, y_high, device=env.device)[0])
                robot_delta_x = x - robot_x
                robot_delta_y = y - robot_y
                forward_distance = robot_delta_x * heading_x + robot_delta_y * heading_y
                lateral_distance = abs(-robot_delta_x * heading_y + robot_delta_y * heading_x)
                clears_front = (
                    forward_distance >= front_min_distance
                    and lateral_distance <= forward_distance * front_tan
                )
                clears_task = math.hypot(x - center_x, y - center_y) >= safe_radius + footprint_scaled
                clears_props = all(
                    math.hypot(x - other_x, y - other_y)
                    >= footprint_scaled + other_radius + min_separation
                    for other_x, other_y, other_radius in occupied
                )
                if clears_front and clears_task and clears_props:
                    occupied.append((x, y, footprint_scaled))
                    break
            else:
                visible = False
            yaw = float(_uniform((1,), yaw_range[0], yaw_range[1], device=env.device)[0])
            slot_path = f"{prop_root}/Slot_{slot:02d}"
            UsdGeom.Xform.Define(stage, slot_path)
            for candidate_name in prop_spec_by_name:
                candidate_path = f"{slot_path}/{candidate_name}"
                candidate_prim = stage.GetPrimAtPath(candidate_path)
                if not candidate_prim.IsValid():
                    candidate_prim = UsdGeom.Xform.Define(stage, candidate_path).GetPrim()
                    candidate_prim.GetReferences().AddReference(
                        str(asset_file),
                        Sdf.Path(f"/{candidate_name}"),
                    )
                UsdGeom.Imageable(candidate_prim).MakeInvisible()

            prim = stage.GetPrimAtPath(f"{slot_path}/{asset_name}")
            xformable = UsdGeom.Xformable(prim)
            translate = None
            rotate = None
            scale_op = None
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                    rotate = op
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
            if translate is None:
                translate = xformable.AddTranslateOp()
            if rotate is None:
                rotate = xformable.AddRotateZOp()
            if scale_op is None:
                scale_op = xformable.AddScaleOp()
            xformable.SetXformOpOrder([translate, rotate, scale_op])
            translate.Set(Gf.Vec3d(x, y, 0.0))
            rotate.Set(yaw * 180.0 / math.pi)
            scale_op.Set(Gf.Vec3f(scale, scale, scale))
            imageable = UsdGeom.Imageable(prim)
            imageable.MakeVisible() if visible else imageable.MakeInvisible()
            samples.append(
                {
                    "slot": slot,
                    "asset": asset_name,
                    "visible": visible,
                    "xy": (round(x, 2), round(y, 2)),
                    "yaw": round(yaw, 3),
                    "scale": round(scale, 2),
                    "forward_m": round(
                        (x - robot_x) * heading_x + (y - robot_y) * heading_y,
                        2,
                    ),
                }
            )
        return samples

    def _randomize_room_background(
        self,
        env,
        env_ids: torch.Tensor,
        workbench_pos_local: torch.Tensor | None,
        robot_pos_local: torch.Tensor,
    ) -> None:
        if not _env_bool("FLIP_TABLE_RANDOMIZE_ROOM", True):
            return

        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

            stage = env.sim.stage
            jitter = _env_float(
                "FLIP_TABLE_ROOM_COLOR_JITTER",
                0.01 + 0.07 * _rl_randomization_level(),
            )
            floor_half_range = _env_range(
                "FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M", _curriculum_range((5.5, 7.5))
            )
            wall_height_range = _env_range(
                "FLIP_TABLE_ROOM_WALL_HEIGHT_M", _curriculum_range((4.0, 5.5))
            )
            tile_size_range = _env_range(
                "FLIP_TABLE_ROOM_TILE_SIZE_M", _curriculum_range((0.35, 0.9))
            )
            tile_line_width_range = _env_range(
                "FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M", _curriculum_range((0.008, 0.025))
            )
            wall_thickness = _env_float("FLIP_TABLE_ROOM_WALL_THICKNESS_M", 0.05)
            floor_thickness = _env_float("FLIP_TABLE_ROOM_FLOOR_THICKNESS_M", 0.02)
            floor_z = _env_float("FLIP_TABLE_ROOM_FLOOR_Z_M", 0.001)
            tile_z = floor_z + 0.003
            max_tile_lines = max(0, int(_env_float("FLIP_TABLE_ROOM_MAX_TILE_LINES", 96)))
            max_pattern_prims = max(0, int(_env_float("FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS", 96)))
            floor_pattern_choices = _env_choices(
                "FLIP_TABLE_ROOM_FLOOR_PATTERNS", _curriculum_choices(ROOM_FLOOR_PATTERNS)
            )
            wall_pattern_choices = _env_choices(
                "FLIP_TABLE_ROOM_WALL_PATTERNS", _curriculum_choices(ROOM_WALL_PATTERNS)
            )
            floor_spec_by_name = {spec[0]: spec for spec in ROOM_FLOOR_MATERIALS}
            wall_spec_by_name = {spec[0]: spec for spec in ROOM_WALL_MATERIALS}
            floor_material_names = _env_choices(
                "FLIP_TABLE_ROOM_FLOOR_MATERIALS",
                _curriculum_choices(tuple(floor_spec_by_name)),
            )
            wall_material_names = _env_choices(
                "FLIP_TABLE_ROOM_WALL_MATERIALS",
                _curriculum_choices(tuple(wall_spec_by_name)),
            )
            unknown_floor = set(floor_material_names) - set(floor_spec_by_name)
            unknown_wall = set(wall_material_names) - set(wall_spec_by_name)
            if unknown_floor or unknown_wall:
                raise ValueError(
                    "Unknown room material names: "
                    + ", ".join(sorted(unknown_floor | unknown_wall))
                )

            for row, env_id in enumerate(env_ids.tolist()):
                palette_count = len(_curriculum_choices(ROOM_BACKGROUND_PALETTES))
                palette_index = int(torch.randint(palette_count, (1,), device=env.device)[0].item())
                palette = ROOM_BACKGROUND_PALETTES[palette_index]
                floor_pattern = floor_pattern_choices[
                    int(torch.randint(len(floor_pattern_choices), (1,), device=env.device)[0].item())
                ]
                wall_pattern = wall_pattern_choices[
                    int(torch.randint(len(wall_pattern_choices), (1,), device=env.device)[0].item())
                ]
                floor_color = self._randomized_color(palette["floor"], jitter, device=env.device)
                tile_color = self._randomized_color(palette["tile"], jitter * 0.6, device=env.device)
                wall_color = self._randomized_color(palette["wall"], jitter, device=env.device)
                accent_color = self._randomized_color(palette["accent"], jitter, device=env.device)
                floor_material_name = floor_material_names[
                    int(torch.randint(len(floor_material_names), (1,), device=env.device)[0].item())
                ]
                wall_material_name = wall_material_names[
                    int(torch.randint(len(wall_material_names), (1,), device=env.device)[0].item())
                ]
                floor_material_spec = floor_spec_by_name[floor_material_name]
                wall_material_spec = wall_spec_by_name[wall_material_name]

                center_x = 0.0
                center_y = 0.0
                if workbench_pos_local is not None:
                    center_x = float(workbench_pos_local[row, 0].item())
                    center_y = float(workbench_pos_local[row, 1].item())
                half_x = float(_uniform((1,), floor_half_range[0], floor_half_range[1], device=env.device)[0].item())
                half_y = float(_uniform((1,), floor_half_range[0], floor_half_range[1], device=env.device)[0].item())
                wall_height = float(_uniform((1,), wall_height_range[0], wall_height_range[1], device=env.device)[0].item())
                tile_size = float(_uniform((1,), tile_size_range[0], tile_size_range[1], device=env.device)[0].item())
                tile_line_width = float(
                    _uniform((1,), tile_line_width_range[0], tile_line_width_range[1], device=env.device)[0].item()
                )
                tile_offset_x = float(_uniform((1,), 0.0, tile_size, device=env.device)[0].item())
                tile_offset_y = float(_uniform((1,), 0.0, tile_size, device=env.device)[0].item())

                root_path = f"/World/envs/env_{env_id}/FlipTableEvalRoom"
                root_prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
                for prim in Usd.PrimRange(root_prim):
                    if prim == root_prim or not prim.IsA(UsdGeom.Imageable):
                        continue
                    UsdGeom.Imageable(prim).MakeInvisible()

                material_prefix = f"/World/Looks/flip_table_room_env_{env_id}"
                floor_material = self._define_preview_material_at(stage, f"{material_prefix}_floor", floor_color, roughness=0.82)
                tile_material = self._define_preview_material_at(stage, f"{material_prefix}_tile", tile_color, roughness=0.78)
                wall_material = self._define_preview_material_at(stage, f"{material_prefix}_wall", wall_color, roughness=0.70)
                accent_material = self._define_preview_material_at(stage, f"{material_prefix}_accent", accent_color, roughness=0.70)
                texture_root = Path(
                    os.environ.get("FLIP_TABLE_ROOM_ASSET_ROOT", "/workspace/flip_table_room_assets")
                ) / "textures"
                floor_texture = texture_root / floor_material_spec[1]
                wall_texture = texture_root / wall_material_spec[1]
                textured_floor_material = None
                textured_wall_material = None
                if floor_texture.is_file() and wall_texture.is_file():
                    textured_floor_material = self._define_textured_material_at(
                        stage,
                        f"{material_prefix}_floor_texture",
                        floor_texture,
                        roughness=floor_material_spec[2],
                    )
                    textured_wall_material = self._define_textured_material_at(
                        stage,
                        f"{material_prefix}_wall_texture",
                        wall_texture,
                        roughness=wall_material_spec[2],
                    )

                self._define_visual_cube(
                    stage,
                    f"{root_path}/Floor",
                    (center_x, center_y, floor_z - 0.5 * floor_thickness),
                    (2.0 * half_x, 2.0 * half_y, floor_thickness),
                    floor_material,
                    floor_color,
                )
                wall_z = 0.5 * wall_height
                self._define_visual_cube(
                    stage,
                    f"{root_path}/BackWall",
                    (center_x, center_y + half_y, wall_z),
                    (2.0 * half_x, wall_thickness, wall_height),
                    accent_material,
                    accent_color,
                )
                self._define_visual_cube(
                    stage,
                    f"{root_path}/FrontWall",
                    (center_x, center_y - half_y, wall_z),
                    (2.0 * half_x, wall_thickness, wall_height),
                    wall_material,
                    wall_color,
                )

                if textured_floor_material is not None and textured_wall_material is not None:
                    floor_repeat_m = floor_material_spec[3]
                    wall_repeat_m = wall_material_spec[3]
                    surface_eps = 0.006
                    self._define_textured_quad(
                        stage,
                        f"{root_path}/FloorTexture",
                        (
                            (center_x - half_x, center_y - half_y, floor_z + surface_eps),
                            (center_x + half_x, center_y - half_y, floor_z + surface_eps),
                            (center_x + half_x, center_y + half_y, floor_z + surface_eps),
                            (center_x - half_x, center_y + half_y, floor_z + surface_eps),
                        ),
                        textured_floor_material,
                        2.0 * half_x / floor_repeat_m,
                        2.0 * half_y / floor_repeat_m,
                    )
                    self._define_textured_quad(
                        stage,
                        f"{root_path}/BackWallTexture",
                        (
                            (center_x - half_x, center_y + half_y - wall_thickness, 0.0),
                            (center_x + half_x, center_y + half_y - wall_thickness, 0.0),
                            (center_x + half_x, center_y + half_y - wall_thickness, wall_height),
                            (center_x - half_x, center_y + half_y - wall_thickness, wall_height),
                        ),
                        textured_wall_material,
                        2.0 * half_x / wall_repeat_m,
                        wall_height / wall_repeat_m,
                    )
                    self._define_textured_quad(
                        stage,
                        f"{root_path}/FrontWallTexture",
                        (
                            (center_x + half_x, center_y - half_y + wall_thickness, 0.0),
                            (center_x - half_x, center_y - half_y + wall_thickness, 0.0),
                            (center_x - half_x, center_y - half_y + wall_thickness, wall_height),
                            (center_x + half_x, center_y - half_y + wall_thickness, wall_height),
                        ),
                        textured_wall_material,
                        2.0 * half_x / wall_repeat_m,
                        wall_height / wall_repeat_m,
                    )
                    self._define_textured_quad(
                        stage,
                        f"{root_path}/LeftWallTexture",
                        (
                            (center_x - half_x + wall_thickness, center_y - half_y, 0.0),
                            (center_x - half_x + wall_thickness, center_y + half_y, 0.0),
                            (center_x - half_x + wall_thickness, center_y + half_y, wall_height),
                            (center_x - half_x + wall_thickness, center_y - half_y, wall_height),
                        ),
                        textured_wall_material,
                        2.0 * half_y / wall_repeat_m,
                        wall_height / wall_repeat_m,
                    )
                    self._define_textured_quad(
                        stage,
                        f"{root_path}/RightWallTexture",
                        (
                            (center_x + half_x - wall_thickness, center_y + half_y, 0.0),
                            (center_x + half_x - wall_thickness, center_y - half_y, 0.0),
                            (center_x + half_x - wall_thickness, center_y - half_y, wall_height),
                            (center_x + half_x - wall_thickness, center_y + half_y, wall_height),
                        ),
                        textured_wall_material,
                        2.0 * half_y / wall_repeat_m,
                        wall_height / wall_repeat_m,
                    )

                window_visible = bool(
                    float(torch.rand(1, device=env.device)[0])
                    < _env_float("FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY", 0.72)
                )
                if window_visible:
                    window_color = self._randomized_color((0.42, 0.58, 0.70), 0.08, device=env.device)
                    window_material = self._define_preview_material_at(
                        stage,
                        f"{material_prefix}_window",
                        window_color,
                        roughness=0.18,
                        metallic=0.08,
                    )
                    window_w = float(_uniform((1,), 1.2, 2.1, device=env.device)[0])
                    window_h = float(_uniform((1,), 0.9, 1.5, device=env.device)[0])
                    window_z = float(_uniform((1,), 1.45, 2.25, device=env.device)[0])
                    self._define_visual_cube(
                        stage,
                        f"{root_path}/WindowGlass",
                        (center_x, center_y + half_y - wall_thickness - 0.012, window_z),
                        (window_w, 0.018, window_h),
                        window_material,
                        window_color,
                    )
                    for name, offset_x, offset_z, scale_x, scale_z in (
                        ("WindowFrameLeft", -0.5 * window_w, 0.0, 0.045, window_h + 0.10),
                        ("WindowFrameRight", 0.5 * window_w, 0.0, 0.045, window_h + 0.10),
                        ("WindowFrameTop", 0.0, 0.5 * window_h, window_w + 0.10, 0.045),
                        ("WindowFrameBottom", 0.0, -0.5 * window_h, window_w + 0.10, 0.045),
                        ("WindowMullion", 0.0, 0.0, 0.035, window_h),
                    ):
                        self._define_visual_cube(
                            stage,
                            f"{root_path}/{name}",
                            (
                                center_x + offset_x,
                                center_y + half_y - wall_thickness - 0.025,
                                window_z + offset_z,
                            ),
                            (scale_x, 0.035, scale_z),
                            accent_material,
                            accent_color,
                        )
                self._define_visual_cube(
                    stage,
                    f"{root_path}/LeftWall",
                    (center_x - half_x, center_y, wall_z),
                    (wall_thickness, 2.0 * half_y, wall_height),
                    wall_material,
                    wall_color,
                )
                self._define_visual_cube(
                    stage,
                    f"{root_path}/RightWall",
                    (center_x + half_x, center_y, wall_z),
                    (wall_thickness, 2.0 * half_y, wall_height),
                    wall_material,
                    wall_color,
                )

                floor_pattern_count = 0
                wall_pattern_count = 0
                floor_pattern_z = floor_z + 0.002
                floor_pattern_thickness = 0.003
                wall_pattern_depth = 0.012
                wall_pattern_eps = 0.003

                def add_floor_pattern(
                    name: str,
                    center: tuple[float, float, float],
                    scale: tuple[float, float, float],
                    material,
                    color: tuple[float, float, float],
                ) -> None:
                    nonlocal floor_pattern_count
                    if floor_pattern_count >= max_pattern_prims:
                        return
                    self._define_visual_cube(stage, f"{root_path}/{name}", center, scale, material, color)
                    floor_pattern_count += 1

                def add_wall_pattern(
                    name: str,
                    wall: str,
                    along_center: float,
                    along_size: float,
                    z_center: float,
                    z_size: float,
                    material,
                    color: tuple[float, float, float],
                ) -> None:
                    nonlocal wall_pattern_count
                    if wall_pattern_count >= max_pattern_prims:
                        return
                    if wall == "back":
                        center = (
                            along_center,
                            center_y + half_y - 0.5 * wall_thickness - 0.5 * wall_pattern_depth - wall_pattern_eps,
                            z_center,
                        )
                        scale = (along_size, wall_pattern_depth, z_size)
                    elif wall == "front":
                        center = (
                            along_center,
                            center_y - half_y + 0.5 * wall_thickness + 0.5 * wall_pattern_depth + wall_pattern_eps,
                            z_center,
                        )
                        scale = (along_size, wall_pattern_depth, z_size)
                    elif wall == "left":
                        center = (
                            center_x - half_x + 0.5 * wall_thickness + 0.5 * wall_pattern_depth + wall_pattern_eps,
                            along_center,
                            z_center,
                        )
                        scale = (wall_pattern_depth, along_size, z_size)
                    elif wall == "right":
                        center = (
                            center_x + half_x - 0.5 * wall_thickness - 0.5 * wall_pattern_depth - wall_pattern_eps,
                            along_center,
                            z_center,
                        )
                        scale = (wall_pattern_depth, along_size, z_size)
                    else:
                        return
                    self._define_visual_cube(stage, f"{root_path}/{name}", center, scale, material, color)
                    wall_pattern_count += 1

                def add_full_wall_band(prefix: str, z_center: float, z_size: float, material, color) -> None:
                    add_wall_pattern(f"{prefix}_Back", "back", center_x, 2.0 * half_x, z_center, z_size, material, color)
                    add_wall_pattern(f"{prefix}_Front", "front", center_x, 2.0 * half_x, z_center, z_size, material, color)
                    add_wall_pattern(f"{prefix}_Left", "left", center_y, 2.0 * half_y, z_center, z_size, material, color)
                    add_wall_pattern(f"{prefix}_Right", "right", center_y, 2.0 * half_y, z_center, z_size, material, color)

                if floor_pattern == "checker":
                    cell_target = max(0.85, tile_size * 2.2)
                    cells_x = max(2, min(12, int(math.ceil((2.0 * half_x) / cell_target))))
                    cells_y = max(2, min(12, int(math.ceil((2.0 * half_y) / cell_target))))
                    cell_x = (2.0 * half_x) / cells_x
                    cell_y = (2.0 * half_y) / cells_y
                    for ix in range(cells_x):
                        for iy in range(cells_y):
                            if (ix + iy) % 2 != 0:
                                continue
                            add_floor_pattern(
                                f"FloorChecker_{ix:02d}_{iy:02d}",
                                (
                                    center_x - half_x + (ix + 0.5) * cell_x,
                                    center_y - half_y + (iy + 0.5) * cell_y,
                                    floor_pattern_z,
                                ),
                                (0.92 * cell_x, 0.92 * cell_y, floor_pattern_thickness),
                                accent_material,
                                accent_color,
                            )
                elif floor_pattern == "planks":
                    along_x = bool(int(torch.randint(2, (1,), device=env.device)[0].item()))
                    plank_count = max(4, min(18, int(math.ceil((2.0 * (half_y if along_x else half_x)) / tile_size))))
                    plank_width = (2.0 * (half_y if along_x else half_x)) / plank_count
                    for index in range(plank_count):
                        if index % 2 != 0:
                            continue
                        if along_x:
                            add_floor_pattern(
                                f"FloorPlank_{index:02d}",
                                (center_x, center_y - half_y + (index + 0.5) * plank_width, floor_pattern_z),
                                (2.0 * half_x, 0.86 * plank_width, floor_pattern_thickness),
                                accent_material,
                                accent_color,
                            )
                        else:
                            add_floor_pattern(
                                f"FloorPlank_{index:02d}",
                                (center_x - half_x + (index + 0.5) * plank_width, center_y, floor_pattern_z),
                                (0.86 * plank_width, 2.0 * half_y, floor_pattern_thickness),
                                accent_material,
                                accent_color,
                            )
                elif floor_pattern == "border":
                    border_width = min(0.28, max(0.08, tile_size * 0.35))
                    add_floor_pattern(
                        "FloorBorder_Back",
                        (center_x, center_y + half_y - 0.5 * border_width, floor_pattern_z),
                        (2.0 * half_x, border_width, floor_pattern_thickness),
                        accent_material,
                        accent_color,
                    )
                    add_floor_pattern(
                        "FloorBorder_Front",
                        (center_x, center_y - half_y + 0.5 * border_width, floor_pattern_z),
                        (2.0 * half_x, border_width, floor_pattern_thickness),
                        accent_material,
                        accent_color,
                    )
                    add_floor_pattern(
                        "FloorBorder_Left",
                        (center_x - half_x + 0.5 * border_width, center_y, floor_pattern_z),
                        (border_width, 2.0 * half_y, floor_pattern_thickness),
                        accent_material,
                        accent_color,
                    )
                    add_floor_pattern(
                        "FloorBorder_Right",
                        (center_x + half_x - 0.5 * border_width, center_y, floor_pattern_z),
                        (border_width, 2.0 * half_y, floor_pattern_thickness),
                        accent_material,
                        accent_color,
                    )

                if wall_pattern == "baseboard":
                    add_full_wall_band("WallBaseboard", 0.08, 0.16, accent_material, accent_color)
                elif wall_pattern == "horizontal_stripes":
                    stripe_count = int(torch.randint(2, 5, (1,), device=env.device)[0].item())
                    for index in range(stripe_count):
                        z_center = wall_height * float(index + 1) / float(stripe_count + 1)
                        z_size = float(_uniform((1,), 0.06, 0.16, device=env.device)[0].item())
                        add_full_wall_band(
                            f"WallStripe_{index:02d}",
                            z_center,
                            z_size,
                            accent_material,
                            accent_color,
                        )
                elif wall_pattern == "vertical_panels":
                    panel_width = float(_uniform((1,), 0.035, 0.08, device=env.device)[0].item())
                    panel_height = max(1.2, wall_height * 0.72)
                    panel_z = 0.5 * panel_height
                    panel_spacing = max(0.75, tile_size * 1.6)
                    panel_index = 0
                    x = center_x - half_x + panel_spacing
                    while x < center_x + half_x - 0.25 and wall_pattern_count < max_pattern_prims:
                        add_wall_pattern(
                            f"WallPanel_Back_{panel_index:02d}",
                            "back",
                            x,
                            panel_width,
                            panel_z,
                            panel_height,
                            accent_material,
                            accent_color,
                        )
                        add_wall_pattern(
                            f"WallPanel_Front_{panel_index:02d}",
                            "front",
                            x,
                            panel_width,
                            panel_z,
                            panel_height,
                            accent_material,
                            accent_color,
                        )
                        panel_index += 1
                        x += panel_spacing
                    y = center_y - half_y + panel_spacing
                    while y < center_y + half_y - 0.25 and wall_pattern_count < max_pattern_prims:
                        add_wall_pattern(
                            f"WallPanel_Left_{panel_index:02d}",
                            "left",
                            y,
                            panel_width,
                            panel_z,
                            panel_height,
                            accent_material,
                            accent_color,
                        )
                        add_wall_pattern(
                            f"WallPanel_Right_{panel_index:02d}",
                            "right",
                            y,
                            panel_width,
                            panel_z,
                            panel_height,
                            accent_material,
                            accent_color,
                        )
                        panel_index += 1
                        y += panel_spacing
                elif wall_pattern == "wainscot":
                    panel_height = min(max(0.75, wall_height * 0.28), 1.35)
                    add_full_wall_band(
                        "WallWainscot",
                        0.5 * panel_height,
                        panel_height,
                        accent_material,
                        accent_color,
                    )
                    add_full_wall_band(
                        "WallWainscotTrim",
                        panel_height + 0.04,
                        0.08,
                        tile_material,
                        tile_color,
                    )

                line_count = 0
                if floor_material_spec[0] == "ceramic_tile" or textured_floor_material is None:
                    x = center_x - half_x + tile_offset_x
                    while x <= center_x + half_x and line_count < max_tile_lines:
                        self._define_visual_cube(
                            stage,
                            f"{root_path}/TileLine_{line_count:03d}",
                            (x, center_y, tile_z),
                            (tile_line_width, 2.0 * half_y, 0.004),
                            tile_material,
                            tile_color,
                        )
                        line_count += 1
                        x += tile_size
                    y = center_y - half_y + tile_offset_y
                    while y <= center_y + half_y and line_count < max_tile_lines:
                        self._define_visual_cube(
                            stage,
                            f"{root_path}/TileLine_{line_count:03d}",
                            (center_x, y, tile_z),
                            (2.0 * half_x, tile_line_width, 0.004),
                            tile_material,
                            tile_color,
                        )
                        line_count += 1
                        y += tile_size

                prop_samples = self._spawn_room_props(
                    stage,
                    env_id,
                    center_x,
                    center_y,
                    half_x,
                    half_y,
                    float(robot_pos_local[row, 0]),
                    float(robot_pos_local[row, 1]),
                    env,
                )
                self._teleop_randomization(env_id)["room"] = {
                    "palette": str(palette["name"]),
                    "floor_material": floor_material_spec[0],
                    "wall_material": wall_material_spec[0],
                    "floor_pattern": floor_pattern,
                    "wall_pattern": wall_pattern,
                    "floor_half_extents_m": [half_x, half_y],
                    "wall_height_m": wall_height,
                    "tile_size_m": tile_size,
                    "tile_line_width_m": tile_line_width,
                    "window_visible": window_visible,
                    "props": prop_samples,
                }

                if _verbose_reset_logs():
                    print(
                        "[FlipTableEvalTask] room randomization: "
                        f"env={env_id}, palette={palette['name']}, "
                        f"floor_material={floor_material_spec[0]}, "
                        f"wall_material={wall_material_spec[0]}, "
                        f"floor_pattern={floor_pattern}, wall_pattern={wall_pattern}, "
                        f"half=({half_x:.2f},{half_y:.2f}), wall_h={wall_height:.2f}, "
                        f"tile={tile_size:.2f}, lines={line_count}, "
                        f"floor_decor={floor_pattern_count}, "
                        f"wall_decor={wall_pattern_count}, "
                        f"window={window_visible}, props={prop_samples}",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001
            _handle_randomization_failure("room", exc)

    def _find_prim_by_suffix(self, env, prim_path: str, env_id: int):
        from pxr import Usd

        stage = env.sim.stage
        suffix = prim_path.strip("/")
        suffixes = [suffix]
        if suffix.startswith("World/"):
            suffixes.append(suffix[len("World/"):])

        env_prefix = f"World/envs/env_{env_id}/"
        for suffix_item in suffixes:
            prim = stage.GetPrimAtPath(f"{env_prefix}{suffix_item}")
            if prim and prim.IsValid():
                return prim

        env_token = f"envs/env_{env_id}/"
        for candidate in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            candidate_path = str(candidate.GetPath()).strip("/")
            if env_token not in candidate_path:
                continue
            if any(candidate_path.endswith(item) for item in suffixes):
                return candidate

        if env.num_envs == 1:
            for candidate in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
                candidate_path = str(candidate.GetPath()).strip("/")
                if "envs/env_" in candidate_path:
                    continue
                if any(candidate_path.endswith(item) for item in suffixes):
                    return candidate
        return None

    def _get_site_local_offset(self, env, object_prim_path: str, site_prim_path: str) -> torch.Tensor | None:
        from pxr import UsdGeom

        cache_key = (object_prim_path, site_prim_path)
        if not hasattr(self, "_site_local_offset_cache"):
            self._site_local_offset_cache = {}
        if cache_key in self._site_local_offset_cache:
            return self._site_local_offset_cache[cache_key]

        object_prim = self._find_prim_by_suffix(env, object_prim_path, env_id=0)
        site_prim = self._find_prim_by_suffix(env, site_prim_path, env_id=0)
        if object_prim is None or site_prim is None:
            return None

        xform_cache = UsdGeom.XformCache()
        object_to_world = xform_cache.GetLocalToWorldTransform(object_prim)
        site_world_pos = xform_cache.GetLocalToWorldTransform(site_prim).ExtractTranslation()
        site_local_pos = object_to_world.GetInverse().Transform(site_world_pos)
        offset = torch.tensor(
            [float(site_local_pos[0]), float(site_local_pos[1]), float(site_local_pos[2])],
            dtype=torch.float32,
            device=env.device,
        )
        self._site_local_offset_cache[cache_key] = offset
        return offset

    def _find_scene_entity(self, env, name: str):
        containers = [env.scene]
        nested_scene = getattr(env.scene, "Scene", None)
        if nested_scene is not None:
            containers.append(nested_scene)
        try:
            nested_scene = env.scene["Scene"]
        except Exception:
            nested_scene = None
        if nested_scene is not None and nested_scene not in containers:
            containers.append(nested_scene)

        entity_maps = []
        for container in containers:
            entity_maps.extend(
                (
                    getattr(container, "rigid_objects", {}),
                    getattr(container, "articulations", {}),
                )
            )
        for entities in entity_maps:
            if name in entities:
                return entities[name]
            matches = [entity for entity_name, entity in entities.items() if str(entity_name).endswith(name)]
            if len(matches) == 1:
                return matches[0]
        return None

    def _scene_entity_names_for_log(self, env) -> str:
        parts = []
        containers = [env.scene]
        nested_scene = getattr(env.scene, "Scene", None)
        if nested_scene is not None:
            containers.append(nested_scene)
        for container_index, container in enumerate(containers):
            container_label = "root" if container_index == 0 else "Scene"
            for label, entities in (
                ("rigid", getattr(container, "rigid_objects", {})),
                ("articulation", getattr(container, "articulations", {})),
            ):
                names = sorted(str(name) for name in getattr(entities, "keys", lambda: [])())
                if names:
                    preview = ", ".join(names[:40])
                    if len(names) > 40:
                        preview += f", ... ({len(names)} total)"
                    parts.append(f"{container_label}.{label}=[{preview}]")
        return "; ".join(parts) if parts else "no scene entities exposed"

    def _asset_root_prim_path(self, object_prim_path: str) -> str:
        return object_prim_path.strip("/").split("/", 1)[0]

    def _pose_root_prim_path(self, env, entity_name: str, object_prim_path: str) -> str:
        if self._find_scene_entity(env, entity_name) is not None:
            return object_prim_path
        return object_prim_path

    def _extract_stage_prim_pose(self, env, prim_path: str) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        from pxr import UsdGeom

        positions = []
        quats = []
        cache = UsdGeom.XformCache()
        for env_id in range(env.num_envs):
            prim = self._find_prim_by_suffix(env, prim_path, env_id=env_id)
            if prim is None:
                return None, None
            transform = cache.GetLocalToWorldTransform(prim)
            pos = transform.ExtractTranslation()
            quat = transform.ExtractRotationQuat()
            imag = quat.GetImaginary()
            positions.append([float(pos[0]), float(pos[1]), float(pos[2])])
            quats.append([float(imag[0]), float(imag[1]), float(imag[2]), float(quat.GetReal())])
        return (
            torch.tensor(positions, dtype=torch.float32, device=env.device),
            torch.tensor(quats, dtype=torch.float32, device=env.device),
        )

    def _set_stage_prim_local_pose(self, prim, pos, quat_xyzw) -> None:
        from pxr import Gf, UsdGeom

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()
        translate_type = str(translate_op.GetAttr().GetTypeName())
        if translate_type == "float3":
            translate_op.Set(Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])))
        else:
            translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

        orient_type = str(orient_op.GetAttr().GetTypeName())
        if orient_type == "quatd":
            orient_op.Set(
                Gf.Quatd(
                    float(quat_xyzw[3]),
                    Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
                )
            )
        else:
            orient_op.Set(
                Gf.Quatf(
                    float(quat_xyzw[3]),
                    Gf.Vec3f(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
                )
            )

    def _randomize_policy_camera_mounts(self, env, env_ids: torch.Tensor) -> None:
        """Apply episode-fixed tolerances while preserving the stereo-head rig."""

        calibration_candidates = _calibration_table_pose_candidates()
        randomize_mounts = _env_bool("FLIP_TABLE_RL_RANDOMIZE_CAMERA_MOUNTS", False)
        calibration_adjusts_camera = calibration_candidates is not None and any(
            any(abs(float(value)) > 1.0e-12 for value in candidate["head_stereo_offset_local_m"])
            or any(abs(float(value)) > 1.0e-12 for value in candidate["head_stereo_rotation_rpy_deg"])
            for candidate in calibration_candidates
        )
        # A table-only reset candidate must leave V1's authored camera xform
        # stack untouched on a fresh environment. In a persistent worker,
        # however, an earlier camera calibration job may already have created
        # cached authored transforms and changed the stage. Continue through
        # the reset path in that case so each job restores those defaults.
        if (
            not randomize_mounts
            and not calibration_adjusts_camera
            and not self._policy_camera_mount_defaults
        ):
            return
        if calibration_candidates is not None and len(calibration_candidates) != env.num_envs:
            raise ValueError(
                "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON length must equal the active "
                f"environment count ({env.num_envs}), got {len(calibration_candidates)}"
            )

        from pxr import UsdGeom

        level = _rl_randomization_level()
        position_jitter = _env_float(
            "FLIP_TABLE_RL_CAMERA_POSITION_JITTER_M",
            0.0005 + 0.0025 * level,
        )
        rotation_jitter_rad = math.radians(
            _env_float("FLIP_TABLE_RL_CAMERA_ROTATION_JITTER_DEG", 0.10 + 0.90 * level)
        )
        camera_groups = (
            (
                "head_stereo",
                (
                    "Robot/torso_link/first_person_camera",
                    "Robot/torso_link/head_right_camera",
                ),
            ),
            ("left_wrist", ("Robot/left_wrist_yaw_link/left_hand_camera",)),
            ("right_wrist", ("Robot/right_wrist_yaw_link/right_hand_camera",)),
        )
        samples = []
        for env_id in env_ids.tolist():
            env_samples = []
            for group_name, suffixes in camera_groups:
                prims = []
                defaults = []
                for suffix in suffixes:
                    prim = self._find_prim_by_suffix(env, suffix, env_id=env_id)
                    if prim is None:
                        raise RuntimeError(
                            f"Missing policy camera for mount randomization: env={env_id}, {suffix}"
                        )
                    key = str(prim.GetPath())
                    if key not in self._policy_camera_mount_defaults:
                        transform = UsdGeom.Xformable(prim).GetLocalTransformation()
                        position = transform.ExtractTranslation()
                        rotation = transform.ExtractRotationQuat()
                        imaginary = rotation.GetImaginary()
                        self._policy_camera_mount_defaults[key] = (
                            (float(position[0]), float(position[1]), float(position[2])),
                            (
                                float(imaginary[0]),
                                float(imaginary[1]),
                                float(imaginary[2]),
                                float(rotation.GetReal()),
                            ),
                        )
                    prims.append(prim)
                    defaults.append(self._policy_camera_mount_defaults[key])

                calibration_candidate = (
                    calibration_candidates[env_id] if calibration_candidates is not None else None
                )
                if calibration_candidate is not None:
                    if group_name == "head_stereo":
                        position_delta = torch.tensor(
                            calibration_candidate["head_stereo_offset_local_m"],
                            dtype=torch.float32,
                            device=env.device,
                        )
                        rpy_delta = torch.tensor(
                            [
                                math.radians(value)
                                for value in calibration_candidate["head_stereo_rotation_rpy_deg"]
                            ],
                            dtype=torch.float32,
                            device=env.device,
                        )
                    else:
                        position_delta = torch.zeros(3, dtype=torch.float32, device=env.device)
                        rpy_delta = torch.zeros(3, dtype=torch.float32, device=env.device)
                elif randomize_mounts:
                    position_delta = _uniform(
                        (3,), -position_jitter, position_jitter, device=env.device
                    )
                    rpy_delta = _uniform(
                        (3,), -rotation_jitter_rad, rotation_jitter_rad, device=env.device
                    )
                else:
                    position_delta = torch.zeros(3, dtype=torch.float32, device=env.device)
                    rpy_delta = torch.zeros(3, dtype=torch.float32, device=env.device)
                delta_quat = _rpy_quat_xyzw(
                    rpy_delta[0:1], rpy_delta[1:2], rpy_delta[2:3]
                )[0]
                delta_rotation = matrix_from_quat(delta_quat.unsqueeze(0))[0]
                center = torch.stack(
                    [
                        torch.tensor(position, dtype=torch.float32, device=env.device)
                        for position, _quaternion in defaults
                    ]
                ).mean(dim=0)
                for prim, (base_position, base_quat) in zip(prims, defaults, strict=True):
                    base_position_tensor = torch.tensor(
                        base_position, dtype=torch.float32, device=env.device
                    )
                    base_quat_tensor = torch.tensor(
                        base_quat, dtype=torch.float32, device=env.device
                    )
                    camera_position = (
                        center
                        + position_delta
                        + delta_rotation @ (base_position_tensor - center)
                    )
                    camera_quat = _quat_mul_xyzw(
                        delta_quat.unsqueeze(0), base_quat_tensor.unsqueeze(0)
                    )[0]
                    self._set_stage_prim_local_pose(
                        prim,
                        camera_position.detach().cpu(),
                        camera_quat.detach().cpu(),
                    )

                sample = {
                    "env": env_id,
                    "rig": group_name,
                    "cameras": [suffix.rsplit("/", 1)[-1] for suffix in suffixes],
                    "position_delta_mm": [
                        round(float(value * 1000.0), 3) for value in position_delta
                    ],
                    "rotation_delta_deg": [
                        round(math.degrees(float(value)), 3) for value in rpy_delta
                    ],
                }
                if calibration_candidate is not None:
                    sample["calibration_label"] = str(calibration_candidate["label"])
                samples.append(sample)
                env_samples.append(sample)
            self._teleop_randomization(env_id)["camera_mounts"] = env_samples
        if _verbose_reset_logs():
            print(
                f"[FlipTableEvalTask] policy camera mount randomization: {samples}",
                flush=True,
            )

    def _randomize_object_masses(self, env, env_ids: torch.Tensor) -> None:
        """Restore and record the official 1.596 kg assembled-table mass."""

        if _env_bool("FLIP_TABLE_EVAL_RANDOMIZE_MASS", False):
            raise ValueError(
                "white-table mass randomization is disabled; keep the official 1.596 kg mass"
            )
        scale_range = (1.0, 1.0)

        entity_names = (
            "Table001_Table001_01",
            "Leg001_Leg001",
            "Leg001_01_Leg001",
            "Leg001_03_Leg001",
            "Leg001_06_Leg001",
        )
        ids = env_ids.to(device=env.device, dtype=torch.long)
        ids_i32 = ids.to(dtype=torch.int32)
        samples_by_env = {int(env_id): [] for env_id in ids.tolist()}
        for entity_name in entity_names:
            entity = self._find_scene_entity(env, entity_name)
            data = getattr(entity, "data", None)
            if (
                entity is None
                or data is None
                or not hasattr(data, "body_mass")
                or not hasattr(data, "body_inertia")
                or not hasattr(entity, "set_masses_index")
                or not hasattr(entity, "set_inertias_index")
            ):
                raise RuntimeError(f"mass randomization asset is unavailable: {entity_name}")
            if entity_name not in self._object_mass_defaults:
                self._object_mass_defaults[entity_name] = (
                    as_torch(data.body_mass).detach().clone(),
                    as_torch(data.body_inertia).detach().clone(),
                )
            default_mass, default_inertia = self._object_mass_defaults[entity_name]
            body_count = int(default_mass.shape[1])
            body_ids = torch.arange(body_count, device=env.device, dtype=torch.int32)
            scales = _uniform(
                (ids.numel(),),
                scale_range[0],
                scale_range[1],
                device=env.device,
                dtype=default_mass.dtype,
            )
            masses = default_mass[ids] * scales[:, None]
            inertias = default_inertia[ids] * scales[:, None, None]
            entity.set_masses_index(
                masses=masses,
                body_ids=body_ids,
                env_ids=ids_i32,
            )
            entity.set_inertias_index(
                inertias=inertias,
                body_ids=body_ids,
                env_ids=ids_i32,
            )
            for row, env_id in enumerate(ids.tolist()):
                samples_by_env[int(env_id)].append(
                    {
                        "entity": entity_name,
                        "scale": float(scales[row]),
                        "body_mass_kg": [float(value) for value in masses[row].tolist()],
                    }
                )
        for env_id, samples in samples_by_env.items():
            record = self._teleop_randomization(env_id)
            record["table_part_masses"] = samples
            assembled_mass_kg = float(
                sum(
                    mass
                    for sample in samples
                    for mass in sample["body_mass_kg"]
                )
            )
            if not math.isclose(assembled_mass_kg, 1.596, rel_tol=0.0, abs_tol=5.0e-4):
                raise RuntimeError(
                    "official assembled-table mass changed: "
                    f"expected 1.596 kg, got {assembled_mass_kg:.6f} kg"
                )
            record["assembled_table_mass_kg"] = assembled_mass_kg

    def _randomize_upper_body_joint_properties(
        self, env, env_ids: torch.Tensor
    ) -> None:
        """Vary only arm actuator dynamics; legs and waist remain fixed.

        The flip-table arm actions use Isaac Lab's explicit ``IdealPDActuator``.
        Its gains live on ``robot.actuators`` and do not read the simulator
        joint-drive stiffness/damping values.  Updating only the latter would
        record a randomization value without changing the commanded torque.
        Keep the explicit control layer and physical armature/friction layers
        distinct, and leave the simulator joint drive disabled for these
        explicit joints to avoid applying a second PD controller.
        """

        if not _env_bool("FLIP_TABLE_RL_RANDOMIZE_JOINT_PROPERTIES", True):
            return
        robot = self._robot(env)
        data = getattr(robot, "data", None)
        required_writers = (
            "write_joint_stiffness_to_sim_index",
            "write_joint_damping_to_sim_index",
            "write_joint_armature_to_sim_index",
            "write_joint_friction_coefficient_to_sim_index",
        )
        if data is None or any(not hasattr(robot, name) for name in required_writers):
            raise RuntimeError("G1 joint-property writer API is unavailable")
        arm_names = FLIP_TABLE_UPPER_BODY_ACTION_JOINT_NAMES[3:]
        joint_ids, resolved_names = robot.find_joints(list(arm_names), preserve_order=True)
        if tuple(resolved_names) != arm_names:
            raise RuntimeError(f"unexpected G1 arm joint order: {resolved_names}")
        ids = env_ids.to(device=env.device, dtype=torch.long)
        ids_i32 = ids.to(dtype=torch.int32)
        joint_ids_i32 = torch.as_tensor(joint_ids, device=env.device, dtype=torch.int32)
        properties = {
            "armature": (
                "joint_armature",
                "write_joint_armature_to_sim_index",
                0.01 + 0.09 * _rl_randomization_level(),
            ),
            "friction": (
                "joint_friction_coeff",
                "write_joint_friction_coefficient_to_sim_index",
                0.02 + 0.18 * _rl_randomization_level(),
            ),
        }
        if self._upper_body_joint_property_defaults is None:
            self._upper_body_joint_property_defaults = {
                name: as_torch(getattr(data, attribute)).detach().clone()
                for name, (attribute, _writer, _half_width) in properties.items()
            }

        # Build a stable mapping from the action's 14 arm joints to their
        # explicit actuator-local columns.  We intentionally fail closed if a
        # future organizer asset changes the ownership; silently falling back
        # to a simulator drive would make calibration results meaningless.
        if self._upper_body_explicit_actuator_property_defaults is None:
            actuator_groups = getattr(robot, "actuators", None)
            if not isinstance(actuator_groups, dict):
                raise RuntimeError("G1 explicit actuator registry is unavailable")
            expected_ids = {int(joint_id): index for index, joint_id in enumerate(joint_ids)}
            ownership_count = {joint_id: 0 for joint_id in expected_ids}
            direct_defaults: dict[str, list[dict[str, object]]] = {"stiffness": [], "damping": []}
            for actuator_name, actuator in actuator_groups.items():
                local_joint_ids = torch.as_tensor(
                    getattr(actuator, "joint_indices", ()), device=env.device, dtype=torch.long
                ).reshape(-1)
                local_positions = [
                    local_index
                    for local_index, joint_id in enumerate(local_joint_ids.tolist())
                    if int(joint_id) in expected_ids
                ]
                if not local_positions:
                    continue
                # Implicit drives intentionally have no direct torque update.
                # The organizer's flip-table arm group is IdealPDActuator.
                if type(actuator).__name__ == "ImplicitActuator":
                    raise RuntimeError(
                        f"arm joints unexpectedly belong to implicit actuator {actuator_name!r}"
                    )
                local_indices = torch.as_tensor(local_positions, device=env.device, dtype=torch.long)
                for index in local_positions:
                    ownership_count[int(local_joint_ids[index])] += 1
                for property_name in direct_defaults:
                    value = getattr(actuator, property_name, None)
                    if not torch.is_tensor(value) or value.ndim != 2:
                        raise RuntimeError(
                            f"explicit actuator {actuator_name!r} lacks tensor {property_name!r}"
                        )
                    direct_defaults[property_name].append(
                        {
                            "actuator": actuator,
                            "local_indices": local_indices,
                            "defaults": value.detach().clone(),
                        }
                    )
            if any(count != 1 for count in ownership_count.values()):
                invalid = [
                    arm_names[index]
                    for joint_id, index in expected_ids.items()
                    if ownership_count[joint_id] != 1
                ]
                raise RuntimeError(
                    "explicit actuator ownership must contain each arm joint exactly once: "
                    f"{invalid}"
                )
            self._upper_body_explicit_actuator_property_defaults = direct_defaults

        samples_by_env = {int(env_id): {} for env_id in ids.tolist()}
        for property_name, default_half_width in (
            ("stiffness", 0.01 + 0.09 * _rl_randomization_level()),
            ("damping", 0.02 + 0.13 * _rl_randomization_level()),
        ):
            scale_range = _env_range(
                f"FLIP_TABLE_ARM_{property_name.upper()}_SCALE_RANGE",
                (1.0 - default_half_width, 1.0 + default_half_width),
            )
            if scale_range[0] <= 0.0:
                raise ValueError(f"arm {property_name} scale must remain positive")
            scales = _uniform(
                (ids.numel(),),
                scale_range[0],
                scale_range[1],
                device=env.device,
                dtype=torch.float32,
            )
            for entry in self._upper_body_explicit_actuator_property_defaults[property_name]:
                actuator = entry["actuator"]
                local_indices = entry["local_indices"]
                defaults = entry["defaults"]
                values = defaults[ids[:, None], local_indices[None, :]] * scales[:, None].to(
                    dtype=defaults.dtype
                )
                getattr(actuator, property_name)[ids[:, None], local_indices[None, :]] = values
            # Explicit actuators submit efforts, so the underlying simulator
            # drive must remain zero.  Without this, a previous reset can leave
            # a second PD term active beside IdealPDActuator.
            zero = torch.zeros(
                (ids.numel(), len(joint_ids)), device=env.device, dtype=torch.float32
            )
            getattr(robot, f"write_joint_{property_name}_to_sim_index")(
                **{
                    property_name: zero,
                    "joint_ids": joint_ids_i32,
                    "env_ids": ids_i32,
                }
            )
            for row, env_id in enumerate(ids.tolist()):
                samples_by_env[int(env_id)][f"{property_name}_scale"] = float(scales[row])
        for property_name, (_attribute, writer_name, default_half_width) in properties.items():
            scale_range = _env_range(
                f"FLIP_TABLE_ARM_{property_name.upper()}_SCALE_RANGE",
                (1.0 - default_half_width, 1.0 + default_half_width),
            )
            if scale_range[0] <= 0.0:
                raise ValueError(f"arm {property_name} scale must remain positive")
            defaults = self._upper_body_joint_property_defaults[property_name]
            scales = _uniform(
                (ids.numel(),),
                scale_range[0],
                scale_range[1],
                device=env.device,
                dtype=defaults.dtype,
            )
            values = defaults[ids[:, None], joint_ids_i32[None, :]] * scales[:, None]
            getattr(robot, writer_name)(
                **{
                    property_name if property_name != "friction" else "joint_friction_coeff": values,
                    "joint_ids": joint_ids_i32,
                    "env_ids": ids_i32,
                }
            )
            for row, env_id in enumerate(ids.tolist()):
                samples_by_env[int(env_id)][f"{property_name}_scale"] = float(
                    scales[row]
                )
        for env_id, sample in samples_by_env.items():
            sample["joint_names"] = list(arm_names)
            sample["control_layer"] = "IdealPDActuator"
            sample["simulator_joint_drive"] = "disabled_for_explicit_arm_actuators"
            self._teleop_randomization(env_id)["arm_joint_properties"] = sample

    def _sample_camera_image_model(self, env, env_ids: torch.Tensor) -> None:
        """Sample episode-fixed intrinsics and time-varying sensor limits."""

        geometry_enabled = _env_bool(
            "FLIP_TABLE_RL_RANDOMIZE_IMAGE_GEOMETRY", True
        )
        sensor_enabled = _env_bool("FLIP_TABLE_RL_ENABLE_SENSOR_NOISE", True)
        level = _rl_randomization_level()
        focal_half_width = 0.02 * level if geometry_enabled else 0.0
        principal_half_width = 3.0 * level if geometry_enabled else 0.0
        distortion_half_width = 0.10 * level if geometry_enabled else 0.0
        exposure_half_width = 0.35 * level if geometry_enabled else 0.0
        latency_max = (
            _env_int(
                "FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS",
                2 if level >= 0.8 else 1,
                minimum=0,
            )
            if sensor_enabled
            else 0
        )
        noise_max = (
            _env_float("FLIP_TABLE_RL_IMAGE_NOISE_STD_MAX", 0.002 + 0.008 * level)
            if sensor_enabled
            else 0.0
        )
        action_delay_max = _env_int(
            "FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS",
            2 if level >= 0.8 else 1,
            minimum=0,
        )
        for env_id in env_ids.tolist():
            rigs = {}
            for rig_name in ("head_stereo", "left_wrist", "right_wrist"):
                rigs[rig_name] = {
                    "focal_scale": float(
                        _uniform(
                            (1,),
                            1.0 - focal_half_width,
                            1.0 + focal_half_width,
                            device=env.device,
                        )[0]
                    ),
                    "principal_point_delta_px": [
                        float(value)
                        for value in _uniform(
                            (2,),
                            -principal_half_width,
                            principal_half_width,
                            device=env.device,
                        ).tolist()
                    ],
                    "distortion_scale": float(
                        _uniform(
                            (1,),
                            1.0 - distortion_half_width,
                            1.0 + distortion_half_width,
                            device=env.device,
                        )[0]
                    ),
                    "exposure_ev": float(
                        _uniform(
                            (1,),
                            -exposure_half_width,
                            exposure_half_width,
                            device=env.device,
                        )[0]
                    ),
                }
            self._teleop_randomization(env_id)["camera_image"] = {
                "rigs": rigs,
                "noise_std_fraction_max": noise_max,
                "latency_max_steps": latency_max,
            }
            self._teleop_randomization(env_id)["control"] = {
                "action_delay_steps": int(
                    torch.randint(
                        action_delay_max + 1,
                        (1,),
                        device=env.device,
                    )[0]
                )
                if action_delay_max
                else 0,
                "action_delay_max_steps": action_delay_max,
            }

    def _prim_world_pose_xyzw(self, prim, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        from pxr import UsdGeom

        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = transform.ExtractTranslation()
        quat = transform.ExtractRotationQuat()
        imag = quat.GetImaginary()
        return (
            torch.tensor([float(pos[0]), float(pos[1]), float(pos[2])], dtype=torch.float32, device=device),
            torch.tensor(
                [float(imag[0]), float(imag[1]), float(imag[2]), float(quat.GetReal())],
                dtype=torch.float32,
                device=device,
            ),
        )

    def _set_stage_prim_world_pose(self, env, prim, pos_w: torch.Tensor, quat_w_xyzw: torch.Tensor) -> None:
        parent = prim.GetParent()
        if parent and parent.IsValid():
            parent_pos, parent_quat = self._prim_world_pose_xyzw(parent, env.device)
            parent_rot = matrix_from_quat(parent_quat.unsqueeze(0))[0]
            local_pos = parent_rot.transpose(0, 1) @ (pos_w - parent_pos)
            local_quat = _quat_mul_xyzw(
                _quat_conjugate_xyzw(parent_quat.unsqueeze(0)),
                quat_w_xyzw.unsqueeze(0),
            )[0]
        else:
            local_pos = pos_w
            local_quat = quat_w_xyzw
        self._set_stage_prim_local_pose(prim, local_pos.detach().cpu(), local_quat.detach().cpu())

    def _write_stage_prim_pose(self, env, prim_path: str, pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        for row, env_id in enumerate(env_ids.tolist()):
            prim = self._find_prim_by_suffix(env, prim_path, env_id=env_id)
            if prim is None:
                print(f"[FlipTableEvalTask] missing prim {prim_path} for env {env_id}", flush=True)
                continue
            self._set_stage_prim_world_pose(env, prim, pose[row, :3], pose[row, 3:7])

    def _extract_entity_root_pose(self, entity) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if entity is None or not hasattr(entity, "data"):
            return None, None
        data = entity.data

        pos = None
        for attr in ("root_pos_w", "root_link_pos_w", "body_com_pos_w"):
            if hasattr(data, attr):
                pos = as_torch(getattr(data, attr))
                break
        quat = None
        for attr in ("root_quat_w", "root_link_quat_w", "body_com_quat_w"):
            if hasattr(data, attr):
                quat = as_torch(getattr(data, attr))
                break
        if pos is None or quat is None:
            return None, None
        if pos.ndim == 3:
            pos = pos[:, 0, :]
        if quat.ndim == 3:
            quat = quat[:, 0, :]
        return pos[:, :3], sim_quat_raw_to_xyzw_torch(quat[:, :4])

    def _extract_object_pose(
        self,
        env,
        entity_name: str,
        prim_path: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        entity = self._find_scene_entity(env, entity_name)
        pos, quat = self._extract_entity_root_pose(entity)
        if pos is not None and quat is not None:
            return pos, quat
        return self._extract_stage_prim_pose(env, prim_path)

    def _write_object_pose(
        self,
        env,
        entity_name: str,
        prim_path: str,
        pose: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        entity = self._find_scene_entity(env, entity_name)
        if entity is None:
            self._write_stage_prim_pose(env, prim_path, pose, env_ids)
            return
        env_ids_i32 = env_ids.to(dtype=torch.int32)
        entity.write_root_pose_to_sim(pose, env_ids=env_ids_i32)
        if hasattr(entity, "write_root_velocity_to_sim"):
            zeros = torch.zeros((env_ids_i32.numel(), 6), dtype=pose.dtype, device=pose.device)
            entity.write_root_velocity_to_sim(zeros, env_ids=env_ids_i32)
        if hasattr(entity, "reset"):
            entity.reset(env_ids_i32)

    def _table_pose_root(self, env) -> tuple[str, str, torch.Tensor | None, torch.Tensor | None]:
        table_name, table_prim_path, _ = self.table_reg_int_sites[0]
        pose_root_prim_path = self._pose_root_prim_path(env, table_name, table_prim_path)
        pos, quat = self._extract_object_pose(env, table_name, pose_root_prim_path)
        return table_name, pose_root_prim_path, pos, quat

    def _table_body_pose(self, env) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        table_name, table_prim_path, _ = self.table_reg_int_sites[0]
        pos, quat = self._extract_object_pose(env, table_name, table_prim_path)
        if pos is None or quat is None:
            return None, None
        # Pose reads are snapshots. Returning simulator-backed views makes a saved
        # "initial" pose change as physics advances and silently zeros deltas.
        return pos.clone(), quat.clone()

    def _workbench_pose(self, env) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self._extract_stage_prim_pose(env, self.workbench_prim_path)

    def _sample_table_pose_on_workbench(
        self,
        env,
        base_table_pos_local: torch.Tensor,
        base_table_quat: torch.Tensor,
        workbench_pos_local: torch.Tensor | None,
        workbench_quat: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        count = base_table_pos_local.shape[0]
        candidates = _calibration_table_pose_candidates()
        if candidates is not None and len(candidates) != count:
            raise ValueError(
                "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON length must equal the active "
                f"environment count ({count}), got {len(candidates)}"
            )
        yaw_range = _env_float("FLIP_TABLE_TABLE_YAW_RANGE_RAD", math.pi)
        yaw_offset = _env_float("FLIP_TABLE_TABLE_YAW_OFFSET_RAD", 0.0)
        if yaw_range < 0:
            raise ValueError("FLIP_TABLE_TABLE_YAW_RANGE_RAD must be non-negative")
        if candidates is None:
            yaw_delta = _uniform((count,), -yaw_range, yaw_range, device=env.device) + yaw_offset
        else:
            yaw_delta = torch.tensor(
                [float(candidate["yaw_rad"]) for candidate in candidates],
                dtype=base_table_pos_local.dtype,
                device=env.device,
            )
        yaw_delta_quat = _yaw_quat_xyzw(yaw_delta)
        table_quat = _quat_mul_xyzw(yaw_delta_quat, base_table_quat)

        legacy_xy_range = os.environ.get("FLIP_TABLE_TABLE_XY_RANGE_M")
        legacy_range = (
            _env_float("FLIP_TABLE_TABLE_XY_RANGE_M", 0.0)
            if legacy_xy_range not in (None, "")
            else None
        )
        if legacy_xy_range and "FLIP_TABLE_TABLE_LONG_RANGE_M" not in os.environ:
            table_long_range = legacy_range
        else:
            table_long_range = _env_float("FLIP_TABLE_TABLE_LONG_RANGE_M", 0.12)
        if legacy_xy_range and "FLIP_TABLE_TABLE_DEPTH_RANGE_M" not in os.environ:
            table_depth_range = min(legacy_range, 0.04)
        else:
            table_depth_range = _env_float("FLIP_TABLE_TABLE_DEPTH_RANGE_M", 0.035)
        if table_long_range < 0 or table_depth_range < 0:
            raise ValueError("table position randomization ranges must be non-negative")

        if workbench_pos_local is None or workbench_quat is None:
            xy_range = _env_float("FLIP_TABLE_TABLE_XY_RANGE_M", 0.08)
            if xy_range < 0:
                raise ValueError("FLIP_TABLE_TABLE_XY_RANGE_M must be non-negative")
            if candidates is None:
                xy_offset = _uniform((count, 2), -xy_range, xy_range, device=env.device)
                z_offset = torch.zeros(count, dtype=base_table_pos_local.dtype, device=env.device)
            else:
                offsets = torch.tensor(
                    [candidate["offset_local_m"] for candidate in candidates],
                    dtype=base_table_pos_local.dtype,
                    device=env.device,
                )
                xy_offset = offsets[:, :2]
                z_offset = offsets[:, 2]
            table_pos_local = base_table_pos_local.clone()
            table_pos_local[:, :2] += xy_offset
            table_pos_local[:, 2] += z_offset
            return table_pos_local, table_quat, yaw_delta, {
                "table_long_range_m": xy_range,
                "table_depth_range_m": xy_range,
                "table_yaw_range_rad": yaw_range,
                "table_yaw_offset_rad": yaw_offset,
            }

        workbench_rot = matrix_from_quat(workbench_quat)
        rel_w = base_table_pos_local - workbench_pos_local
        base_offset = _env_tuple_or_none("FLIP_TABLE_TABLE_BASE_OFFSET_LOCAL", 3)
        if base_offset is not None:
            rel_w = rel_w + torch.tensor(
                base_offset,
                dtype=rel_w.dtype,
                device=env.device,
            ).expand_as(rel_w)
        rel_local = torch.bmm(workbench_rot.transpose(1, 2), rel_w.unsqueeze(-1)).squeeze(-1)

        workbench_half_length = _env_float("FLIP_TABLE_WORKBENCH_HALF_LENGTH_M", 0.90)
        workbench_half_depth = _env_float("FLIP_TABLE_WORKBENCH_HALF_DEPTH_M", 0.375)
        configured_margin = _env_float("FLIP_TABLE_WORKBENCH_EDGE_MARGIN_M", 0.03)
        success_margin = _env_float("FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M", 0.03)
        placement_buffer = _env_float("FLIP_TABLE_TABLE_PLACEMENT_BUFFER_M", 0.02)
        nominal_margin = max(configured_margin, success_margin) + placement_buffer
        # Source-calibrated resets may reproduce an overhang visible in real
        # demonstrations. Physical plausibility is governed by the supported
        # center and contact area, not by whether every tabletop corner lies
        # within the workbench outline. This never relaxes randomized
        # placement or the success predicate, which retain nominal_margin.
        calibration_center_margin = _env_float(
            "FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M", 0.05
        )
        calibration_min_support_fraction = _env_float(
            "FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION", 0.70
        )
        if candidates is None:
            margin = nominal_margin
        else:
            if calibration_center_margin < 0.0:
                raise ValueError(
                    "FLIP_TABLE_CALIBRATION_SUPPORT_CENTER_MARGIN_M must be non-negative"
                )
            if not 0.0 < calibration_min_support_fraction <= 1.0:
                raise ValueError(
                    "FLIP_TABLE_CALIBRATION_MIN_WORKBENCH_SUPPORT_FRACTION must be within (0, 1]"
                )
            margin = 0.0
        if min(workbench_half_length, workbench_half_depth) <= 0:
            raise ValueError("workbench dimensions must be positive")
        if min(configured_margin, success_margin, placement_buffer) < 0:
            raise ValueError("table placement and success margins must be non-negative")

        # Constrain the complete tabletop footprint, not only its center.  A
        # center-only clamp allowed rotated corners to overhang the support and
        # made otherwise valid randomized resets physically unstable.  The
        # projected half extents are evaluated for every sampled yaw so the
        # yaw distribution remains broad whenever the workbench can support it.
        tabletop_half_extents = torch.tensor(
            (
                0.5 * _env_float("FLIP_TABLE_TOP_LENGTH_M", 0.58),
                0.5 * _env_float("FLIP_TABLE_TOP_DEPTH_M", 0.42),
            ),
            dtype=rel_local.dtype,
            device=env.device,
        )
        if bool(torch.any(tabletop_half_extents <= 0.0)):
            raise ValueError("tabletop dimensions must be positive")
        def footprint_allowance(
            sampled_table_quat: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            table_rot = matrix_from_quat(sampled_table_quat)
            table_in_workbench = torch.bmm(
                workbench_rot.transpose(1, 2), table_rot
            )
            projected = torch.bmm(
                torch.abs(table_in_workbench[:, :2, :2]),
                tabletop_half_extents.expand(count, -1).unsqueeze(-1),
            ).squeeze(-1)
            return (
                workbench_half_length - margin - projected[:, 0],
                workbench_half_depth - margin - projected[:, 1],
            )

        allowed_length, allowed_depth = footprint_allowance(table_quat)
        invalid_yaw = (allowed_length < 0.0) | (allowed_depth < 0.0)
        if candidates is None:
            # Some diagonal orientations cannot fit on the narrow workbench at
            # the requested margin. Rejection-sample within the configured yaw
            # interval rather than accepting overhang or biasing via a clamp.
            for _ in range(64):
                if not bool(torch.any(invalid_yaw)):
                    break
                replacement = (
                    _uniform((count,), -yaw_range, yaw_range, device=env.device)
                    + yaw_offset
                )
                yaw_delta = torch.where(invalid_yaw, replacement, yaw_delta)
                yaw_delta_quat = _yaw_quat_xyzw(yaw_delta)
                table_quat = _quat_mul_xyzw(yaw_delta_quat, base_table_quat)
                allowed_length, allowed_depth = footprint_allowance(table_quat)
                invalid_yaw = (allowed_length < 0.0) | (allowed_depth < 0.0)
        if bool(torch.any(invalid_yaw)):
            if candidates is not None:
                raise ValueError(
                    "a calibration tabletop yaw exceeds the bounded workbench support envelope"
                )
            raise ValueError(
                "sampled tabletop yaw cannot preserve the configured workbench footprint margin"
            )

        if candidates is None:
            length_noise = _uniform((count,), -1.0, 1.0, device=env.device) * torch.minimum(
                allowed_length,
                torch.full_like(allowed_length, table_long_range),
            )
            depth_noise = _uniform((count,), -1.0, 1.0, device=env.device) * torch.minimum(
                allowed_depth,
                torch.full_like(allowed_depth, table_depth_range),
            )
        else:
            offsets = torch.tensor(
                [candidate["offset_local_m"] for candidate in candidates],
                dtype=base_table_pos_local.dtype,
                device=env.device,
            )
            length_noise = offsets[:, 0]
            depth_noise = offsets[:, 1]
        requested_length = rel_local[:, 0] + length_noise
        requested_depth = rel_local[:, 1] + depth_noise
        if candidates is not None:
            requested_center = torch.stack((requested_length, requested_depth), dim=-1)
            workbench_half_extents = torch.tensor(
                (workbench_half_length, workbench_half_depth),
                dtype=rel_local.dtype,
                device=env.device,
            )
            center_allowance = workbench_half_extents - calibration_center_margin
            center_is_supported = torch.all(
                torch.abs(requested_center) <= center_allowance.expand_as(requested_center),
                dim=-1,
            )
            footprint_min = requested_center - torch.stack((
                workbench_half_length - allowed_length,
                workbench_half_depth - allowed_depth,
            ), dim=-1)
            footprint_max = requested_center + torch.stack((
                workbench_half_length - allowed_length,
                workbench_half_depth - allowed_depth,
            ), dim=-1)
            overlap_min = torch.maximum(footprint_min, -workbench_half_extents.expand_as(footprint_min))
            overlap_max = torch.minimum(footprint_max, workbench_half_extents.expand_as(footprint_max))
            overlap_extent = torch.clamp(overlap_max - overlap_min, min=0.0)
            footprint_extent = 2.0 * torch.stack((
                workbench_half_length - allowed_length,
                workbench_half_depth - allowed_depth,
            ), dim=-1)
            support_fraction = torch.prod(overlap_extent, dim=-1) / torch.prod(footprint_extent, dim=-1)
            if bool(torch.any(~center_is_supported)) or bool(
                torch.any(support_fraction < calibration_min_support_fraction)
            ):
                raise ValueError(
                    "a calibration tabletop pose does not satisfy the bounded workbench support condition"
                )
            rel_local[:, 0] = requested_length
            rel_local[:, 1] = requested_depth
        else:
            rel_local[:, 0] = torch.clamp(requested_length, -allowed_length, allowed_length)
            rel_local[:, 1] = torch.clamp(requested_depth, -allowed_depth, allowed_depth)

        table_pos_local = workbench_pos_local + torch.bmm(workbench_rot, rel_local.unsqueeze(-1)).squeeze(-1)
        if candidates is None:
            table_pos_local[:, 2] = base_table_pos_local[:, 2]
        else:
            table_pos_local[:, 2] = base_table_pos_local[:, 2] + offsets[:, 2]
        return table_pos_local, table_quat, yaw_delta, {
            "table_long_range_m": table_long_range,
            "table_depth_range_m": table_depth_range,
            "table_yaw_range_rad": yaw_range,
            "table_yaw_offset_rad": yaw_offset,
            "workbench_footprint_margin_m": margin,
            "nominal_workbench_footprint_margin_m": nominal_margin,
            "calibration_support_center_margin_m": calibration_center_margin if candidates is not None else 0.0,
            "calibration_min_workbench_support_fraction": (
                calibration_min_support_fraction if candidates is not None else 0.0
            ),
            "calibration_candidates": float(0 if candidates is None else len(candidates)),
        }

    def _ensure_assembled_table_joints(self, env) -> None:
        from pxr import Gf, UsdPhysics

        table_body_prim_path = self.table_reg_int_sites[0][1]
        stage = env.sim.stage
        created = 0

        for env_id in range(env.num_envs):
            table_body_prim = self._find_prim_by_suffix(env, table_body_prim_path, env_id=env_id)
            if table_body_prim is None:
                continue
            table_body_pos, table_body_quat = self._extract_stage_prim_pose(env, table_body_prim_path)
            if table_body_pos is None or table_body_quat is None:
                continue

            # Scene02.usd contains one obsolete joint named exactly
            # `FixedJoint`. Its body relationships point at pre-renamed prims.
            # Preserve the valid `FlipTableEvalFixedJoint_*` joints generated
            # by the scene-preparation tool so they can be reused and verified.
            for child in list(table_body_prim.GetChildren()):
                if child.GetName() == "FixedJoint":
                    stage.RemovePrim(child.GetPath())
                    print(
                        f"[FlipTableEvalTask] removed stale table joint: {child.GetPath()}",
                        flush=True,
                    )

            existing_joints = []
            for leg_index, (_leg_name, leg_body_prim_path, _leg_site_prim_path) in enumerate(self.leg_reg_int_sites):
                joint_path = table_body_prim.GetPath().AppendChild(f"FlipTableEvalFixedJoint_{leg_index}")
                joint_prim = stage.GetPrimAtPath(joint_path)
                if not joint_prim.IsValid():
                    existing_joints = []
                    break
                joint = UsdPhysics.FixedJoint(joint_prim)
                body0_targets = [str(target) for target in joint.GetBody0Rel().GetTargets()]
                body1_targets = [str(target) for target in joint.GetBody1Rel().GetTargets()]
                leg_body_prim = self._find_prim_by_suffix(env, leg_body_prim_path, env_id=env_id)
                if (
                    not body0_targets
                    or not body1_targets
                    or not body0_targets[0].rstrip("/").endswith(table_body_prim_path.strip("/"))
                    or leg_body_prim is None
                    or not body1_targets[0].rstrip("/").endswith(str(leg_body_prim.GetPath()).strip("/"))
                ):
                    existing_joints = []
                    break
                existing_joints.append(str(joint_path))
            if len(existing_joints) == len(self.leg_reg_int_sites):
                # Prepared V1 joints retain their authored local frames.  A
                # calibration or DR reset moves the whole assembled table, so
                # refresh those frames to avoid a constraint impulse.
                table_pos_i = table_body_pos[env_id]
                table_quat_i = table_body_quat[env_id]
                for leg_index, (_leg_name, leg_body_prim_path, _leg_site_prim_path) in enumerate(
                    self.leg_reg_int_sites
                ):
                    leg_body_pos, leg_body_quat = self._extract_stage_prim_pose(env, leg_body_prim_path)
                    if leg_body_pos is None or leg_body_quat is None:
                        raise RuntimeError("assembled table leg pose is unavailable")
                    leg_pos_i = leg_body_pos[env_id]
                    leg_quat_i = leg_body_quat[env_id]
                    leg_rot_i = matrix_from_quat(leg_quat_i.unsqueeze(0))[0]
                    rel_pos = leg_rot_i.transpose(0, 1) @ (table_pos_i - leg_pos_i)
                    rel_quat = _quat_mul_xyzw(
                        _quat_conjugate_xyzw(leg_quat_i.unsqueeze(0)),
                        table_quat_i.unsqueeze(0),
                    )[0]
                    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(existing_joints[leg_index]))
                    rel_xyz = rel_pos.detach().cpu().tolist()
                    rel_q = rel_quat.detach().cpu().tolist()
                    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    joint.CreateLocalPos1Attr().Set(
                        Gf.Vec3f(float(rel_xyz[0]), float(rel_xyz[1]), float(rel_xyz[2]))
                    )
                    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
                    joint.CreateLocalRot1Attr().Set(
                        Gf.Quatf(
                            float(rel_q[3]),
                            Gf.Vec3f(float(rel_q[0]), float(rel_q[1]), float(rel_q[2])),
                        )
                    )
                created += len(existing_joints)
                print(
                    f"[FlipTableEvalTask] refreshed assembled table joints: {len(existing_joints)}",
                    flush=True,
                )
                continue

            for child in list(table_body_prim.GetChildren()):
                if child.GetName().startswith("FlipTableEvalFixedJoint_"):
                    stage.RemovePrim(child.GetPath())

            for leg_index, (leg_name, leg_body_prim_path, _leg_site_prim_path) in enumerate(self.leg_reg_int_sites):
                leg_body_prim = self._find_prim_by_suffix(env, leg_body_prim_path, env_id=env_id)
                leg_body_pos, leg_body_quat = self._extract_stage_prim_pose(env, leg_body_prim_path)
                if leg_body_prim is None or leg_body_pos is None or leg_body_quat is None:
                    print(f"[FlipTableEvalTask] fixed joint skipped for {leg_name}", flush=True)
                    continue

                joint_path = table_body_prim.GetPath().AppendChild(f"FlipTableEvalFixedJoint_{leg_index}")
                joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
                joint.CreateBody0Rel().SetTargets([table_body_prim.GetPath()])
                joint.CreateBody1Rel().SetTargets([leg_body_prim.GetPath()])

                table_pos_i = table_body_pos[env_id]
                table_quat_i = table_body_quat[env_id]
                leg_pos_i = leg_body_pos[env_id]
                leg_quat_i = leg_body_quat[env_id]
                leg_rot_i = matrix_from_quat(leg_quat_i.unsqueeze(0))[0]
                rel_pos = leg_rot_i.transpose(0, 1) @ (table_pos_i - leg_pos_i)
                rel_quat = _quat_mul_xyzw(
                    _quat_conjugate_xyzw(leg_quat_i.unsqueeze(0)),
                    table_quat_i.unsqueeze(0),
                )[0]

                identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
                rel_xyz = rel_pos.detach().cpu().tolist()
                rel_q = rel_quat.detach().cpu().tolist()
                joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                joint.CreateLocalPos1Attr().Set(Gf.Vec3f(float(rel_xyz[0]), float(rel_xyz[1]), float(rel_xyz[2])))
                joint.CreateLocalRot0Attr().Set(identity)
                joint.CreateLocalRot1Attr().Set(
                    Gf.Quatf(
                        float(rel_q[3]),
                        Gf.Vec3f(float(rel_q[0]), float(rel_q[1]), float(rel_q[2])),
                    )
                )
                created += 1

        self._assembled_table_joints_created = created == env.num_envs * len(self.leg_reg_int_sites)
        print(f"[FlipTableEvalTask] fixed joints ready: {created}", flush=True)

    def _sample_robot_pose_for_workbench_front(
        self,
        env,
        env_ids: torch.Tensor,
        table_pos_local: torch.Tensor,
        workbench_pos_local: torch.Tensor | None,
        workbench_quat_xyzw: torch.Tensor | None,
        table_yaw_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distance = _env_float("FLIP_TABLE_ROBOT_DISTANCE_M", 0.26)
        distance_range = _env_float("FLIP_TABLE_ROBOT_DISTANCE_RANGE_M", 0.04)
        lateral_range = _env_float("FLIP_TABLE_ROBOT_LATERAL_RANGE_M", 0.10)
        yaw_range = _env_float("FLIP_TABLE_ROBOT_YAW_RANGE_RAD", 0.08)
        yaw_offset = _env_float("FLIP_TABLE_ROBOT_YAW_OFFSET_RAD", 0.0)
        min_table_distance = _env_float("FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M", 0.62)
        # The floating-base WBC settles the spawned robot during the first few
        # control ticks. Reserve explicit placement clearance for that physical
        # transient instead of weakening the post-settle geometry contract.
        wbc_settle_clearance = _env_float(
            "FLIP_TABLE_ROBOT_WBC_SETTLE_CLEARANCE_M", 0.03
        )
        min_workbench_clearance = _env_float(
            "FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M", 0.20
        )
        if min_table_distance <= 0:
            raise ValueError("FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M must be positive")
        if wbc_settle_clearance < 0:
            raise ValueError(
                "FLIP_TABLE_ROBOT_WBC_SETTLE_CLEARANCE_M must be non-negative"
            )
        if min_workbench_clearance <= 0:
            raise ValueError("FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M must be positive")
        base_height = _env_float("FLIP_TABLE_ROBOT_BASE_HEIGHT_M", 0.78)
        workbench_half_depth = _env_float("FLIP_TABLE_WORKBENCH_HALF_DEPTH_M", 0.375)
        fixed_robot_root = _env_tuple_or_none("FLIP_TABLE_ROBOT_ROOT_POS_LOCAL", 3)
        fixed_robot_yaw = _env_tuple_or_none("FLIP_TABLE_ROBOT_ROOT_YAW_RAD", 1)
        calibration_candidates = _calibration_table_pose_candidates()
        if calibration_candidates is not None and len(calibration_candidates) != env.num_envs:
            raise ValueError(
                "FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON length must equal the active "
                f"environment count ({env.num_envs}), got {len(calibration_candidates)}"
            )
        robot_pos_local = torch.empty((env_ids.numel(), 3), dtype=table_pos_local.dtype, device=env.device)
        robot_yaw = torch.empty((env_ids.numel(),), dtype=table_pos_local.dtype, device=env.device)

        workbench_rot = (
            matrix_from_quat(workbench_quat_xyzw)
            if workbench_quat_xyzw is not None
            else None
        )
        # Scene02 rotates the workbench by +90 deg. Its operator side is local
        # -Y, which maps to world +X and places G1 at the validated x ~= -0.8 m
        # side of the table, facing world -X.
        front_axis_xy = _env_axis_xy("FLIP_TABLE_WORKBENCH_FRONT_AXIS", (0.0, -1.0))
        front_axis = torch.tensor(
            [front_axis_xy[0], front_axis_xy[1], 0.0],
            dtype=table_pos_local.dtype,
            device=env.device,
        )

        for i, _env_id in enumerate(env_ids.tolist()):
            dist = float(_uniform((1,), distance - distance_range, distance + distance_range, device=env.device)[0])
            lateral = float(_uniform((1,), -lateral_range, lateral_range, device=env.device)[0])
            heading_jitter = float(_uniform((1,), -0.5 * yaw_range, 0.5 * yaw_range, device=env.device)[0])
            table_xy = table_pos_local[i, :2].detach().cpu().tolist()
            workbench_xy_for_clearance: list[float] | None = None
            outward_xy_for_clearance: tuple[float, float] | None = None

            calibration_candidate = (
                calibration_candidates[_env_id] if calibration_candidates is not None else None
            )
            calibration_root = (
                calibration_candidate.get("robot_root_pos_local_m")
                if calibration_candidate is not None
                else None
            )
            calibration_yaw = (
                calibration_candidate.get("robot_root_yaw_rad")
                if calibration_candidate is not None
                else None
            )
            selected_root = calibration_root if calibration_root is not None else fixed_robot_root
            if selected_root is not None:
                robot_x, robot_y, robot_z = selected_root
                table_distance = math.hypot(
                    robot_x - float(table_xy[0]), robot_y - float(table_xy[1])
                )
                placement_min_table_distance = min_table_distance + wbc_settle_clearance
                if table_distance < placement_min_table_distance:
                    raise ValueError(
                        "a fixed calibration robot root violates the robot-table clearance"
                    )
                if workbench_rot is not None and workbench_pos_local is not None:
                    outward = workbench_rot[i] @ front_axis
                    outward_xy_t = _normalize_xy(outward).detach().cpu()
                    workbench_xy = workbench_pos_local[i, :2].detach().cpu().tolist()
                    outward_distance = (
                        (robot_x - float(workbench_xy[0])) * float(outward_xy_t[0])
                        + (robot_y - float(workbench_xy[1])) * float(outward_xy_t[1])
                    )
                    if outward_distance - workbench_half_depth < min_workbench_clearance:
                        raise ValueError(
                            "a fixed calibration robot root violates the workbench clearance"
                        )
                approach_yaw = math.atan2(float(table_xy[1] - robot_y), float(table_xy[0] - robot_x))
                yaw = (
                    float(calibration_yaw)
                    if calibration_yaw is not None
                    else float(fixed_robot_yaw[0])
                    if fixed_robot_yaw is not None
                    else approach_yaw + yaw_offset + heading_jitter
                )
                robot_pos_local[i] = torch.tensor(
                    [robot_x, robot_y, robot_z],
                    dtype=table_pos_local.dtype,
                    device=env.device,
                )
                robot_yaw[i] = yaw
                continue

            if workbench_rot is not None:
                outward = workbench_rot[i] @ front_axis
                outward_xy_t = _normalize_xy(outward).detach().cpu()
                lateral_axis_local = torch.tensor(
                    [-front_axis[1], front_axis[0], 0.0],
                    dtype=table_pos_local.dtype,
                    device=env.device,
                )
                lateral_axis = workbench_rot[i] @ lateral_axis_local
                lateral_xy_t = _normalize_xy(lateral_axis).detach().cpu()
                outward_xy = (float(outward_xy_t[0]), float(outward_xy_t[1]))
                outward_xy_for_clearance = outward_xy
                lateral_xy = (float(lateral_xy_t[0]), float(lateral_xy_t[1]))
                if workbench_pos_local is not None:
                    workbench_xy = workbench_pos_local[i, :2].detach().cpu().tolist()
                    workbench_xy_for_clearance = workbench_xy
                    table_rel_xy = [
                        float(table_xy[0] - workbench_xy[0]),
                        float(table_xy[1] - workbench_xy[1]),
                    ]
                    table_lateral = table_rel_xy[0] * lateral_xy[0] + table_rel_xy[1] * lateral_xy[1]
                    workbench_front_xy = [
                        float(workbench_xy[0] + workbench_half_depth * outward_xy[0]),
                        float(workbench_xy[1] + workbench_half_depth * outward_xy[1]),
                    ]
                else:
                    table_lateral = 0.0
                    workbench_front_xy = table_xy
                robot_lateral = table_lateral + lateral
                # Robot placement is fixed to the workbench operator side.
                # Table yaw must never move G1 around the workbench; otherwise
                # full-circle table randomization aliases back to a short-edge view.
                robot_x = float(
                    workbench_front_xy[0]
                    + dist * outward_xy[0]
                    + robot_lateral * lateral_xy[0]
                )
                robot_y = float(
                    workbench_front_xy[1]
                    + dist * outward_xy[1]
                    + robot_lateral * lateral_xy[1]
                )
                approach_yaw = math.atan2(
                    float(table_xy[1] - robot_y),
                    float(table_xy[0] - robot_x),
                )

            else:
                approach_yaw = float(_uniform((1,), -yaw_range, yaw_range, device=env.device)[0])
                forward = (math.cos(approach_yaw), math.sin(approach_yaw))
                left = (-forward[1], forward[0])
                robot_x = float(table_xy[0] - dist * forward[0] + lateral * left[0])
                robot_y = float(table_xy[1] - dist * forward[1] + lateral * left[1])

            table_to_robot_x = robot_x - float(table_xy[0])
            table_to_robot_y = robot_y - float(table_xy[1])
            table_distance = math.hypot(table_to_robot_x, table_to_robot_y)
            placement_min_table_distance = min_table_distance + wbc_settle_clearance
            if table_distance < placement_min_table_distance:
                # Keep the requested lateral/distance randomization, but reject
                # its unsafe inward tail by projecting the root radially away
                # from the table. This prevents reset-time arm/table overlap.
                if table_distance < 1e-6:
                    table_to_robot_x = -math.cos(approach_yaw)
                    table_to_robot_y = -math.sin(approach_yaw)
                    table_distance = 1.0
                scale = placement_min_table_distance / table_distance
                robot_x = float(table_xy[0]) + table_to_robot_x * scale
                robot_y = float(table_xy[1]) + table_to_robot_y * scale
                approach_yaw = math.atan2(
                    float(table_xy[1] - robot_y),
                    float(table_xy[0] - robot_x),
                )

            if (
                workbench_xy_for_clearance is not None
                and outward_xy_for_clearance is not None
            ):
                outward_distance = (
                    (robot_x - workbench_xy_for_clearance[0])
                    * outward_xy_for_clearance[0]
                    + (robot_y - workbench_xy_for_clearance[1])
                    * outward_xy_for_clearance[1]
                )
                clearance = outward_distance - workbench_half_depth
                if clearance < min_workbench_clearance:
                    correction = min_workbench_clearance - clearance
                    robot_x += correction * outward_xy_for_clearance[0]
                    robot_y += correction * outward_xy_for_clearance[1]
                    approach_yaw = math.atan2(
                        float(table_xy[1] - robot_y),
                        float(table_xy[0] - robot_x),
                    )

            robot_pos_local[i] = torch.tensor(
                [robot_x, robot_y, base_height],
                dtype=table_pos_local.dtype,
                device=env.device,
            )
            robot_yaw[i] = approach_yaw + yaw_offset + heading_jitter

        return robot_pos_local, robot_yaw

    def _prepare_robot_default_pose_for_reset(self, env, env_ids=None) -> None:
        if not _env_bool("FLIP_TABLE_PREPARE_ROBOT_DEFAULT_ON_RESET", True):
            return
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        for env_id in env_ids.tolist():
            self._teleop_randomization_samples[int(env_id)] = {
                "profile_level": _rl_randomization_level(),
                "policy_uses_privileged_values": False,
            }

        table_name, table_root_prim_path, table_pos_w, table_quat_xyzw = self._table_pose_root(env)
        del table_name, table_root_prim_path, table_quat_xyzw
        if table_pos_w is None:
            return
        if self._base_table_pos_local is None:
            self._base_table_pos_local = (table_pos_w - env.scene.env_origins).clone()

        workbench_pos_w, workbench_quat_xyzw = self._workbench_pose(env)
        if workbench_pos_w is not None and workbench_quat_xyzw is not None:
            if self._base_workbench_pos_local is None or self._base_workbench_quat is None:
                self._base_workbench_pos_local = (workbench_pos_w - env.scene.env_origins).clone()
                self._base_workbench_quat = workbench_quat_xyzw.clone()

        table_pos_local = self._base_table_pos_local[env_ids].clone()
        workbench_pos_local = None
        workbench_quat = None
        if self._base_workbench_pos_local is not None and self._base_workbench_quat is not None:
            workbench_pos_local = self._base_workbench_pos_local[env_ids].clone()
            workbench_quat = self._base_workbench_quat[env_ids].clone()

        robot_pos_local, robot_yaw = self._sample_robot_pose_for_workbench_front(
            env,
            env_ids,
            table_pos_local,
            workbench_pos_local,
            workbench_quat,
        )

        robot = self._robot(env)
        data = getattr(robot, "data", None)
        if robot is None or data is None:
            return
        root_quat = _yaw_quat_xyzw(robot_yaw)
        default_root_pose = getattr(data, "default_root_pose", None)
        if default_root_pose is not None and hasattr(default_root_pose, "torch"):
            default_root_pose.torch[env_ids, :3] = robot_pos_local
            default_root_pose.torch[env_ids, 3:7] = root_quat
        if hasattr(data, "default_joint_pos"):
            x_joint_id, y_joint_id, yaw_joint_id = self._find_robot_base_joint_ids(robot)
            if x_joint_id is not None and y_joint_id is not None and yaw_joint_id is not None:
                default_joint_pos = as_torch(data.default_joint_pos).to(device=env.device)
                default_joint_pos[env_ids[:, None], torch.tensor([x_joint_id, y_joint_id, yaw_joint_id], device=env.device)] = 0.0

        if _verbose_reset_logs():
            for row, env_id in enumerate(env_ids.tolist()):
                print(
                    "[FlipTableEvalTask] prepared robot default root: "
                    f"env={env_id}, root=({float(robot_pos_local[row, 0]):.3f}, "
                    f"{float(robot_pos_local[row, 1]):.3f}, {float(robot_pos_local[row, 2]):.3f}), "
                    f"yaw={float(robot_yaw[row]):.3f}",
                    flush=True,
                )

    def _reset_flip_table_scene(self, env, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.int64)
        if env_ids.numel() == 0:
            return
        for env_id in env_ids.tolist():
            self._teleop_randomization_samples[int(env_id)] = {
                "profile_level": _rl_randomization_level(),
                "policy_uses_privileged_values": False,
            }

        table_name, table_root_prim_path, table_pos_w, table_quat_xyzw = self._table_pose_root(env)
        if table_pos_w is None or table_quat_xyzw is None:
            print(
                "[FlipTableEvalTask] table root not found; leaving default reset. "
                f"entities: {self._scene_entity_names_for_log(env)}",
                flush=True,
            )
            return

        if self._base_table_pos_local is None or self._base_table_quat is None:
            self._base_table_pos_local = (table_pos_w - env.scene.env_origins).clone()
            self._base_table_quat = table_quat_xyzw.clone()

        if self._base_leg_positions_local is None or self._base_leg_quats is None:
            base_leg_positions_local = []
            base_leg_quats = []
            for leg_name, leg_prim_path, _leg_site_path in self.leg_reg_int_sites:
                leg_pos, leg_quat = self._extract_object_pose(env, leg_name, leg_prim_path)
                if leg_pos is None or leg_quat is None:
                    print(f"[FlipTableEvalTask] leg pose unavailable for {leg_name}", flush=True)
                    return
                base_leg_positions_local.append((leg_pos - env.scene.env_origins).clone())
                base_leg_quats.append(leg_quat.clone())
            self._base_leg_positions_local = tuple(base_leg_positions_local)
            self._base_leg_quats = tuple(base_leg_quats)

        workbench_pos_w, workbench_quat_xyzw = self._workbench_pose(env)
        if workbench_pos_w is not None and workbench_quat_xyzw is not None:
            if self._base_workbench_pos_local is None or self._base_workbench_quat is None:
                self._base_workbench_pos_local = (workbench_pos_w - env.scene.env_origins).clone()
                self._base_workbench_quat = workbench_quat_xyzw.clone()

        env_origins = env.scene.env_origins[env_ids]
        base_table_pos_local = self._base_table_pos_local[env_ids].clone()
        authored_table_quat = self._base_table_quat[env_ids].clone()
        # The organizer assembly scene stores the tabletop on its side.  The
        # source demonstrations begin with the assembled table upside down on
        # the workbench, whose exact body-frame orientation is a pi rotation
        # about X.  Yaw randomization is applied on top of this canonical pose.
        base_table_quat = torch.zeros_like(authored_table_quat)
        base_table_quat[:, 0] = 1.0
        count = env_ids.numel()

        workbench_pos_local = None
        workbench_quat = None
        if self._base_workbench_pos_local is not None and self._base_workbench_quat is not None:
            workbench_pos_local = self._base_workbench_pos_local[env_ids].clone()
            workbench_quat = self._base_workbench_quat[env_ids].clone()

        table_pos_local, table_quat, yaw_delta, randomization_meta = self._sample_table_pose_on_workbench(
            env,
            base_table_pos_local,
            base_table_quat,
            workbench_pos_local,
            workbench_quat,
        )
        table_pose = torch.cat([table_pos_local + env_origins, table_quat], dim=-1)
        self._write_object_pose(env, table_name, table_root_prim_path, table_pose, env_ids)
        for row, env_id in enumerate(env_ids.tolist()):
            self._teleop_randomization(env_id)["table"] = {
                "position_local_m": [float(value) for value in table_pos_local[row].tolist()],
                "quaternion_xyzw": [float(value) for value in table_quat[row].tolist()],
                "yaw_delta_rad": float(yaw_delta[row]),
                "configured_ranges": dict(randomization_meta),
            }

        assembly_delta_quat = _quat_mul_xyzw(
            table_quat,
            _quat_conjugate_xyzw(authored_table_quat),
        )
        assembly_delta_rot = matrix_from_quat(assembly_delta_quat)

        for leg_index, (leg_name, leg_prim, _leg_site) in enumerate(self.leg_reg_int_sites):
            leg_root_prim_path = self._pose_root_prim_path(env, leg_name, leg_prim)
            # Apply one rigid transform to the complete assembly.  Transforming
            # the tabletop and legs independently changes the authored fixed-
            # joint frames and destabilizes PhysX.
            base_leg_pos_local = self._base_leg_positions_local[leg_index][env_ids]
            base_leg_quat = self._base_leg_quats[leg_index][env_ids]
            leg_quat = _quat_mul_xyzw(assembly_delta_quat, base_leg_quat)
            relative_leg_pos = base_leg_pos_local - base_table_pos_local
            leg_root = (
                table_pos_local
                + torch.bmm(assembly_delta_rot, relative_leg_pos.unsqueeze(-1)).squeeze(-1)
                + env_origins
            )
            leg_pose = torch.cat([leg_root, leg_quat], dim=-1)
            self._write_object_pose(env, leg_name, leg_root_prim_path, leg_pose, env_ids)

        self._ensure_assembled_table_joints(env)
        self._disable_workbench_collision_for_diagnosis(env)
        self._ensure_robot_visual_materials(env)
        self._randomize_contact_materials(env, env_ids)
        self._randomize_object_masses(env, env_ids)
        if _env_bool("FLIP_TABLE_USE_DEFAULT_ROBOT_POSE", False):
            self._log_robot_default_pose(env, env_ids)
            self._apply_default_robot_pose_offset(env, env_ids)
            room_robot_pos_local, _room_robot_quat = self._robot_root_pose_local(env, env_ids)
        else:
            # The organizer reset restores the robot from ``init_state`` after
            # reset event terms have prepared ``default_root_pose``.  Apply the
            # selected world pose after that reset so the actual USD/PhysX root,
            # not only the cached default, is moved in the scene.
            robot_pos_local, robot_yaw = self._sample_robot_pose_for_workbench_front(
                env,
                env_ids,
                table_pos_local,
                workbench_pos_local,
                workbench_quat,
                yaw_delta,
            )
            for row, env_id in enumerate(env_ids.tolist()):
                self._set_robot_root_planar_pose(
                    env,
                    env_id,
                    float(robot_pos_local[row, 0]),
                    float(robot_pos_local[row, 1]),
                    float(robot_yaw[row]),
                )
            self._randomize_robot_joints(env, env_ids)
            room_robot_pos_local = robot_pos_local
        self._randomize_upper_body_joint_properties(env, env_ids)
        room_robot_yaw = _yaw_from_quat_xyzw(
            self._robot_root_pose_local(env, env_ids)[1]
        )
        for row, env_id in enumerate(env_ids.tolist()):
            self._teleop_randomization(env_id)["robot"] = {
                "position_local_m": [
                    float(value) for value in room_robot_pos_local[row].tolist()
                ],
                "yaw_rad": float(room_robot_yaw[row]),
            }
        self._randomize_policy_camera_mounts(env, env_ids)
        self._sample_camera_image_model(env, env_ids)
        self._randomize_lighting(env, env_ids)
        self._randomize_room_background(
            env,
            env_ids,
            workbench_pos_local,
            room_robot_pos_local,
        )

        env.scene.write_data_to_sim()
        env.sim.forward()
        self._sync_robot_root_to_planar_base(env, env_ids)
        env.sim.forward()
        self._capture_lower_body_lock(env, env_ids)
        self._apply_lower_body_lock(env, env_ids)
        env.sim.forward()
        self._log_robot_visual_pose(env, env_ids, "reset final")
        # CameraSensor keeps a Fabric-side pose cache.  A root pose authored
        # during reset is visible in USD immediately, but the first rendered
        # RGB can otherwise use the pre-reset camera transform.  Render once,
        # refresh all policy sensors, then render the synchronized frame before
        # the evaluator asks the policy for its first observation.
        env.sim.render()
        self._refresh_camera_sensors(env)
        env.sim.render()
        self._log_runtime_analysis_pose(env, env_ids)

        reset_table_pos, reset_table_quat = self._table_body_pose(env)
        if reset_table_pos is not None and reset_table_quat is not None:
            rot = matrix_from_quat(reset_table_quat[env_ids])
            if (
                self._initial_table_normal is None
                or self._initial_table_normal.shape != (env.num_envs, 3)
            ):
                self._initial_table_normal = torch.zeros(
                    (env.num_envs, 3), dtype=rot.dtype, device=rot.device
                )
            self._initial_table_normal[env_ids] = rot[:, :, 2]
            if self._initial_table_pos is None or self._initial_table_pos.shape[0] != env.num_envs:
                self._initial_table_pos = torch.zeros(
                    (env.num_envs, 3), dtype=reset_table_pos.dtype, device=reset_table_pos.device
                )
            self._initial_table_pos[env_ids] = reset_table_pos[env_ids, :3]
        self._success_cache[env_ids] = 0
        self._success_flag[env_ids] = False
        self._stable_success_streak[env_ids] = 0
        self._stable_success_result[env_ids] = False
        self._stable_success_previous_candidate[env_ids] = False
        if _verbose_reset_logs():
            print(
                "[FlipTableEvalTask] reset applied: "
                f"table_root={table_root_prim_path}, envs={env_ids.tolist()}, "
                f"table_long_range={randomization_meta['table_long_range_m']:.3f}, "
                f"table_depth_range={randomization_meta['table_depth_range_m']:.3f}, "
                f"yaw_range={randomization_meta['table_yaw_range_rad']:.3f}, "
                f"table_pos_local={[[round(float(value), 5) for value in row] for row in table_pos_local.detach().cpu().tolist()]}, "
                f"table_yaw_delta={[round(float(value), 5) for value in yaw_delta.detach().cpu().tolist()]}",
                flush=True,
            )

    def _find_robot_base_joint_ids(self, robot) -> tuple[int | None, int | None, int | None]:
        data = getattr(robot, "data", None)
        joint_names = list(getattr(data, "joint_names", getattr(robot, "joint_names", [])))

        def find_joint_id(*candidates: str) -> int | None:
            for candidate in candidates:
                if candidate in joint_names:
                    return joint_names.index(candidate)
            return None

        return (
            find_joint_id("base_x_joint", "base_x"),
            find_joint_id("base_y_joint", "base_y"),
            find_joint_id("base_yaw_joint", "base_yaw"),
        )

    def _log_robot_default_pose(self, env, env_ids: torch.Tensor) -> None:
        robot = self._robot(env)
        if robot is None or not hasattr(robot, "data"):
            print("[FlipTableEvalTask] using default robot pose: robot articulation unavailable", flush=True)
            return
        data = robot.data
        joint_names = list(getattr(data, "joint_names", getattr(robot, "joint_names", [])))
        joint_pos = as_torch(data.joint_pos) if hasattr(data, "joint_pos") else None
        for env_id in env_ids.tolist():
            base_values = []
            if joint_pos is not None:
                for name in ("base_x_joint", "base_y_joint", "base_yaw_joint"):
                    if name in joint_names:
                        base_values.append(f"{name}={float(joint_pos[env_id, joint_names.index(name)]):.3f}")
            print(
                "[FlipTableEvalTask] using organizer default robot pose: "
                f"env={env_id}, {', '.join(base_values) if base_values else 'base joints unavailable'}",
                flush=True,
            )

    def _apply_default_robot_pose_offset(self, env, env_ids: torch.Tensor) -> None:
        grid = _env_float("FLIP_TABLE_DEBUG_GRID_CELL_M", 0.25)
        right_cells = _env_float("FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS", 0.0)
        forward_cells = _env_float("FLIP_TABLE_DEFAULT_ROBOT_FORWARD_CELLS", 0.0)
        yaw_offset = _env_float("FLIP_TABLE_DEFAULT_ROBOT_YAW_OFFSET_RAD", 0.0)
        if grid == 0.0 or (right_cells == 0.0 and forward_cells == 0.0 and yaw_offset == 0.0):
            return

        root_pos_local, root_quat = self._robot_root_pose_local(env, env_ids)
        root_yaw = _yaw_from_quat_xyzw(root_quat)

        for row, env_id in enumerate(env_ids.tolist()):
            start_x = float(root_pos_local[row, 0])
            start_y = float(root_pos_local[row, 1])
            start_z = float(root_pos_local[row, 2])
            yaw = float(root_yaw[row])
            target_yaw = yaw + yaw_offset
            forward = (math.cos(yaw), math.sin(yaw))
            right = (math.sin(yaw), -math.cos(yaw))
            dx = grid * (forward_cells * forward[0] + right_cells * right[0])
            dy = grid * (forward_cells * forward[1] + right_cells * right[1])
            root_pos_local[row, 0] = start_x + dx
            root_pos_local[row, 1] = start_y + dy
            root_yaw[row] = target_yaw
            print(
                "[FlipTableEvalTask] default robot offset applied: "
                f"env={env_id}, grid={grid:.3f}, right_cells={right_cells:.3f}, "
                f"forward_cells={forward_cells:.3f}, yaw_offset={yaw_offset:.3f}, "
                f"root=({start_x:.3f}, {start_y:.3f}, {start_z:.3f}, {yaw:.3f}) -> "
                f"({float(root_pos_local[row, 0]):.3f}, "
                f"{float(root_pos_local[row, 1]):.3f}, {start_z:.3f}, {target_yaw:.3f})",
                flush=True,
            )

        self._write_robot_root_pose(env, env_ids, root_pos_local, root_yaw)
        self._zero_robot_planar_base_joints(env, env_ids)

    def _robot_root_z_local(self, env, env_id: int) -> float:
        robot = self._robot(env)
        data = getattr(robot, "data", None)
        root_state = None
        for attr in ("root_state_w", "root_link_state_w"):
            if data is not None and hasattr(data, attr):
                root_state = as_torch(getattr(data, attr)).to(device=env.device)
                break
        if root_state is None or root_state.shape[-1] < 3:
            return _env_float("FLIP_TABLE_ROBOT_BASE_HEIGHT_M", 0.78)
        z_world = float(root_state[env_id, 2])
        origin_z = float(env.scene.env_origins[env_id, 2])
        return z_world - origin_z

    def _robot_root_pose_local(self, env, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        robot = self._robot(env)
        data = getattr(robot, "data", None)
        root_state = None
        for attr in ("root_state_w", "root_link_state_w"):
            if data is not None and hasattr(data, attr):
                root_state = as_torch(getattr(data, attr)).to(device=env.device)
                break
        if root_state is not None and root_state.shape[-1] >= 7:
            pos_local = root_state[env_ids, :3] - env.scene.env_origins[env_ids]
            quat = sim_quat_raw_to_xyzw_torch(root_state[env_ids, 3:7])
            return pos_local.clone(), quat.clone()

        stage_pos, stage_quat = self._extract_stage_prim_pose(env, "Robot")
        if stage_pos is not None and stage_quat is not None:
            return (
                (stage_pos[env_ids] - env.scene.env_origins[env_ids]).clone(),
                stage_quat[env_ids].clone(),
            )

        pos = torch.zeros((env_ids.numel(), 3), dtype=torch.float32, device=env.device)
        pos[:, 2] = _env_float("FLIP_TABLE_ROBOT_BASE_HEIGHT_M", 0.78)
        quat = _yaw_quat_xyzw(torch.zeros((env_ids.numel(),), dtype=torch.float32, device=env.device))
        return pos, quat

    def _write_robot_root_pose(
        self,
        env,
        env_ids: torch.Tensor,
        root_pos_local: torch.Tensor,
        root_yaw: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        env_ids_i32 = env_ids.to(dtype=torch.int32)
        root_pos_w = root_pos_local.to(device=env.device, dtype=torch.float32) + env.scene.env_origins[env_ids]
        root_quat = _yaw_quat_xyzw(root_yaw.to(device=env.device, dtype=torch.float32))
        root_pose = torch.cat([root_pos_w, root_quat], dim=-1)

        robot = self._robot(env)
        if robot is not None and hasattr(robot, "write_root_pose_to_sim"):
            robot.write_root_pose_to_sim(root_pose, env_ids=env_ids_i32)
        if robot is not None and hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(
                torch.zeros((env_ids.numel(), 6), dtype=root_pose.dtype, device=env.device),
                env_ids=env_ids_i32,
            )

        for row, env_id in enumerate(env_ids.tolist()):
            prim = self._find_prim_by_suffix(env, "Robot", env_id=env_id)
            if prim is None:
                continue
            self._set_stage_prim_world_pose(env, prim, root_pose[row, :3], root_pose[row, 3:7])

    def _zero_robot_planar_base_joints(self, env, env_ids: torch.Tensor) -> bool:
        robot = self._robot(env)
        if robot is None or not hasattr(robot, "data") or not hasattr(robot, "write_joint_state_to_sim"):
            return False
        data = robot.data
        x_joint_id, y_joint_id, yaw_joint_id = self._find_robot_base_joint_ids(robot)
        if x_joint_id is None or y_joint_id is None or yaw_joint_id is None:
            return False
        if not hasattr(data, "joint_pos") or not hasattr(data, "joint_vel"):
            return False

        env_ids_i32 = env_ids.to(device=env.device, dtype=torch.int32)
        env_ids_long = env_ids_i32.to(dtype=torch.long)
        joint_pos = as_torch(data.joint_pos)[env_ids_long].clone()
        joint_vel = as_torch(data.joint_vel)[env_ids_long].clone()
        base_joint_ids = torch.tensor(
            [x_joint_id, y_joint_id, yaw_joint_id],
            device=env.device,
            dtype=torch.long,
        )
        joint_pos[:, base_joint_ids] = 0.0
        joint_vel[:, base_joint_ids] = 0.0
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_i32)
        if hasattr(robot, "set_joint_position_target"):
            robot.set_joint_position_target(
                torch.zeros((env_ids.numel(), 3), dtype=joint_pos.dtype, device=env.device),
                joint_ids=base_joint_ids.to(dtype=torch.int32),
                env_ids=env_ids_i32,
            )
        return True

    def _set_robot_root_planar_pose(self, env, env_id: int, x: float, y: float, yaw: float) -> None:
        z = self._robot_root_z_local(env, env_id)
        one_id = torch.tensor([env_id], device=env.device, dtype=torch.long)
        root_pos_local = torch.tensor([[x, y, z]], dtype=torch.float32, device=env.device)
        root_yaw = torch.tensor([yaw], dtype=torch.float32, device=env.device)
        self._write_robot_root_pose(env, one_id, root_pos_local, root_yaw)
        if _verbose_reset_logs():
            print(
                "[FlipTableEvalTask] robot root pose set: "
                f"env={env_id}, root=({x:.3f}, {y:.3f}, {z:.3f}), yaw={yaw:.3f}",
                flush=True,
            )

    def _sync_robot_root_to_planar_base(self, env, env_ids: torch.Tensor) -> None:
        # The G1 local scene uses base_x/base_y/base_yaw as internal planar joints.
        # The world placement must be the Robot root pose; treating those joint
        # values as world coordinates leaves the rendered robot at the spawn pose.
        self._zero_robot_planar_base_joints(env, env_ids)

    def _log_robot_visual_pose(self, env, env_ids: torch.Tensor, label: str) -> None:
        if not _verbose_reset_logs():
            return
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        suffixes = (
            "Robot",
            "Robot/base_x",
            "Robot/base_y",
            "Robot/base_yaw",
            "Robot/world",
            "Robot/pelvis",
            "Robot/torso_link",
            "Robot/torso_link/first_person_camera",
            "Table001/Table001_01",
            "global_camera",
        )
        for env_id in env_ids.tolist():
            parts = []
            for suffix in suffixes:
                prim = self._find_prim_by_suffix(env, suffix, env_id=env_id)
                if prim is None:
                    continue
                transform = cache.GetLocalToWorldTransform(prim)
                pos = transform.ExtractTranslation()
                pose = f"{suffix}=({float(pos[0]):.6f},{float(pos[1]):.6f},{float(pos[2]):.6f})"
                quat = transform.ExtractRotationQuat()
                imag = quat.GetImaginary()
                pose += (
                    ",quat_xyzw=("
                    f"{float(imag[0]):.8f},{float(imag[1]):.8f},"
                    f"{float(imag[2]):.8f},{float(quat.GetReal()):.8f})"
                )
                parts.append(pose)
            if parts:
                print(
                    f"[FlipTableEvalTask] robot visual pose {label}: env={env_id}, "
                    + "; ".join(parts),
                    flush=True,
                )

    def _log_runtime_analysis_pose(self, env, env_ids: torch.Tensor) -> None:
        """Log tensor-backed poses for post-run audits; policy observations never include them."""

        if not _verbose_reset_logs():
            return
        robot = self._find_scene_entity(env, "robot")
        if robot is None:
            robot = getattr(env.scene, "robot", None)
        robot_data = getattr(robot, "data", None)
        body_names = list(getattr(robot_data, "body_names", getattr(robot, "body_names", [])))
        body_pos_w = getattr(robot_data, "body_pos_w", None)
        body_quat_w = getattr(robot_data, "body_quat_w", None)
        joint_names = list(getattr(robot_data, "joint_names", getattr(robot, "joint_names", [])))
        joint_pos = getattr(robot_data, "joint_pos", None)
        camera = getattr(env.scene, "sensors", {}).get("first_person_camera")
        camera_data = getattr(camera, "data", None)
        table_pos_w, table_quat_xyzw = self._table_body_pose(env)

        for env_id in env_ids.tolist():
            payload = {
                "env": env_id,
                "robot_bodies": {},
                "robot_joints": {},
                "head_camera": {},
                "table": {},
            }
            if body_pos_w is not None and body_quat_w is not None:
                for body_name in ("pelvis", "torso_link"):
                    if body_name not in body_names:
                        continue
                    body_id = body_names.index(body_name)
                    payload["robot_bodies"][body_name] = {
                        "position_w": [float(value) for value in body_pos_w[env_id, body_id].tolist()],
                        "quaternion_w_raw": [
                            float(value) for value in body_quat_w[env_id, body_id].tolist()
                        ],
                    }
            if joint_pos is not None:
                payload["robot_joints"] = {
                    "names": joint_names,
                    "positions": [float(value) for value in joint_pos[env_id].tolist()],
                }
            if camera_data is not None:
                for attribute in ("pos_w", "quat_w_world", "quat_w_ros", "quat_w_opengl"):
                    value = getattr(camera_data, attribute, None)
                    if value is not None:
                        payload["head_camera"][attribute] = [
                            float(item) for item in value[env_id].tolist()
                        ]
            if table_pos_w is not None and table_quat_xyzw is not None:
                payload["table"] = {
                    "position_w": [float(value) for value in table_pos_w[env_id].tolist()],
                    "quaternion_xyzw": [
                        float(value) for value in table_quat_xyzw[env_id].tolist()
                    ],
                }
            print(
                "[FlipTableEvalTask] runtime analysis pose: "
                + json.dumps(payload, separators=(",", ":")),
                flush=True,
            )

    def _set_robot_planar_base_pose(self, env, env_ids: torch.Tensor, x: float, y: float, yaw: float) -> bool:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        for env_id in env_ids.tolist():
            self._set_robot_root_planar_pose(env, env_id, x, y, yaw)
        self._zero_robot_planar_base_joints(env, env_ids)
        return True

    def _refresh_camera_sensors(self, env) -> None:
        sensor_names = (
            "first_person_camera",
            "head_right_camera",
            "left_hand_camera",
            "right_hand_camera",
            "hand_camera",
            "global_camera",
            "d435_camera",
        )
        sensors = getattr(env.scene, "sensors", {})
        dt = float(getattr(env, "physics_dt", 0.0))
        for name in sensor_names:
            sensor = sensors.get(name)
            if sensor is None:
                continue
            cfg = getattr(sensor, "cfg", None)
            if cfg is not None and hasattr(cfg, "update_latest_camera_pose"):
                cfg.update_latest_camera_pose = True
            try:
                sensor.update(dt, force_recompute=True)
            except TypeError:
                sensor.update(dt)

    def _robot(self, env):
        return getattr(env.scene, "articulations", {}).get("robot")

    def _resolve_lower_body_joint_ids(self, robot, device: torch.device) -> torch.Tensor:
        if self._lower_body_lock_joint_ids is not None:
            return self._lower_body_lock_joint_ids
        data = getattr(robot, "data", None)
        joint_names = list(getattr(data, "joint_names", getattr(robot, "joint_names", [])))
        patterns = tuple(
            part.strip().lower()
            for part in os.environ.get(
                "FLIP_TABLE_LOWER_BODY_LOCK_PATTERNS",
                "base_,hip,knee,ankle,waist",
            ).split(",")
            if part.strip()
        )
        indices = []
        names = []
        for index, name in enumerate(joint_names):
            lowered = str(name).lower()
            if any(pattern in lowered for pattern in patterns):
                indices.append(index)
                names.append(str(name))
        self._lower_body_lock_joint_names = tuple(names)
        self._lower_body_lock_joint_ids = torch.tensor(indices, dtype=torch.long, device=device)
        if _env_bool("FLIP_TABLE_REQUIRE_WAIST_LOCK", False):
            required_waist = (
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
            )
            missing_waist = [name for name in required_waist if name not in names]
            if missing_waist:
                raise RuntimeError(
                    "FLIP_TABLE_REQUIRE_WAIST_LOCK=true but the lower-body lock "
                    "does not include "
                    f"{missing_waist}; patterns={patterns}, matched={names}"
                )
        if not self._lower_body_lock_logged:
            self._lower_body_lock_logged = True
            print(
                "[FlipTableEvalTask] lower-body lock joints: "
                + (", ".join(names) if names else "(none found)"),
                flush=True,
            )
        return self._lower_body_lock_joint_ids

    def _capture_lower_body_lock(self, env, env_ids: torch.Tensor) -> None:
        if self._sim_body_mode != "fixed_diagnostic" or not _env_bool("FLIP_TABLE_LOCK_LOWER_BODY", False):
            return
        robot = self._robot(env)
        if robot is None or not hasattr(robot, "data"):
            return
        data = robot.data
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        lower_ids = self._resolve_lower_body_joint_ids(robot, env.device)

        if lower_ids.numel() > 0 and hasattr(data, "joint_pos"):
            joint_pos = as_torch(data.joint_pos).to(device=env.device)
            if self._lower_body_lock_joint_pos is None or self._lower_body_lock_joint_pos.shape != (
                env.num_envs,
                lower_ids.numel(),
            ):
                self._lower_body_lock_joint_pos = torch.zeros(
                    (env.num_envs, lower_ids.numel()),
                    dtype=joint_pos.dtype,
                    device=env.device,
                )
            self._lower_body_lock_joint_pos[env_ids] = joint_pos[env_ids][:, lower_ids].clone()

        if _env_bool("FLIP_TABLE_LOCK_ROBOT_ROOT", False):
            root_state = None
            for attr in ("root_state_w", "root_link_state_w"):
                if hasattr(data, attr):
                    root_state = as_torch(getattr(data, attr)).to(device=env.device)
                    break
            if root_state is not None and root_state.shape[-1] >= 7:
                if self._lower_body_lock_root_pose is None or self._lower_body_lock_root_pose.shape != (
                    env.num_envs,
                    7,
                ):
                    self._lower_body_lock_root_pose = torch.zeros(
                        (env.num_envs, 7),
                        dtype=root_state.dtype,
                        device=env.device,
                    )
                self._lower_body_lock_root_pose[env_ids] = root_state[env_ids, :7].clone()

    def _apply_lower_body_lock(self, env, env_ids=None) -> None:
        if self._sim_body_mode != "fixed_diagnostic" or not _env_bool("FLIP_TABLE_LOCK_LOWER_BODY", False):
            return
        robot = self._robot(env)
        if robot is None or not hasattr(robot, "data"):
            return
        data = robot.data
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        else:
            env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        env_ids_i32 = env_ids.to(dtype=torch.int32)

        robot_cfg = getattr(robot, "cfg", None)
        articulation_cfg = getattr(robot_cfg, "articulation_props", None)
        if articulation_cfg is None:
            articulation_cfg = getattr(getattr(robot_cfg, "spawn", None), "articulation_props", None)
        native_root_lock = bool(getattr(articulation_cfg, "fix_root_link", False))
        if self._lower_body_lock_root_pose is not None and not native_root_lock:
            root_pose = self._lower_body_lock_root_pose[env_ids].clone()
            if hasattr(robot, "write_root_pose_to_sim"):
                robot.write_root_pose_to_sim(root_pose, env_ids=env_ids_i32)
            if hasattr(robot, "write_root_velocity_to_sim"):
                robot.write_root_velocity_to_sim(
                    torch.zeros((env_ids.numel(), 6), dtype=root_pose.dtype, device=env.device),
                    env_ids=env_ids_i32,
                )
            for row, env_id in enumerate(env_ids.tolist()):
                prim = self._find_prim_by_suffix(env, "Robot", env_id=env_id)
                if prim is not None:
                    self._set_stage_prim_world_pose(env, prim, root_pose[row, :3], root_pose[row, 3:7])

        lower_ids = self._resolve_lower_body_joint_ids(robot, env.device)
        if lower_ids.numel() == 0 or self._lower_body_lock_joint_pos is None:
            return
        if not hasattr(robot, "write_joint_state_to_sim"):
            return
        locked_pos = self._lower_body_lock_joint_pos[env_ids]
        locked_vel = torch.zeros_like(locked_pos)
        lower_ids_i32 = lower_ids.to(dtype=torch.int32)
        robot.write_joint_state_to_sim(
            locked_pos,
            locked_vel,
            joint_ids=lower_ids_i32,
            env_ids=env_ids_i32,
        )
        if hasattr(robot, "set_joint_position_target"):
            robot.set_joint_position_target(
                locked_pos,
                joint_ids=lower_ids_i32,
                env_ids=env_ids_i32,
            )
        if hasattr(robot, "set_joint_velocity_target"):
            robot.set_joint_velocity_target(
                locked_vel,
                joint_ids=lower_ids_i32,
                env_ids=env_ids_i32,
            )

    def _randomize_robot_joints(self, env, env_ids: torch.Tensor) -> None:
        noise = _env_float("FLIP_TABLE_JOINT_NOISE_RAD", 0.02)
        randomize_upper_body = _env_bool("FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE", True)
        upper_body_scale = _env_float("FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE", 1.0)
        finger_noise_m = _env_float("FLIP_TABLE_DEX1_FINGER_NOISE_M", 0.002)
        if upper_body_scale < 0:
            raise ValueError("FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE must be non-negative")
        if finger_noise_m < 0:
            raise ValueError("FLIP_TABLE_DEX1_FINGER_NOISE_M must be non-negative")
        use_dataset_pose = _env_bool("FLIP_TABLE_USE_DATASET_INITIAL_UPPER_BODY", True)
        replay_initial_state = _env_float_vector("FLIP_TABLE_INITIAL_UPPER_BODY_STATE", 19)
        full_body_initial_state = _env_float_vector("FLIP_TABLE_INITIAL_FULL_BODY_STATE", 31)
        # Kept only to make existing diagnostic output fail safely rather than
        # silently changing its scene initialization. New calibration runs
        # must use measured state, never the first q_desired target.
        deprecated_initial_action = _env_float_vector("FLIP_TABLE_INITIAL_UPPER_BODY_ACTION", 19)
        if replay_initial_state is not None and full_body_initial_state is not None:
            raise ValueError(
                "set only one of FLIP_TABLE_INITIAL_UPPER_BODY_STATE or "
                "FLIP_TABLE_INITIAL_FULL_BODY_STATE"
            )
        if full_body_initial_state is not None and self._sim_body_mode != "full_body_diagnostic":
            raise ValueError(
                "FLIP_TABLE_INITIAL_FULL_BODY_STATE requires "
                "FLIP_TABLE_SIM_BODY_MODE=full_body_diagnostic"
            )
        if replay_initial_state is not None and deprecated_initial_action is not None:
            raise ValueError(
                "set only FLIP_TABLE_INITIAL_UPPER_BODY_STATE; "
                "FLIP_TABLE_INITIAL_UPPER_BODY_ACTION is deprecated"
            )
        if deprecated_initial_action is not None:
            raise ValueError(
                "FLIP_TABLE_INITIAL_UPPER_BODY_ACTION is not valid for real-to-sim replay; "
                "use measured FLIP_TABLE_INITIAL_UPPER_BODY_STATE"
            )
        if (
            noise <= 0
            and not randomize_upper_body
            and not use_dataset_pose
            and replay_initial_state is None
            and full_body_initial_state is None
        ):
            return
        robot = self._robot(env)
        if robot is None or not hasattr(robot, "data") or not hasattr(robot, "write_joint_state_to_sim"):
            return
        data = robot.data
        if not hasattr(data, "default_joint_pos"):
            return
        env_ids_i32 = env_ids.to(dtype=torch.int32)
        current_joint_pos = as_torch(data.joint_pos)[env_ids].clone()
        joint_pos = as_torch(data.default_joint_pos)[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        lower_ids = self._resolve_lower_body_joint_ids(robot, env.device)
        joint_names = list(getattr(data, "joint_names", getattr(robot, "joint_names", [])))
        joint_noise = torch.zeros_like(joint_pos)
        if noise > 0 and not randomize_upper_body:
            # Backward-compatible uniform noise for explicit diagnostics. Normal
            # evaluation uses the safer joint-specific ranges below.
            joint_noise = _uniform(tuple(joint_pos.shape), -noise, noise, device=env.device, dtype=joint_pos.dtype)
        if lower_ids.numel() > 0 and full_body_initial_state is None:
            joint_pos[:, lower_ids] = current_joint_pos[:, lower_ids]
            joint_noise[:, lower_ids] = 0.0
        if full_body_initial_state is not None:
            # Full-body source q_current is written once at reset.  The root
            # remains dynamic for the entire replay; no per-frame state write
            # or lower-body lock is used by this diagnostic.
            for joint_name, value in zip(
                FLIP_TABLE_FULL_BODY_ACTION_JOINT_NAMES, full_body_initial_state[:29]
            ):
                if joint_name not in joint_names:
                    raise ValueError(
                        f"Unknown joint in FLIP_TABLE_INITIAL_FULL_BODY_STATE: {joint_name}"
                    )
                joint_pos[:, joint_names.index(joint_name)] = value
                joint_noise[:, joint_names.index(joint_name)] = 0.0

            hand_open = 0.0245
            hand_close = -0.02
            hand_scale = 4.5
            for hand_offset, side in ((29, "left"), (30, "right")):
                hand_value = max(0.0, min(hand_scale, full_body_initial_state[hand_offset]))
                finger_pos = hand_close + (hand_open - hand_close) * (hand_value / hand_scale)
                for finger_name in (
                    f"{side}_dex1_finger_joint_1",
                    f"{side}_dex1_finger_joint_2",
                ):
                    if finger_name not in joint_names:
                        raise ValueError(
                            f"Unknown joint in FLIP_TABLE_INITIAL_FULL_BODY_STATE: {finger_name}"
                        )
                    joint_pos[:, joint_names.index(finger_name)] = finger_pos
                    joint_noise[:, joint_names.index(finger_name)] = 0.0
        elif replay_initial_state is not None:
            # Source images capture q_current/hand_state.  Initialize from
            # those measurements; q_desired is sent only once replay starts.
            for joint_name, value in zip(FLIP_TABLE_UPPER_BODY_ACTION_JOINT_NAMES, replay_initial_state[:17]):
                if joint_name not in joint_names:
                    raise ValueError(f"Unknown joint in FLIP_TABLE_INITIAL_UPPER_BODY_STATE: {joint_name}")
                joint_id = joint_names.index(joint_name)
                joint_pos[:, joint_id] = value
                joint_noise[:, joint_id] = 0.0

            hand_open = 0.0245
            hand_close = -0.02
            hand_scale = 4.5
            for hand_offset, side in ((17, "left"), (18, "right")):
                hand_value = max(0.0, min(hand_scale, replay_initial_state[hand_offset]))
                finger_pos = hand_close + (hand_open - hand_close) * (hand_value / hand_scale)
                for finger_name in (
                    f"{side}_dex1_finger_joint_1",
                    f"{side}_dex1_finger_joint_2",
                ):
                    if finger_name not in joint_names:
                        raise ValueError(
                            f"Unknown joint in FLIP_TABLE_INITIAL_UPPER_BODY_STATE: {finger_name}"
                        )
                    joint_pos[:, joint_names.index(finger_name)] = finger_pos
                    joint_noise[:, joint_names.index(finger_name)] = 0.0
        elif use_dataset_pose:
            initial_joint_pos = {
                **FLIP_TABLE_DATASET_INITIAL_UPPER_BODY_JOINT_POS,
                **FLIP_TABLE_DATASET_INITIAL_DEX1_FINGER_JOINT_POS,
            }
            for joint_name, value in initial_joint_pos.items():
                if joint_name not in joint_names:
                    continue
                joint_id = joint_names.index(joint_name)
                joint_pos[:, joint_id] = value
                joint_noise[:, joint_id] = 0.0
            for joint_name, offset in _env_named_float_map("FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS").items():
                if joint_name not in joint_names:
                    raise ValueError(f"Unknown joint in FLIP_TABLE_DATASET_INITIAL_JOINT_OFFSETS: {joint_name}")
                joint_pos[:, joint_names.index(joint_name)] += offset
        if randomize_upper_body:
            range_overrides = _env_named_float_map("FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD")
            unknown_names = set(range_overrides) - set(FLIP_TABLE_UPPER_BODY_INITIAL_POSE_RANGES_RAD)
            if unknown_names:
                raise ValueError(
                    "Unknown joint in FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD: "
                    + ", ".join(sorted(unknown_names))
                )
            for joint_name, default_range in FLIP_TABLE_UPPER_BODY_INITIAL_POSE_RANGES_RAD.items():
                if joint_name not in joint_names:
                    continue
                joint_range = range_overrides.get(joint_name, default_range) * upper_body_scale
                if joint_range < 0:
                    raise ValueError(f"Initial pose range for {joint_name} must be non-negative")
                joint_noise[:, joint_names.index(joint_name)] = _uniform(
                    (env_ids.numel(),),
                    -joint_range,
                    joint_range,
                    device=env.device,
                    dtype=joint_pos.dtype,
                )
            for side in ("left", "right"):
                hand_noise = _uniform(
                    (env_ids.numel(),),
                    -finger_noise_m,
                    finger_noise_m,
                    device=env.device,
                    dtype=joint_pos.dtype,
                )
                for finger_index in (1, 2):
                    finger_name = f"{side}_dex1_finger_joint_{finger_index}"
                    if finger_name in joint_names:
                        # A real Dex1 hand receives one open/close command; both
                        # simulated prismatic finger joints must start in sync.
                        joint_noise[:, joint_names.index(finger_name)] = hand_noise
        joint_pos += joint_noise
        if hasattr(data, "soft_joint_pos_limits"):
            limits = as_torch(data.soft_joint_pos_limits)[env_ids]
            if limits.ndim == 3 and limits.shape[-1] == 2:
                joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_i32)
        # State writes do not change an articulation's actuator targets.  Keep
        # both synchronized so a reset cannot execute the previous episode's
        # command during the first simulation tick.
        if hasattr(robot, "set_joint_position_target"):
            robot.set_joint_position_target(joint_pos, env_ids=env_ids_i32)
        if hasattr(robot, "set_joint_velocity_target"):
            robot.set_joint_velocity_target(joint_vel, env_ids=env_ids_i32)

    def _randomize_lighting(self, env, env_ids: torch.Tensor) -> None:
        if not _env_bool("FLIP_TABLE_RANDOMIZE_LIGHTING", True):
            return
        intensity_low, intensity_high = _env_range(
            "FLIP_TABLE_LIGHT_INTENSITY_RANGE", _curriculum_range((450.0, 1200.0))
        )
        indoor_temperature = _env_range(
            "FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K", _curriculum_range((3800.0, 6500.0))
        )
        sun_temperature = _env_range(
            "FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K", _curriculum_range((5000.0, 7000.0))
        )
        sun_intensity = _env_range(
            "FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE", _curriculum_range((180.0, 750.0))
        )
        sun_elevation = _env_range(
            "FLIP_TABLE_SUN_ELEVATION_DEG", _curriculum_range((18.0, 72.0))
        )
        sun_azimuth = _env_range(
            "FLIP_TABLE_SUN_AZIMUTH_DEG", _curriculum_range((-180.0, 180.0))
        )
        light_exposure = _env_range(
            "FLIP_TABLE_LIGHT_EXPOSURE_RANGE", _curriculum_range((-0.35, 0.35))
        )
        try:
            from pxr import Gf, Sdf, UsdGeom, UsdLux

            stage = env.sim.stage
            samples = []

            def link_to_environment(light_prim, env_id: int) -> None:
                target = Sdf.Path(f"/World/envs/env_{env_id}")
                light_api = UsdLux.LightAPI(light_prim)
                for collection in (
                    light_api.GetLightLinkCollectionAPI(),
                    light_api.GetShadowLinkCollectionAPI(),
                ):
                    collection.CreateIncludeRootAttr(False)
                    collection.CreateExpansionRuleAttr("expandPrims")
                    collection.CreateIncludesRel().SetTargets([target])

            for env_id in env_ids.tolist():
                root_path = f"/World/envs/env_{env_id}/FlipTableEvalLighting"
                UsdGeom.Xform.Define(stage, root_path)
                center_x = 0.0
                center_y = 0.0
                if self._base_workbench_pos_local is not None:
                    center_x = float(self._base_workbench_pos_local[env_id, 0])
                    center_y = float(self._base_workbench_pos_local[env_id, 1])

                # Infinite-distance lights would illuminate every clone and make
                # brightness depend on vectorization count. A finite window light
                # keeps reset and rendering isolated to this environment.
                sun = UsdLux.SphereLight.Define(stage, f"{root_path}/WindowSun")
                sun_prim = sun.GetPrim()
                link_to_environment(sun_prim, env_id)
                sun_intensity_value = float(
                    _uniform((1,), sun_intensity[0], sun_intensity[1], device=env.device)[0]
                )
                sun_temperature_value = float(
                    _uniform((1,), sun_temperature[0], sun_temperature[1], device=env.device)[0]
                )
                elevation = float(_uniform((1,), sun_elevation[0], sun_elevation[1], device=env.device)[0])
                azimuth = float(_uniform((1,), sun_azimuth[0], sun_azimuth[1], device=env.device)[0])
                sun_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(sun_intensity_value)
                sun_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(
                    float(_uniform((1,), 0.35, 0.9, device=env.device)[0])
                )
                sun_prim.CreateAttribute("inputs:normalize", Sdf.ValueTypeNames.Bool).Set(True)
                sun_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
                sun_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
                sun_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(
                    sun_temperature_value
                )
                sun_exposure_value = float(
                    _uniform(
                        (1,),
                        light_exposure[0],
                        light_exposure[1],
                        device=env.device,
                    )[0]
                )
                sun_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
                    sun_exposure_value
                )
                sun_distance = float(_env_float("FLIP_TABLE_LOCAL_SUN_DISTANCE_M", 5.0))
                if sun_distance <= 0.0:
                    raise ValueError("FLIP_TABLE_LOCAL_SUN_DISTANCE_M must be positive")
                elevation_rad = math.radians(elevation)
                azimuth_rad = math.radians(azimuth)
                horizontal_distance = sun_distance * math.cos(elevation_rad)
                sun_position = Gf.Vec3d(
                    center_x + horizontal_distance * math.cos(azimuth_rad),
                    center_y + horizontal_distance * math.sin(azimuth_rad),
                    1.0 + sun_distance * math.sin(elevation_rad),
                )
                sun_xformable = UsdGeom.Xformable(sun_prim)
                translation = None
                for op in sun_xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translation = op
                        break
                if translation is None:
                    translation = sun_xformable.AddTranslateOp()
                translation.Set(sun_position)
                sun_visible = bool(
                    float(torch.rand(1, device=env.device)[0])
                    < _env_float("FLIP_TABLE_SUN_VISIBLE_PROBABILITY", 0.78)
                )
                (UsdGeom.Imageable(sun_prim).MakeVisible() if sun_visible else UsdGeom.Imageable(sun_prim).MakeInvisible())

                sphere_count_range = _env_range("FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE", (2.0, 4.0))
                sphere_count = int(
                    torch.randint(
                        max(1, int(sphere_count_range[0])),
                        max(1, int(sphere_count_range[1])) + 1,
                        (1,),
                        device=env.device,
                    )[0]
                )
                generated_spheres = []
                sphere_pool_size = max(sphere_count, max(1, int(sphere_count_range[1])))
                for light_index in range(sphere_pool_size):
                    sphere = UsdLux.SphereLight.Define(stage, f"{root_path}/CeilingLight_{light_index:02d}")
                    sphere_prim = sphere.GetPrim()
                    link_to_environment(sphere_prim, env_id)
                    sphere_visible = light_index < sphere_count
                    if not sphere_visible:
                        UsdGeom.Imageable(sphere_prim).MakeInvisible()
                        continue
                    UsdGeom.Imageable(sphere_prim).MakeVisible()
                    x = center_x + float(_uniform((1,), -3.2, 3.2, device=env.device)[0])
                    y = center_y + float(_uniform((1,), -3.2, 3.2, device=env.device)[0])
                    z = float(_uniform((1,), 3.1, 4.6, device=env.device)[0])
                    sphere_xformable = UsdGeom.Xformable(sphere_prim)
                    translation = None
                    for op in sphere_xformable.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                            translation = op
                            break
                    if translation is None:
                        translation = sphere_xformable.AddTranslateOp()
                    translation.Set(Gf.Vec3d(x, y, z))
                    sphere_intensity = float(
                        _uniform((1,), intensity_low, intensity_high * 1.35, device=env.device)[0]
                    )
                    sphere_temperature = float(
                        _uniform((1,), indoor_temperature[0], indoor_temperature[1], device=env.device)[0]
                    )
                    sphere_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(sphere_intensity)
                    sphere_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(
                        float(_uniform((1,), 0.18, 0.55, device=env.device)[0])
                    )
                    sphere_prim.CreateAttribute("inputs:normalize", Sdf.ValueTypeNames.Bool).Set(True)
                    sphere_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
                        Gf.Vec3f(1.0, 1.0, 1.0)
                    )
                    sphere_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
                    sphere_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(
                        sphere_temperature
                    )
                    sphere_exposure = float(
                        _uniform(
                            (1,),
                            light_exposure[0],
                            light_exposure[1],
                            device=env.device,
                        )[0]
                    )
                    sphere_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
                        sphere_exposure
                    )
                    generated_spheres.append(
                        {
                            "xyz": (round(x, 2), round(y, 2), round(z, 2)),
                            "intensity": round(sphere_intensity),
                            "temperature_k": round(sphere_temperature),
                            "exposure_ev": sphere_exposure,
                        }
                    )
                sample = {
                        "env": env_id,
                        "sun_visible": sun_visible,
                        "sun_intensity": round(sun_intensity_value),
                        "sun_temperature_k": round(sun_temperature_value),
                        "sun_elevation_deg": round(elevation, 1),
                        "sun_azimuth_deg": round(azimuth, 1),
                        "sun_exposure_ev": sun_exposure_value,
                        "ceiling_lights": generated_spheres,
                    }
                samples.append(sample)
                self._teleop_randomization(env_id)["lighting"] = sample
            if _verbose_reset_logs():
                print(
                    f"[FlipTableEvalTask] light randomization: {samples[:8]}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            _handle_randomization_failure("light", exc)

    def _disable_workbench_collision_for_diagnosis(self, env) -> None:
        """Optionally remove only the workbench collision for physics diagnosis."""
        if not _env_bool("FLIP_TABLE_DISABLE_WORKBENCH_COLLISION", False):
            return
        try:
            from pxr import UsdPhysics

            stage = env.sim.stage
            disabled = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if "/Scene/Table278" not in path:
                    continue
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
                disabled += 1
            print(
                f"[FlipTableEvalTask] diagnostic workbench collision disabled: {disabled}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FlipTableEvalTask] workbench collision diagnostic failed: {exc}", flush=True)

    def _ensure_robot_visual_materials(self, env) -> None:
        if self._robot_visual_materials_applied:
            return
        if not _env_bool("FLIP_TABLE_APPLY_G1_VISUAL_MATERIALS", False):
            return

        try:
            from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

            stage = env.sim.stage
            materials = {
                key: self._define_preview_material(stage, key, color)
                for key, color in G1_MATERIAL_COLORS.items()
            }

            applied = 0
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                material_key = _g1_material_key_for_path(path)
                if material_key is None:
                    continue
                material = materials[material_key]
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                gprim = UsdGeom.Gprim(prim)
                if gprim:
                    color = G1_MATERIAL_COLORS[material_key]
                    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
                applied += 1

            collision_prims = [
                prim
                for prim in stage.Traverse()
                if "/Robot/" in str(prim.GetPath())
                and any(
                    token in str(prim.GetPath()).lower()
                    for token in ("/collisions/", "/colliders/", "/collision")
                )
                and prim.IsActive()
            ]
            collision_api_prims = []
            for instance_prim in collision_prims:
                prototype = instance_prim.GetPrototype() if instance_prim.IsInstance() else instance_prim
                if not prototype or not prototype.IsValid():
                    continue
                for prototype_prim in Usd.PrimRange(prototype):
                    if prototype_prim.HasAPI(UsdPhysics.CollisionAPI):
                        collision_api_prims.append(
                            (str(instance_prim.GetPath()), str(prototype_prim.GetPath()), prototype_prim.GetTypeName())
                        )
            collision_prim_type_counts: dict[str, int] = {}
            for prim in collision_prims:
                collision_prim_type_counts[prim.GetTypeName()] = collision_prim_type_counts.get(prim.GetTypeName(), 0) + 1
            collision_type_counts: dict[str, int] = {}
            for _instance_path, _prototype_path, prim_type in collision_api_prims:
                collision_type_counts[prim_type] = collision_type_counts.get(prim_type, 0) + 1
            robot_collision_paths = collision_api_prims[:8]
            hand_collision_paths = [
                item for item in collision_api_prims if "dex1" in item[0].lower() or "dex1" in item[1].lower()
            ]
            collision_path_examples = [
                (str(prim.GetPath()), prim.GetTypeName(), list(prim.GetAppliedSchemas()))
                for prim in collision_prims[:16]
            ]

            self._robot_visual_materials_applied = True
            print(
                f"[FlipTableEvalTask] applied G1 visual materials to {applied} robot prims; "
                f"active robot collision prims={len(collision_prims)}, "
                f"collision_prim_types={collision_prim_type_counts}, "
                f"UsdPhysics.CollisionAPI={len(collision_api_prims)}, "
                f"collision_types={collision_type_counts}, "
                f"dex1_collision_paths={hand_collision_paths}, "
                f"collision_examples={robot_collision_paths}, "
                f"collision_paths={collision_path_examples}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            _handle_randomization_failure("G1 visual material binding", exc)

    def _define_preview_material(self, stage, name: str, color: tuple[float, float, float]):
        return self._define_preview_material_at(stage, f"/World/Looks/flip_table_g1_{name}", color)

    def _define_preview_material_at(
        self,
        stage,
        material_path: str,
        color: tuple[float, float, float],
        *,
        roughness: float = 0.55,
        metallic: float = 0.0,
    ):
        from pxr import Gf, Sdf, UsdShade

        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _gripper_far_from_all_legs(self, env) -> torch.Tensor:
        ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for object_name, _, _ in self.leg_reg_int_sites:
            if self._find_scene_entity(env, object_name) is None:
                continue
            ok &= OU.gripper_obj_far(env, object_name, th=self.gripper_leg_distance_threshold)
        return ok

    def _debug_robot_body_positions(self, env) -> str:
        robot = self._find_scene_entity(env, "robot")
        if robot is None or not hasattr(robot, "data"):
            return "robot_entity=missing"
        data = robot.data
        positions = getattr(data, "body_pos_w", None)
        names = getattr(data, "body_names", None)
        if positions is None or names is None:
            return "robot_body_positions=missing"
        positions = as_torch(positions).detach()
        name_to_index = {str(name): index for index, name in enumerate(names)}
        selected = {}
        for key in (
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_hand_palm_link",
            "right_hand_palm_link",
            "left_dex1_finger_link_1",
            "left_dex1_finger_link_2",
            "right_dex1_finger_link_1",
            "right_dex1_finger_link_2",
        ):
            index = name_to_index.get(key)
            if index is not None:
                selected[key] = [round(float(value), 4) for value in positions[0, index, :3].cpu().tolist()]
        if not selected:
            return f"robot_body_names={list(name_to_index)[:12]}"
        root_pos = getattr(data, "root_pos_w", None)
        root_quat = getattr(data, "root_quat_w", None)
        root_desc = ""
        if root_pos is not None and root_quat is not None:
            root_desc = (
                f", root_pos={[round(float(v), 4) for v in root_pos[0, :3].detach().cpu().tolist()]}, "
                f"root_quat={[round(float(v), 4) for v in root_quat[0, :4].detach().cpu().tolist()]}"
            )
        joint_pos = getattr(data, "joint_pos", None)
        joint_desc = ""
        if joint_pos is not None:
            joint_names = list(getattr(data, "joint_names", ()))
            upper_named_indices = {
                name: index
                for index, name in enumerate(joint_names)
                if name in FLIP_TABLE_UPPER_BODY_ACTION_JOINT_NAMES
            }
            upper_values = {
                name: round(float(joint_pos[0, index].detach().cpu().item()), 5)
                for name, index in upper_named_indices.items()
            }
            joint_desc = f", upper_body_joint_pos={upper_values}"
            named_indices = {
                name: index
                for index, name in enumerate(joint_names)
                if name in {
                    "left_dex1_finger_joint_1",
                    "left_dex1_finger_joint_2",
                    "right_dex1_finger_joint_1",
                    "right_dex1_finger_joint_2",
                }
            }
            dex_values = {
                name: round(float(joint_pos[0, index].detach().cpu().item()), 5)
                for name, index in named_indices.items()
            }
            joint_desc += f", dex1_joint_pos={dex_values}"

        contact_desc = ""
        for contact_attr in ("net_contact_forces_w", "contact_forces_w"):
            contact_forces = getattr(data, contact_attr, None)
            if contact_forces is None:
                continue
            contact_forces = contact_forces.detach() if torch.is_tensor(contact_forces) else torch.as_tensor(contact_forces)
            if contact_forces.ndim < 3:
                continue
            finger_contacts = {}
            for name in (
                "left_dex1_finger_link_1",
                "left_dex1_finger_link_2",
                "right_dex1_finger_link_1",
                "right_dex1_finger_link_2",
            ):
                index = name_to_index.get(name)
                if index is not None and index < contact_forces.shape[1]:
                    finger_contacts[name] = round(
                        float(torch.linalg.norm(contact_forces[0, index, :3].float()).cpu().item()),
                        5,
                    )
            contact_desc = f", {contact_attr}={finger_contacts}"
            break

        if not contact_desc and not self._runtime_contact_debug_printed:
            data_attrs = sorted(
                name
                for name in dir(data)
                if any(token in name.lower() for token in ("contact", "force", "collision"))
            )
            scene = getattr(env, "scene", None)
            scene_names = []
            if scene is not None:
                try:
                    scene_names = [str(name) for name in scene.keys()]
                except Exception:
                    scene_names = []
            physx_view = getattr(robot, "root_physx_view", None)
            view_attrs = []
            if physx_view is not None:
                view_attrs = sorted(
                    name
                    for name in dir(physx_view)
                    if any(
                        token in name.lower()
                        for token in ("contact", "force", "collision", "transform", "pose", "link", "body")
                    )
                )
            print(
                "[FlipTableEvalTask] runtime contact API: "
                f"data_type={type(data).__name__}, data_attrs={data_attrs}, "
                f"scene_names={scene_names}, "
                f"physx_view_type={type(physx_view).__name__ if physx_view is not None else None}, "
                f"physx_view_attrs={view_attrs}, "
                f"link_paths={list(getattr(physx_view, 'link_paths', ()))[:8] if physx_view is not None else []}",
                flush=True,
            )
            if physx_view is not None and hasattr(physx_view, "get_link_transforms"):
                try:
                    link_transforms = physx_view.get_link_transforms()
                    print(
                        "[FlipTableEvalTask] runtime link transforms: "
                        f"type={type(link_transforms).__name__}, "
                        f"shape={getattr(link_transforms, 'shape', None)}, "
                        f"finger_samples="
                        f"{[(path.rsplit('/', 1)[-1], str(link_transforms[0, index])) for index, path in enumerate(getattr(physx_view, 'link_paths', [[]])[0]) if 'dex1_finger_link' in path]}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[FlipTableEvalTask] runtime link transforms unavailable: {exc}", flush=True)
            self._runtime_contact_debug_printed = True

        leg_positions = []
        for object_name, object_prim_path, _site_prim_path in self.leg_reg_int_sites:
            leg_pos, _leg_quat = self._extract_object_pose(env, object_name, object_prim_path)
            if leg_pos is not None:
                leg_positions.append(leg_pos[0, :3])
        finger_positions = [
            positions[0, name_to_index[name], :3]
            for name in (
                "left_dex1_finger_link_1",
                "left_dex1_finger_link_2",
                "right_dex1_finger_link_1",
                "right_dex1_finger_link_2",
            )
            if name in name_to_index
        ]
        finger_collision_positions = []
        quaternions = getattr(data, "body_quat_w", None)
        if quaternions is not None:
            quaternions = as_torch(quaternions)
            finger_collision_offsets = {
                "left_dex1_finger_link_1": (0.1102, -0.0309, 0.0),
                "left_dex1_finger_link_2": (0.1112, 0.0309, 0.0),
                "right_dex1_finger_link_1": (0.1102, -0.0309, 0.0),
                "right_dex1_finger_link_2": (0.1112, 0.0309, 0.0),
            }
            for name, offset in finger_collision_offsets.items():
                index = name_to_index.get(name)
                if index is None:
                    continue
                # Isaac Lab changed the raw asset-buffer layout across major
                # versions.  Reuse RoboFinals' compatibility helper instead of
                # assuming scalar-first or scalar-last in this diagnostic.
                quat = sim_quat_raw_to_xyzw_torch(quaternions[0, index, :4].float())
                vector = torch.tensor(offset, device=positions.device, dtype=positions.dtype)
                xyz = quat[:3]
                twice_cross = 2.0 * torch.linalg.cross(xyz, vector)
                rotated = vector + quat[3] * twice_cross + torch.linalg.cross(xyz, twice_cross)
                finger_collision_positions.append(positions[0, index, :3] + rotated)
        distance_desc = ""
        if leg_positions and finger_positions:
            leg_tensor = torch.stack(leg_positions)
            finger_tensor = torch.stack(finger_positions)
            min_distances = torch.cdist(finger_tensor, leg_tensor).amin(dim=1)
            distance_desc = (
                ", finger_link_origin_leg_min_dist="
                f"{[round(float(value), 4) for value in min_distances.detach().cpu().tolist()]}"
            )
            if finger_collision_positions:
                collision_tensor = torch.stack(finger_collision_positions)
                collision_distances = torch.cdist(collision_tensor, leg_tensor).amin(dim=1)
                distance_desc += (
                    ", finger_collision_center_leg_min_dist="
                    f"{[round(float(value), 4) for value in collision_distances.detach().cpu().tolist()]}"
                )
                if _env_bool("FLIP_TABLE_GEOMETRY_DEBUG", False):
                    distance_desc += (
                        ", leg_positions="
                        f"{[[round(float(v), 4) for v in row.detach().cpu().tolist()] for row in leg_tensor]}"
                        ", finger_collision_positions="
                        f"{[[round(float(v), 4) for v in row.detach().cpu().tolist()] for row in collision_tensor]}"
                    )
        return f"robot_body_pos={selected}{root_desc}{joint_desc}{distance_desc}{contact_desc}"

    def _debug_table_part_positions(self, env) -> str:
        parts = {}
        for object_name, object_prim_path, _ in (self.table_reg_int_sites[:1] + self.leg_reg_int_sites):
            position, _quat = self._extract_object_pose(env, object_name, object_prim_path)
            if position is not None:
                parts[object_name] = [round(float(value), 4) for value in position[0, :3].detach().cpu().tolist()]
        if not self._runtime_table_dynamics_debug_printed:
            try:
                from pxr import UsdPhysics

                stage = env.sim.stage
                collision_groups = []
                for group_prim in stage.Traverse():
                    if group_prim.GetTypeName() != "PhysicsCollisionGroup":
                        continue
                    includes = group_prim.GetRelationship("collection:colliders:includes")
                    filtered = group_prim.GetRelationship("physics:filteredGroups")
                    collision_groups.append(
                        (
                            str(group_prim.GetPath()),
                            [str(path) for path in includes.GetTargets()] if includes else [],
                            [str(path) for path in filtered.GetTargets()] if filtered else [],
                        )
                    )
                if collision_groups:
                    print(
                        f"[FlipTableEvalTask] runtime collision groups: {collision_groups}",
                        flush=True,
                    )
                prim_specs = [
                    ("table", "Table001/Table001_01"),
                    ("leg0", "Leg001/Leg001"),
                    ("leg1", "Leg001_01/Leg001"),
                    ("leg2", "Leg001_03/Leg001"),
                    ("leg3", "Leg001_06/Leg001"),
                    ("workbench", "Table278/Table278"),
                ]
                dynamics = []
                for label, suffix in prim_specs:
                    prim = self._find_prim_by_suffix(env, suffix, env_id=0)
                    if prim is None:
                        dynamics.append((label, "missing"))
                        continue
                    rigid = UsdPhysics.RigidBodyAPI(prim)
                    mass = UsdPhysics.MassAPI(prim)
                    collision_count = 0
                    collision_enabled = 0
                    collision_disabled = 0
                    collision_unset = 0
                    for child in stage.Traverse():
                        path = str(child.GetPath())
                        if str(prim.GetPath()) in path and child.HasAPI(UsdPhysics.CollisionAPI):
                            collision_count += 1
                            enabled_attr = child.GetAttribute("physics:collisionEnabled")
                            if enabled_attr and enabled_attr.HasAuthoredValue():
                                if bool(enabled_attr.Get()):
                                    collision_enabled += 1
                                else:
                                    collision_disabled += 1
                            else:
                                collision_unset += 1
                    dynamics.append(
                        (
                            label,
                            str(prim.GetPath()),
                            f"rigid={bool(rigid)}, "
                            f"kinematic={rigid.GetKinematicEnabledAttr().Get() if rigid else None}, "
                            f"mass={mass.GetMassAttr().Get() if mass else None}, "
                            f"collision_shapes={collision_count}, "
                            f"enabled={collision_enabled}, disabled={collision_disabled}, unset={collision_unset}",
                        )
                    )
                robot_collision_counts = []
                for label, suffix in (
                    ("left_finger_1", "Robot/left_dex1_finger_link_1"),
                    ("left_finger_2", "Robot/left_dex1_finger_link_2"),
                    ("right_finger_1", "Robot/right_dex1_finger_link_1"),
                    ("right_finger_2", "Robot/right_dex1_finger_link_2"),
                ):
                    prim = self._find_prim_by_suffix(env, suffix, env_id=0)
                    counts = {"total": 0, "enabled": 0, "disabled": 0, "unset": 0}
                    if prim is not None:
                        for child in stage.Traverse():
                            if str(prim.GetPath()) not in str(child.GetPath()):
                                continue
                            if not child.HasAPI(UsdPhysics.CollisionAPI):
                                continue
                            counts["total"] += 1
                            enabled_attr = child.GetAttribute("physics:collisionEnabled")
                            if enabled_attr and enabled_attr.HasAuthoredValue():
                                counts["enabled" if bool(enabled_attr.Get()) else "disabled"] += 1
                            else:
                                counts["unset"] += 1
                    robot_collision_counts.append((label, counts))
                self._runtime_table_dynamics_debug_printed = True
                runtime_states = []
                for label, entity_name in (
                    ("table", "Table001_Table001_01"),
                    ("leg0", "Leg001_Leg001"),
                    ("workbench", "Table278"),
                ):
                    entity = self._find_scene_entity(env, entity_name)
                    data = getattr(entity, "data", None)
                    if data is None:
                        runtime_states.append((label, "entity_missing"))
                        continue
                    state = {}
                    for attr in ("root_lin_vel_w", "root_ang_vel_w", "root_pos_w", "root_quat_w"):
                        value = getattr(data, attr, None)
                        if value is None:
                            continue
                        tensor = as_torch(value).detach()[0].reshape(-1).cpu().tolist()
                        state[attr] = [round(float(item), 5) for item in tensor[:7]]
                    runtime_states.append((label, state))
                print(
                    f"[FlipTableEvalTask] runtime table dynamics: {dynamics}; "
                    f"robot_finger_collision_flags={robot_collision_counts}; "
                    f"runtime_states={runtime_states}; "
                    f"scene_entities={self._scene_entity_names_for_log(env)}; "
                    f"sim_gravity={getattr(getattr(env.sim, 'cfg', None), 'gravity', None)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                self._runtime_table_dynamics_debug_printed = True
                print(f"[FlipTableEvalTask] runtime table dynamics unavailable: {exc}", flush=True)
        if not self._runtime_table_joint_debug_printed:
            try:
                from pxr import UsdPhysics

                stage = env.sim.stage
                joint_status = []
                table_body_suffix = self.table_reg_int_sites[0][1]
                table_body = self._find_prim_by_suffix(env, table_body_suffix, env_id=0)
                if table_body is not None:
                    for leg_index, (_leg_name, leg_body_suffix, _site) in enumerate(self.leg_reg_int_sites):
                        joint_path = table_body.GetPath().AppendChild(f"FlipTableEvalFixedJoint_{leg_index}")
                        joint_prim = stage.GetPrimAtPath(joint_path)
                        if not joint_prim.IsValid():
                            joint_status.append((str(joint_path), "missing"))
                            continue
                        joint = UsdPhysics.FixedJoint(joint_prim)
                        body0 = [str(target) for target in joint.GetBody0Rel().GetTargets()]
                        body1 = [str(target) for target in joint.GetBody1Rel().GetTargets()]
                        enabled_attr = joint_prim.GetAttribute("physics:jointEnabled")
                        enabled = enabled_attr.Get() if enabled_attr and enabled_attr.HasAuthoredValue() else True
                        joint_status.append(
                            (
                                str(joint_path),
                                f"enabled={enabled}, body0={body0}, body1={body1}",
                            )
                        )
                self._runtime_table_joint_debug_printed = True
                print(f"[FlipTableEvalTask] runtime table joints: {joint_status}", flush=True)
            except Exception as exc:  # noqa: BLE001
                self._runtime_table_joint_debug_printed = True
                print(f"[FlipTableEvalTask] runtime table joints unavailable: {exc}", flush=True)
        return f"table_parts={parts}"

    def _debug_gripper_contacts(self, env) -> str:
        try:
            sensors = getattr(env.scene, "sensors", {})
            parts = []
            for sensor_name in (
                "left_gripper_contact",
                "left_gripper_contact_2",
                "right_gripper_contact",
                "right_gripper_contact_2",
            ):
                sensor = sensors.get(sensor_name) if hasattr(sensors, "get") else None
                if sensor is None:
                    try:
                        sensor = env.scene[sensor_name]
                    except Exception:
                        sensor = None
                if sensor is None:
                    continue
                values = None
                for attr in ("net_forces_w", "force_matrix_w"):
                    candidate = getattr(sensor.data, attr, None)
                    if candidate is not None:
                        values = candidate
                        break
                if values is None:
                    parts.append(f"{sensor_name}=unavailable")
                    continue
                if torch.is_tensor(values):
                    magnitude = torch.linalg.norm(values.detach().float(), dim=-1).amax().item()
                else:
                    magnitude = float(torch.as_tensor(values).float().norm().item())
                parts.append(f"{sensor_name}_max={magnitude:.4f}")
                if _env_bool("FLIP_TABLE_CONTACT_POINT_DEBUG", False):
                    positions = getattr(sensor.data, "contact_pos_w", None)
                    if positions is not None:
                        positions = positions.detach() if torch.is_tensor(positions) else torch.as_tensor(positions)
                        force_tensor = values.detach() if torch.is_tensor(values) else torch.as_tensor(values)
                        if positions.ndim >= 3 and force_tensor.ndim >= 3:
                            point_positions = positions[0].reshape(-1, 3).float()
                            point_forces = force_tensor[0].reshape(-1, force_tensor.shape[-1]).float()
                            point_magnitudes = torch.linalg.norm(point_forces[..., :3], dim=-1)
                            valid = point_magnitudes > 1e-3
                            if torch.any(valid):
                                ranked = torch.argsort(point_magnitudes, descending=True)
                                contact_points = []
                                for point_index in ranked.tolist():
                                    if not bool(valid[point_index]):
                                        continue
                                    contact_points.append(
                                        {
                                            "p": [round(float(v), 4) for v in point_positions[point_index].cpu().tolist()],
                                            "f": round(float(point_magnitudes[point_index].cpu().item()), 4),
                                        }
                                    )
                                    if len(contact_points) >= 4:
                                        break
                                parts.append(f"{sensor_name}_points={contact_points}")
            return "gripper_contacts=" + (";".join(parts) if parts else "unavailable")
        except Exception as exc:  # noqa: BLE001
            return f"contact_debug_error={type(exc).__name__}"

    def _gripper_contact_force_metrics(self, env) -> dict[str, object]:
        """Return recorder-only gripper force magnitudes in newtons."""

        sensors = getattr(env.scene, "sensors", {})
        per_sensor: dict[str, float] = {}
        per_side = {"left": 0.0, "right": 0.0}
        for sensor_name in (
            "left_gripper_contact",
            "left_gripper_contact_2",
            "right_gripper_contact",
            "right_gripper_contact_2",
        ):
            sensor = sensors.get(sensor_name) if hasattr(sensors, "get") else None
            if sensor is None:
                try:
                    sensor = env.scene[sensor_name]
                except Exception:  # noqa: BLE001
                    continue
            values = None
            for attribute in ("net_forces_w", "force_matrix_w"):
                candidate = getattr(sensor.data, attribute, None)
                if candidate is not None:
                    values = candidate
                    break
            if values is None:
                continue
            # Isaac Lab 3 contact data is a Warp-backed ProxyArray.  Use the
            # compatibility conversion explicitly so the live zero-copy torch
            # view is sampled instead of relying on generic array coercion.
            tensor = as_torch(values).detach()
            if tensor.numel() == 0 or tensor.shape[-1] < 3:
                continue
            magnitude = float(
                torch.linalg.norm(tensor[0, ..., :3].float(), dim=-1)
                .amax()
                .detach()
                .cpu()
                .item()
            )
            if not math.isfinite(magnitude) or magnitude < 0.0:
                raise RuntimeError(f"invalid contact force from {sensor_name}")
            per_sensor[sensor_name] = magnitude
            side = "left" if sensor_name.startswith("left") else "right"
            per_side[side] = max(per_side[side], magnitude)
        return {
            "available": bool(per_sensor),
            "left_max_n": per_side["left"],
            "right_max_n": per_side["right"],
            "sensor_max_n": per_sensor,
        }

    def _dex1_drive_force_metrics(self, env) -> dict[str, object]:
        """Return diagnostic Dex1 prismatic drive forces after effort clipping."""

        robot = self._robot(env)
        data = getattr(robot, "data", None)
        joint_names = [str(name) for name in getattr(data, "joint_names", ())]
        applied = getattr(data, "applied_torque", None)
        if applied is None or not joint_names:
            return {"available": False, "effort_limit_n": 20.0}
        values = as_torch(applied)[0].detach().float().cpu()
        per_joint: dict[str, float] = {}
        per_side = {"left": 0.0, "right": 0.0}
        for side in ("left", "right"):
            for finger_index in (1, 2):
                name = f"{side}_dex1_finger_joint_{finger_index}"
                if name not in joint_names:
                    continue
                force = abs(float(values[joint_names.index(name)].item()))
                if not math.isfinite(force):
                    raise RuntimeError(f"invalid applied Dex1 force from {name}")
                per_joint[name] = force
                per_side[side] = max(per_side[side], force)
        return {
            "available": len(per_joint) == 4,
            "effort_limit_n": 20.0,
            "left_max_n": per_side["left"],
            "right_max_n": per_side["right"],
            "joint_abs_n": per_joint,
        }

    def _debug_collision_bounds(self, env) -> str:
        """Report finite world bounds for selected body roots during calibration.

        Isaac Sim's USD scene can expose collision geometry through an unloaded
        payload or through Fabric, in which case traversing CollisionAPI
        descendants returns an empty range.  Reporting that sentinel would look
        like a real geometry measurement, so use the body root's composed bound
        and omit non-finite ranges.
        """
        if not _env_bool("FLIP_TABLE_GEOMETRY_DEBUG", False):
            return ""
        try:
            from pxr import Usd, UsdGeom

            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=False,
            )
            paths = (
                "Robot/left_dex1_finger_link_1",
                "Robot/left_dex1_finger_link_2",
                "Leg001_01/Leg001",
                "Leg001/Leg001",
                "Table001/Table001_01",
            )
            bounds = []
            for suffix in paths:
                prim = self._find_prim_by_suffix(env, suffix, env_id=0)
                if prim is None:
                    continue
                aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                minimum = aligned.GetMin()
                maximum = aligned.GetMax()
                values = [float(minimum[i]) for i in range(3)] + [float(maximum[i]) for i in range(3)]
                if not all(math.isfinite(value) for value in values):
                    continue
                if any(minimum[i] > maximum[i] for i in range(3)):
                    continue
                bounds.append(
                    f"{suffix}=("
                    f"{[round(float(minimum[i]), 4) for i in range(3)]},"
                    f"{[round(float(maximum[i]), 4) for i in range(3)]})"
                )
            return "collision_bounds=" + ";".join(bounds)
        except Exception as exc:  # noqa: BLE001
            return f"collision_bounds_error={type(exc).__name__}"

    def _debug_wbc_action_term(self, env) -> str:
        try:
            def values(value):
                if value is None:
                    return None
                if torch.is_tensor(value):
                    value = value[0].detach().cpu().reshape(-1).tolist()
                return [round(float(item), 4) for item in value]
            term_name = "base_action"
            try:
                term = env.action_manager.get_term(term_name)
                fields = (
                    f"wbc_raw={values(getattr(term, 'raw_actions', None))}, "
                    f"wbc_target={values(getattr(term, 'target_robot_joints_mujoco', None))}, "
                    f"wbc_processed={values(getattr(term, 'processed_actions', None))}"
                )
                return fields
            except KeyError:
                parts = []
                for term_name in ("waist_action", "left_arm_action", "right_arm_action"):
                    term = env.action_manager.get_term(term_name)
                    parts.append(f"{term_name}={values(getattr(term, 'processed_actions', None))}")
                for term_name in ("left_hand_action", "right_hand_action"):
                    try:
                        term = env.action_manager.get_term(term_name)
                    except KeyError:
                        continue
                    parts.append(f"{term_name}={values(getattr(term, 'processed_actions', None))}")
                return "joint_action_terms=" + ";".join(parts)
        except Exception as exc:  # noqa: BLE001
            return f"wbc_debug_error={type(exc).__name__}"

    def _debug_global_camera_pose(self, env) -> str:
        try:
            from pxr import UsdGeom

            prim = self._find_prim_by_suffix(env, "global_camera", env_id=0)
            if prim is None:
                return "global_camera_pose=missing"
            transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            pos = transform.ExtractTranslation()
            quat = transform.ExtractRotationQuat()
            imag = quat.GetImaginary()
            return (
                "global_camera_pose=("
                f"{float(pos[0]):.6f},{float(pos[1]):.6f},{float(pos[2]):.6f};"
                f"{float(imag[0]):.8f},{float(imag[1]):.8f},"
                f"{float(imag[2]):.8f},{float(quat.GetReal()):.8f})"
            )
        except Exception as exc:  # noqa: BLE001
            return f"global_camera_pose_error={type(exc).__name__}"

    def get_flip_table_teleop_diagnostics(self, env) -> dict[str, object]:
        """Return one-environment recorder diagnostics, never policy observations."""

        if env.num_envs != 1:
            raise ValueError("AVP teleoperation requires exactly one simulator environment")
        robot = self._robot(env)
        data = getattr(robot, "data", None)
        if data is None or not hasattr(data, "joint_pos") or not hasattr(data, "joint_vel"):
            raise RuntimeError("robot joint state is unavailable for teleoperation recording")
        joint_names = [str(name) for name in getattr(data, "joint_names", ())]
        joint_position = as_torch(data.joint_pos)[0].detach().float().cpu()
        joint_velocity = as_torch(data.joint_vel)[0].detach().float().cpu()
        if len(joint_names) != joint_position.numel() or joint_velocity.numel() != len(joint_names):
            raise RuntimeError("simulator joint names and state vectors differ")

        root_position_local, root_quaternion = self._robot_root_pose_local(
            env, torch.zeros(1, dtype=torch.long, device=env.device)
        )
        root_position_world = root_position_local[0] + env.scene.env_origins[0]
        torso_position, torso_quaternion = self._extract_stage_prim_pose(env, "Robot/torso_link")
        table_position, table_quaternion = self._table_body_pose(env)
        if table_position is None or table_quaternion is None:
            raise RuntimeError("white-table pose is unavailable for recorder diagnostics")
        workbench_position, workbench_quaternion = self._workbench_pose(env)
        policy_camera_poses: dict[str, object] = {}
        for role, suffix in (
            ("head_left", "Robot/torso_link/first_person_camera"),
            ("head_right", "Robot/torso_link/head_right_camera"),
        ):
            camera_position, camera_quaternion = self._extract_stage_prim_pose(env, suffix)
            policy_camera_poses[role] = (
                None
                if camera_position is None or camera_quaternion is None
                else {
                    "position_world_m": [
                        float(item) for item in camera_position[0, :3].detach().cpu().tolist()
                    ],
                    "quaternion_xyzw": [
                        float(item) for item in camera_quaternion[0, :4].detach().cpu().tolist()
                    ],
                }
            )

        table_entity = self._find_scene_entity(env, "Table001_Table001_01")
        table_data = getattr(table_entity, "data", None)

        def velocity(name: str) -> list[float]:
            value = getattr(table_data, name, None)
            if value is None:
                return [0.0, 0.0, 0.0]
            tensor = as_torch(value)
            if tensor.ndim == 3:
                tensor = tensor[:, 0]
            return [float(item) for item in tensor[0, :3].detach().cpu().tolist()]

        components = self._stable_flip_success_components(env)
        component_values: dict[str, object] = {}
        for name, value in components.items():
            scalar = value[0].detach().cpu().item()
            component_values[name] = bool(scalar) if value.dtype == torch.bool else float(scalar)
        hold_steps = _env_int("FLIP_TABLE_SUCCESS_HOLD_STEPS", 20, minimum=1)
        success = bool(component_values["candidate"]) and int(self._stable_success_streak[0]) >= hold_steps

        leg_poses = []
        for leg_name, leg_prim_path, _site_path in self.leg_reg_int_sites:
            position, quaternion = self._extract_object_pose(env, leg_name, leg_prim_path)
            if position is None or quaternion is None:
                continue
            leg_poses.append(
                {
                    "name": leg_name,
                    "position_world_m": [float(item) for item in position[0, :3].detach().cpu().tolist()],
                    "quaternion_xyzw": [float(item) for item in quaternion[0, :4].detach().cpu().tolist()],
                }
            )

        def actuator_summary(name: str) -> dict[str, object] | None:
            """Expose effective drive values only in offline recorder telemetry."""

            actuator = getattr(robot, "actuators", {}).get(name)
            if actuator is None:
                return None
            result: dict[str, object] = {}
            joint_ids = getattr(actuator, "joint_ids", ())
            ids = [int(value) for value in joint_ids]
            if not ids:
                ids = [int(value) for value in getattr(actuator, "_joint_ids", ())]
            result["joint_ids"] = ids
            resolved_names = [joint_names[value] for value in ids if 0 <= value < len(joint_names)]
            if not resolved_names:
                actuator_names = getattr(actuator, "joint_names", ()) or getattr(
                    actuator, "_joint_names", ()
                )
                resolved_names = [str(value) for value in actuator_names]
            result["joint_names"] = resolved_names
            if not resolved_names:
                config = getattr(actuator, "cfg", None)
                expressions = getattr(config, "joint_names_expr", ())
                result["joint_name_expressions"] = [str(value) for value in expressions]
            for field in ("stiffness", "damping", "armature", "friction", "effort_limit", "velocity_limit"):
                value = getattr(actuator, field, None)
                if value is None:
                    continue
                tensor = as_torch(value).detach().float()
                if tensor.ndim > 1:
                    tensor = tensor[0]
                if tensor.numel() == 0:
                    continue
                values = tensor.reshape(-1)
                result[field] = {
                    "min": float(values.min().cpu().item()),
                    "median": float(values.median().cpu().item()),
                    "max": float(values.max().cpu().item()),
                }
            return result

        return {
            "joint_names": joint_names,
            "joint_position_rad": [float(item) for item in joint_position.tolist()],
            "joint_velocity_rad_s": [float(item) for item in joint_velocity.tolist()],
            "root_pose_world_xyzw": [
                *[float(item) for item in root_position_world.detach().cpu().tolist()],
                *[float(item) for item in root_quaternion[0].detach().cpu().tolist()],
            ],
            "torso_link": None
            if torso_position is None or torso_quaternion is None
            else {
                "position_world_m": [
                    float(item) for item in torso_position[0, :3].detach().cpu().tolist()
                ],
                "quaternion_xyzw": [
                    float(item) for item in torso_quaternion[0, :4].detach().cpu().tolist()
                ],
            },
            "white_table": {
                "position_world_m": [
                    float(item) for item in table_position[0, :3].detach().cpu().tolist()
                ],
                "quaternion_xyzw": [
                    float(item) for item in table_quaternion[0, :4].detach().cpu().tolist()
                ],
                "linear_velocity_m_s": velocity("root_lin_vel_w"),
                "angular_velocity_rad_s": velocity("root_ang_vel_w"),
                "legs": leg_poses,
            },
            # The workbench is stationary and belongs only to offline reset
            # calibration. Keeping its pose in the diagnostic trace lets a
            # source-CAD table pose become a workbench-local reset candidate
            # without exposing either pose to a policy observation.
            "workbench": None
            if workbench_position is None or workbench_quaternion is None
            else {
                "position_world_m": [
                    float(item) for item in workbench_position[0, :3].detach().cpu().tolist()
                ],
                "quaternion_xyzw": [
                    float(item) for item in workbench_quaternion[0, :4].detach().cpu().tolist()
                ],
            },
            # Camera poses are offline calibration telemetry.  They are not
            # part of observations and must never reach a policy or planner.
            "policy_camera_poses": policy_camera_poses,
            "success": success,
            "success_components": component_values,
            "gripper_contact_force_n": self._gripper_contact_force_metrics(env),
            "dex1_drive_force_n": self._dex1_drive_force_metrics(env),
            # This is simulator-only evidence for actuator identification. It
            # is never an observation, action, policy feature, or reward.
            "actuator_drive_diagnostics": {
                "arms": actuator_summary("arms"),
                "grippers": actuator_summary("grippers"),
            },
            "randomization": json.loads(
                json.dumps(self._teleop_randomization_samples.get(0, {}), allow_nan=False)
            ),
            "privileged_policy_features": [],
        }

    def get_flip_table_teleop_force_diagnostics(self, env) -> dict[str, object]:
        """Return lightweight, diagnostic-only force samples at servo rate.

        The full recorder diagnostics synchronize object poses and metadata to
        the CPU, which is appropriate at camera rate but too expensive at every
        50 Hz control step.  This endpoint is deliberately limited to the four
        Dex1 drives and finger contact sensors so short contact impulses are not
        lost between camera frames.
        """

        if env.num_envs != 1:
            raise ValueError(
                "AVP teleoperation requires exactly one simulator environment"
            )
        return {
            "gripper_contact_force_n": self._gripper_contact_force_metrics(env),
            "dex1_drive_force_n": self._dex1_drive_force_metrics(env),
        }

    def _stable_flip_success_components(self, env) -> dict[str, torch.Tensor]:
        table_pos, table_quat = self._table_body_pose(env)
        if (
            table_pos is None
            or table_quat is None
            or self._initial_table_normal is None
            or self._initial_table_pos is None
        ):
            false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            inf = torch.full((env.num_envs,), torch.inf, device=env.device)
            return {
                "candidate": false,
                "normal_dot": inf,
                "tabletop_lift_m": -inf,
                "linear_speed_m_s": inf,
                "angular_speed_rad_s": inf,
                "flipped": false,
                "lifted": false,
                "stable": false,
                "within_workbench": false,
                "gripper_clear": false,
            }

        dot_threshold = _env_float("FLIP_TABLE_SUCCESS_DOT_THRESHOLD", -0.95)
        min_lift_m = _env_float("FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M", 0.35)
        max_linear_speed = _env_float("FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S", 0.15)
        max_angular_speed = _env_float("FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S", 0.50)
        if not -1.0 <= dot_threshold <= 1.0:
            raise ValueError("FLIP_TABLE_SUCCESS_DOT_THRESHOLD must be in [-1, 1]")
        if min_lift_m < 0.0 or max_linear_speed < 0.0 or max_angular_speed < 0.0:
            raise ValueError("flip-table success lift and speed thresholds must be non-negative")

        table_rot = matrix_from_quat(table_quat)
        current_normal = table_rot[:, :, 2]
        initial_normal = self._initial_table_normal.to(
            device=env.device, dtype=current_normal.dtype
        )
        normal_dot = torch.sum(current_normal * initial_normal, dim=-1)
        tabletop_lift = table_pos[:, 2] - self._initial_table_pos.to(table_pos)[:, 2]
        flipped = normal_dot <= dot_threshold
        lifted = tabletop_lift >= min_lift_m

        table_entity = self._find_scene_entity(env, "Table001_Table001_01")
        data = getattr(table_entity, "data", None)
        root_linear = getattr(data, "root_lin_vel_w", None)
        root_angular = getattr(data, "root_ang_vel_w", None)
        if root_linear is None or root_angular is None:
            linear_speed = torch.full(
                (env.num_envs,), torch.inf, dtype=table_pos.dtype, device=env.device
            )
            angular_speed = linear_speed.clone()
        else:
            linear_speed = torch.linalg.norm(as_torch(root_linear)[:, :3], dim=-1)
            angular_speed = torch.linalg.norm(as_torch(root_angular)[:, :3], dim=-1)
        stable = (linear_speed <= max_linear_speed) & (angular_speed <= max_angular_speed)

        workbench_pos, workbench_quat = self._workbench_pose(env)
        if workbench_pos is None or workbench_quat is None:
            within_workbench = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        else:
            workbench_rot = matrix_from_quat(workbench_quat)
            table_relative = torch.bmm(
                workbench_rot.transpose(1, 2),
                (table_pos[:, :3] - workbench_pos[:, :3]).unsqueeze(-1),
            ).squeeze(-1)
            half_length = _env_float("FLIP_TABLE_WORKBENCH_HALF_LENGTH_M", 0.90)
            half_depth = _env_float("FLIP_TABLE_WORKBENCH_HALF_DEPTH_M", 0.375)
            edge_margin = _env_float("FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M", 0.03)
            usable_half_length = half_length - edge_margin
            usable_half_depth = half_depth - edge_margin
            within_workbench = (
                (torch.abs(table_relative[:, 0]) <= usable_half_length)
                & (torch.abs(table_relative[:, 1]) <= usable_half_depth)
            )

        gripper_clear = self._gripper_far_from_all_legs(env)
        candidate = flipped & lifted & stable & within_workbench & gripper_clear
        return {
            "candidate": candidate,
            "normal_dot": normal_dot,
            "tabletop_lift_m": tabletop_lift,
            "linear_speed_m_s": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "flipped": flipped,
            "lifted": lifted,
            "stable": stable,
            "within_workbench": within_workbench,
            "gripper_clear": gripper_clear,
        }

    def _check_success(self, env):
        self._apply_lower_body_lock(env)
        check_interval = _env_int(
            "FLIP_TABLE_SUCCESS_CHECK_INTERVAL_STEPS", 1, minimum=1
        )
        should_check = self._success_debug_step % check_interval == 0
        if not should_check:
            self._success_debug_step += 1
            return self._stable_success_result
        fk_audit_steps = _env_int("FLIP_TABLE_FK_AUDIT_STEPS", 0, minimum=0)
        if self._success_debug_step < fk_audit_steps:
            self._log_runtime_analysis_pose(
                env,
                torch.arange(env.num_envs, device=env.device, dtype=torch.long),
            )
        components = self._stable_flip_success_components(env)
        candidate = components["candidate"]
        hold_steps = _env_int("FLIP_TABLE_SUCCESS_HOLD_STEPS", 20, minimum=1)
        reset = env.episode_length_buf <= 1
        streak_increment = torch.where(
            self._stable_success_previous_candidate,
            torch.full_like(self._stable_success_streak, check_interval),
            torch.ones_like(self._stable_success_streak),
        )
        self._stable_success_streak = torch.where(
            reset,
            torch.zeros_like(self._stable_success_streak),
            torch.where(
                candidate,
                self._stable_success_streak + streak_increment,
                torch.zeros_like(self._stable_success_streak),
            ),
        )
        success = candidate & (self._stable_success_streak >= hold_steps)
        self._stable_success_previous_candidate = torch.where(
            reset,
            torch.zeros_like(candidate),
            candidate,
        )
        self._stable_success_result = success

        debug_every = _env_int("FLIP_TABLE_SUCCESS_DEBUG_EVERY", 0, minimum=0)
        if debug_every > 0 and self._success_debug_step % debug_every == 0:
            values = {
                key: value.detach().cpu().tolist()
                for key, value in components.items()
                if key != "candidate"
            }
            print(
                "[FlipTableEvalTask] success debug: "
                f"step={self._success_debug_step}, components={values}, "
                f"streak={self._stable_success_streak.detach().cpu().tolist()}, "
                f"hold_steps={hold_steps}, "
                f"check_interval={check_interval}, "
                f"{self._debug_table_part_positions(env)}, "
                f"{self._debug_robot_body_positions(env)}, "
                f"{self._debug_gripper_contacts(env)}, "
                f"{self._debug_collision_bounds(env)}, "
                f"{self._debug_wbc_action_term(env)}, "
                f"{self._debug_global_camera_pose(env)}",
                flush=True,
            )
        self._success_debug_step += 1
        return success

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["task_name"] = self.task_name
        ep_meta["lang"] = "flip the already assembled table over"
        ep_meta["success_condition"] = (
            "The tabletop normal is inverted, the tabletop has risen into the upright "
            "configuration, the assembled table is nearly stationary, its center remains on "
            "the workbench, and both grippers are clear for the configured hold duration."
        )
        ep_meta["flip_table_success"] = {
            "normal_dot_threshold": _env_float("FLIP_TABLE_SUCCESS_DOT_THRESHOLD", -0.95),
            "min_tabletop_lift_m": _env_float(
                "FLIP_TABLE_SUCCESS_MIN_TABLETOP_LIFT_M", 0.35
            ),
            "max_linear_speed_m_s": _env_float(
                "FLIP_TABLE_SUCCESS_MAX_LINEAR_SPEED_M_S", 0.15
            ),
            "max_angular_speed_rad_s": _env_float(
                "FLIP_TABLE_SUCCESS_MAX_ANGULAR_SPEED_RAD_S", 0.50
            ),
            "workbench_edge_margin_m": _env_float(
                "FLIP_TABLE_SUCCESS_WORKBENCH_EDGE_MARGIN_M", 0.03
            ),
            "hold_steps": _env_int("FLIP_TABLE_SUCCESS_HOLD_STEPS", 20, minimum=1),
        }
        ep_meta["flip_table_eval_randomization"] = {
            "evaluation_mode": os.environ.get("FLIP_TABLE_EVAL_MODE", "randomized"),
            "groot_dr_profile": os.environ.get(
                "FLIP_TABLE_GROOT_DR_PROFILE", "generic_v1"
            ),
            "table_long_range_m": _env_float("FLIP_TABLE_TABLE_LONG_RANGE_M", 0.12),
            "table_depth_range_m": _env_float("FLIP_TABLE_TABLE_DEPTH_RANGE_M", 0.035),
            "table_yaw_range_rad": _env_float("FLIP_TABLE_TABLE_YAW_RANGE_RAD", math.pi),
            "table_yaw_offset_rad": _env_float("FLIP_TABLE_TABLE_YAW_OFFSET_RAD", 0.0),
            "robot_distance_m": _env_float("FLIP_TABLE_ROBOT_DISTANCE_M", 0.26),
            "robot_distance_range_m": _env_float("FLIP_TABLE_ROBOT_DISTANCE_RANGE_M", 0.04),
            "robot_table_min_distance_m": _env_float("FLIP_TABLE_ROBOT_TABLE_MIN_DISTANCE_M", 0.62),
            "robot_wbc_settle_clearance_m": _env_float(
                "FLIP_TABLE_ROBOT_WBC_SETTLE_CLEARANCE_M", 0.03
            ),
            "robot_workbench_clearance_m": _env_float(
                "FLIP_TABLE_ROBOT_WORKBENCH_CLEARANCE_M", 0.20
            ),
            "robot_lateral_range_m": _env_float("FLIP_TABLE_ROBOT_LATERAL_RANGE_M", 0.10),
            "robot_yaw_range_rad": _env_float("FLIP_TABLE_ROBOT_YAW_RANGE_RAD", 0.08),
            "robot_yaw_offset_rad": _env_float("FLIP_TABLE_ROBOT_YAW_OFFSET_RAD", 0.0),
            "robot_base_height_m": _env_float("FLIP_TABLE_ROBOT_BASE_HEIGHT_M", 0.78),
            "robot_root_pos_local": os.environ.get("FLIP_TABLE_ROBOT_ROOT_POS_LOCAL", ""),
            "robot_root_yaw_rad": os.environ.get("FLIP_TABLE_ROBOT_ROOT_YAW_RAD", ""),
            "use_default_robot_pose": _env_bool("FLIP_TABLE_USE_DEFAULT_ROBOT_POSE", False),
            "default_robot_right_cells": _env_float("FLIP_TABLE_DEFAULT_ROBOT_RIGHT_CELLS", 0.0),
            "default_robot_forward_cells": _env_float("FLIP_TABLE_DEFAULT_ROBOT_FORWARD_CELLS", 0.0),
            "default_robot_yaw_offset_rad": _env_float("FLIP_TABLE_DEFAULT_ROBOT_YAW_OFFSET_RAD", 0.0),
            "debug_grid_cell_m": _env_float("FLIP_TABLE_DEBUG_GRID_CELL_M", 0.25),
            "workbench_front_axis": os.environ.get("FLIP_TABLE_WORKBENCH_FRONT_AXIS", "-y"),
            "joint_noise_rad": _env_float("FLIP_TABLE_JOINT_NOISE_RAD", 0.02),
            "randomize_upper_body_pose": _env_bool("FLIP_TABLE_RANDOMIZE_UPPER_BODY_POSE", True),
            "upper_body_pose_range_scale": _env_float("FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE", 0.5),
            "upper_body_joint_ranges_rad": {
                **FLIP_TABLE_UPPER_BODY_INITIAL_POSE_RANGES_RAD,
                **_env_named_float_map("FLIP_TABLE_UPPER_BODY_JOINT_RANGES_RAD"),
            },
            "dex1_finger_noise_m": _env_float("FLIP_TABLE_DEX1_FINGER_NOISE_M", 0.002),
            "initial_upper_body_state_override": _env_float_vector(
                "FLIP_TABLE_INITIAL_UPPER_BODY_STATE", 19
            )
            is not None,
            "initial_full_body_state_override": _env_float_vector(
                "FLIP_TABLE_INITIAL_FULL_BODY_STATE", 31
            )
            is not None,
            "body_mode": self._sim_body_mode,
            "lock_lower_body": _env_bool("FLIP_TABLE_LOCK_LOWER_BODY", False),
            "lock_robot_root": _env_bool("FLIP_TABLE_LOCK_ROBOT_ROOT", False),
            "randomize_room": _env_bool("FLIP_TABLE_RANDOMIZE_ROOM", True),
            "room_floor_materials": ",".join(
                _env_choices("FLIP_TABLE_ROOM_FLOOR_MATERIALS", tuple(spec[0] for spec in ROOM_FLOOR_MATERIALS))
            ),
            "room_wall_materials": ",".join(
                _env_choices("FLIP_TABLE_ROOM_WALL_MATERIALS", tuple(spec[0] for spec in ROOM_WALL_MATERIALS))
            ),
            "randomize_room_props": _env_bool("FLIP_TABLE_RANDOMIZE_ROOM_PROPS", True),
            "room_prop_assets": ",".join(
                _env_choices("FLIP_TABLE_ROOM_PROP_ASSETS", tuple(spec[0] for spec in ROOM_PROP_ASSETS))
            ),
            "room_prop_slots": int(_env_float("FLIP_TABLE_ROOM_PROP_SLOTS", 10)),
            "room_prop_visible_probability": _env_float("FLIP_TABLE_ROOM_PROP_VISIBLE_PROBABILITY", 0.62),
            "room_prop_x_range_m": os.environ.get("FLIP_TABLE_ROOM_PROP_X_RANGE_M", "-4.8,4.8"),
            "room_prop_y_range_m": os.environ.get("FLIP_TABLE_ROOM_PROP_Y_RANGE_M", "-4.8,4.8"),
            "room_prop_yaw_range_rad": os.environ.get(
                "FLIP_TABLE_ROOM_PROP_YAW_RANGE_RAD", f"{-math.pi},{math.pi}"
            ),
            "room_prop_scale_range": os.environ.get("FLIP_TABLE_ROOM_PROP_SCALE_RANGE", "0.80,1.18"),
            "room_prop_safe_radius_m": _env_float("FLIP_TABLE_ROOM_PROP_SAFE_RADIUS_M", 2.20),
            "room_prop_min_separation_m": _env_float("FLIP_TABLE_ROOM_PROP_MIN_SEPARATION_M", 0.30),
            "room_prop_wall_clearance_m": _env_float("FLIP_TABLE_ROOM_PROP_WALL_CLEARANCE_M", 0.20),
            "room_prop_front_min_distance_m": _env_float(
                "FLIP_TABLE_ROOM_PROP_FRONT_MIN_DISTANCE_M", 0.50
            ),
            "room_prop_front_half_angle_deg": _env_float(
                "FLIP_TABLE_ROOM_PROP_FRONT_HALF_ANGLE_DEG", 80.0
            ),
            "room_prop_front_axis": os.environ.get("FLIP_TABLE_ROOM_PROP_FRONT_AXIS", "+x"),
            "randomize_contact_materials": _env_bool("FLIP_TABLE_RANDOMIZE_CONTACT_MATERIALS", True),
            "contact_hand_white_static_range": os.environ.get(
                "FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE", "0.65,0.95"
            ),
            "contact_hand_white_dynamic_range": os.environ.get(
                "FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE", "0.48,0.64"
            ),
            "contact_hand_white_restitution_range": os.environ.get(
                "FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE", "0.02,0.08"
            ),
            "contact_white_workbench_static_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE", "0.50,0.75"
            ),
            "contact_white_workbench_dynamic_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE", "0.35,0.46"
            ),
            "contact_white_workbench_restitution_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE", "0.01,0.05"
            ),
            "contact_workbench_hand_static_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE", "0.60,0.90"
            ),
            "contact_workbench_hand_dynamic_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE", "0.42,0.56"
            ),
            "contact_workbench_hand_restitution_range": os.environ.get(
                "FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE", "0.02,0.08"
            ),
            "room_window_visible_probability": _env_float(
                "FLIP_TABLE_ROOM_WINDOW_VISIBLE_PROBABILITY", 0.72
            ),
            "room_floor_half_extents_m": os.environ.get("FLIP_TABLE_ROOM_FLOOR_HALF_EXTENTS_M", "5.5,7.5"),
            "room_wall_height_m": os.environ.get("FLIP_TABLE_ROOM_WALL_HEIGHT_M", "4.0,5.5"),
            "room_tile_size_m": os.environ.get("FLIP_TABLE_ROOM_TILE_SIZE_M", "0.35,0.9"),
            "room_tile_line_width_m": os.environ.get("FLIP_TABLE_ROOM_TILE_LINE_WIDTH_M", "0.008,0.025"),
            "room_color_jitter": _env_float("FLIP_TABLE_ROOM_COLOR_JITTER", 0.08),
            "room_floor_patterns": ",".join(_env_choices("FLIP_TABLE_ROOM_FLOOR_PATTERNS", ROOM_FLOOR_PATTERNS)),
            "room_wall_patterns": ",".join(_env_choices("FLIP_TABLE_ROOM_WALL_PATTERNS", ROOM_WALL_PATTERNS)),
            "room_max_pattern_prims": int(_env_float("FLIP_TABLE_ROOM_MAX_PATTERN_PRIMS", 96)),
            "indoor_light_temperature_k": os.environ.get(
                "FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K", "3800,6500"
            ),
            "indoor_light_count_range": os.environ.get("FLIP_TABLE_INDOOR_LIGHT_COUNT_RANGE", "2,4"),
            "sun_visible_probability": _env_float("FLIP_TABLE_SUN_VISIBLE_PROBABILITY", 0.78),
            "sun_light_temperature_k": os.environ.get(
                "FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K", "5000,7000"
            ),
            "sun_light_intensity_range": os.environ.get(
                "FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE", "180,750"
            ),
            "sun_elevation_deg": os.environ.get("FLIP_TABLE_SUN_ELEVATION_DEG", "18,72"),
            "sun_azimuth_deg": os.environ.get("FLIP_TABLE_SUN_AZIMUTH_DEG", "-180,180"),
            "light_exposure_range": os.environ.get("FLIP_TABLE_LIGHT_EXPOSURE_RANGE", "-0.35,0.35"),
        }
        return ep_meta
