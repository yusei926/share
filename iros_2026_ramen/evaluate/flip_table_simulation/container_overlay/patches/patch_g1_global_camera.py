#!/usr/bin/env python3
"""Build a reproducible Dex1 G1 configuration for flip-table evaluation.

RoboFinals-IKEA-V1 ships a three-finger G1 config whose hand-action helper calls
an API absent from the runtime packaged in the same image. The competition task
uses Dex1-1, so this startup patch restores the immutable V1 Python baseline,
repairs that API mismatch, and adds a deterministic Dex1/WBC embodiment without
reusing the organizer's overpowered WBC-training arm actuators. The direct
joint-target profile is bounded by the generated G1/Dex1 URDF and uses the
real G1 arm-controller gains. The patch then configures the real camera
contract, selects the generated G1_GRIPPER USD variants, and repairs missing
MaterialBindingAPI schemas while preserving geometry and physics data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import os
from pathlib import Path
import shutil


DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
POLICY_CAMERA_UPDATE_PERIOD = "0.03333333333333333"
D405_FOCAL_LENGTH = "24.0"
# This is the mean of the two pinned raw D405 color calibrations in
# ``pipeline_v1.json``.  Keep the USD camera square-pixel and centered, then
# apply each recorded sensor's principal point/distortion in the policy-image
# adapter.  The old 45.55 mm value came from a product-sheet FOV and produced
# a visibly wider image than the cameras that recorded this dataset.
D405_HORIZONTAL_APERTURE = "35.31010639776536"
D405_VERTICAL_APERTURE = "26.482579798324018"
HEAD_LEFT_OPTICAL_CENTER_POS = "(0.10209156, 0.02077481159355057, 0.42446595)"
# The measured head baseline is 60.30046318710113 mm around the organizer V1
# rig center y=-9.37542 mm.  Both sim eyes use the rectified optical rotation;
# head-right is operator-only and never enters a policy feature map.
HEAD_RIGHT_OPTICAL_CENTER_POS = "(0.10209156, -0.039525651593550565, 0.42446595)"
HEAD_LEFT_FOCAL_LENGTH = "24.0"
HEAD_LEFT_HORIZONTAL_APERTURE = "45.56883749280177"
HEAD_LEFT_VERTICAL_APERTURE = "34.176628119601325"
# The competition Dex1/WBC configuration defines the wrist camera parent frames
# and nominal bracket mount. The local Dex1-1 D405 bracket STEP and
# Unitree Device.md confirm this is a wrist/M5010-ring mount, not a palm-frame
# mount. A direct STEP-to-wrist transform is underconstrained without the real
# hand-eye calibration/TF tree and placed the camera outside the real dataset
# distribution, so the default below keeps the organizer wrist-link x/y mount and
# calibrates forward/z plus the optical axis against the real flip-table wrist
# frames.
#
# The aperture below is derived from the pinned two-device mean raw D405 color
# intrinsics, not a generic D405 product-sheet FOV. With focal_length=24 mm it
# yields approximately 72.68 x 57.77 degrees. The rotation starts from the V1
# Dex1/WBC hand-camera quaternion and applies a -20 degree optical-axis
# adjustment. The current default is the user-selected x115/z70/down20
# candidate from the frame-10 Dex1+D405 render sweep; keep it guarded with
# real wrist-image distribution checks so future changes remain explicit.
# Keep this guarded with tools/verify_d405_wrist_camera_calibration.py.
DEX1_D405_LEFT_OPTICAL_CENTER_POS = "(0.11500, 0, 0.07000)"
DEX1_D405_RIGHT_OPTICAL_CENTER_POS = "(0.11500, 0, 0.07000)"
DEX1_D405_OPTICAL_CENTER_ROT = "(0.28059008, -0.25108559, -0.64082457, 0.66900606)"
# Captured from the accepted torso-mounted inspection view after the fixed
# scene reset. The camera is spawned under the environment root so these are
# environment/world coordinates and never inherit robot articulation motion.
GLOBAL_CAMERA_WORLD_POS = "(-0.977742, 2.372581, 1.994000)"
GLOBAL_CAMERA_WORLD_ROT = "(0.25915552, 0.28195817, 0.64845940, 0.65790457)"
DEFAULT_CAMERA_NAMES = (
    "first_person_camera",
    "head_right_camera",
    "left_hand_camera",
    "right_hand_camera",
    "hand_camera",
    "global_camera",
    "left_shoulder_camera",
    "right_shoulder_camera",
    "eye_in_hand_camera",
    "d435_camera",
)
GRIPPER_CFG_CLASSES = ("UnitreeG1GripperControllerDecoupledWBCEnvCfg",)
GLOBAL_CAMERA_CFG_CLASSES = ("UnitreeG1EnvCfg",) + GRIPPER_CFG_CLASSES
BACKUP_SUFFIX = ".original_flip_table_global_camera"
ASSETS_BACKUP_SUFFIX = ".original_flip_table_gripper_variants"
MATERIAL_BINDING_BACKUP_SUFFIX = ".original_flip_table_material_binding_api"
CONTACT_MATERIAL_BACKUP_SUFFIX = ".original_flip_table_contact_material"
PHYSX_COLLISION_BACKUP_SUFFIX = ".original_flip_table_physx_collision"
CAMERA_CONFIG_FIELDS_MARKER = "# FLIP_TABLE_CAMERA_CONFIG_FIELDS_V1"
GRIPPER_VARIANTS = {
    "Physics": "PhysX",
    "Robot": "Robot",
    "Sensor": "Sensors",
}
UNITREE_G1_PBR_MATERIALS = {
    # Exact OmniPBR values from Unitree's official unitree_sim_isaaclab
    # g1_29dof_with_dex1_base_fix1.usd. RoboFinals uses a simplified Dex1
    # mesh/material set, so its 696969 finger material uses Unitree's second
    # dark Dex1 surface value.
    "material_dark": ((0.01544404, 0.014847745, 0.014847745), 1.0, 0.50, 0.22),
    "material_white": ((0.7, 0.7, 0.71), 1.0, 0.60, 0.27),
    "material_CAD1EE": ((0.014907375, 0.015011734, 0.01544404), 1.0, 0.50, 0.22),
    "material_696969": ((0.029933952, 0.030530358, 0.030888021), 1.0, 0.50, 0.22),
    "material_E5EAED": ((0.029933952, 0.030530358, 0.030888021), 1.0, 0.50, 0.22),
}
UPPER_BODY_JOINT_ACTION_CLASS = "FlipTableUpperBodyJointActionsCfg"
UPPER_BODY_JOINT_ACTION_ENV = "FLIP_TABLE_ACT_USE_JOINT_POSITION_ACTION"
PINK_EEF_ACTION_CLASS = "FlipTablePinkEEFActionsCfg"
PINK_EEF_ACTION_ENV = "FLIP_TABLE_USE_PINK_EEF_ACTION"
ROBOT_COLLISION_ENV = "FLIP_TABLE_ENABLE_ROBOT_COLLISIONS"
ROBOT_SELF_COLLISION_ENV = "FLIP_TABLE_ENABLE_ROBOT_SELF_COLLISIONS"
ROBOT_USD_PATH_ENV = "FLIP_TABLE_G1_USD_PATH"
MATERIAL_BINDING_PATCH_ENV = "FLIP_TABLE_PATCH_G1_GRIPPER_MATERIAL_BINDINGS"
UNITREE_MATERIAL_VALUES_ENV = "FLIP_TABLE_MATCH_UNITREE_G1_MATERIAL_VALUES"
DISABLE_COLLISION_FILTER_ENV = "FLIP_TABLE_DISABLE_COLLISION_FILTER"
CONTACT_MATERIAL_PATCH_ENV = "FLIP_TABLE_PATCH_G1_CONTACT_MATERIAL"
RL_CAMERA_ENV = "FLIP_TABLE_ENABLE_RL_CAMERAS"
G1_CONTACT_SENSOR_FIELD_MARKER = "left_gripper_contact: ContactSensorCfg"
G1_SECOND_CONTACT_SENSOR_FIELD_MARKER = "left_gripper_contact_2: ContactSensorCfg"
WHITE_LEG_CONTACT_SENSOR_FIELD_MARKER = "white_leg_contact_0: ContactSensorCfg"
LEGACY_WHITE_TABLE_CONTACT_FILTER_NAME = "_FLIP_TABLE_WHITE_TABLE_CONTACT_FILTERS"
OFFICIAL_V1_BACKUP_ROOT_ENV = "FLIP_TABLE_OFFICIAL_V1_BACKUP_ROOT"
RESTORE_OFFICIAL_V1_ENV = "FLIP_TABLE_RESTORE_OFFICIAL_V1_ROBOT_FILES"
OFFICIAL_V1_ROBOT_FILES = (
    Path("robofinals/core/robots/unitree/g1.py"),
    Path("robofinals/core/robots/unitree/assets_cfg.py"),
)
OFFICIAL_V1_SOURCE_SHA256 = {
    "robofinals/core/robots/unitree/g1.py":
        "4b42f29b5732e3e8cb7ded512ba67bae9cc40370c48b1a9e3c325cf6eb229ea9",
    "robofinals/core/robots/unitree/assets_cfg.py":
        "4ec02f85b65ba9588e7bd24e3785a280343149dc5e84da89e8b5bb50d02c114a",
}
# The V1 startup process generates the Dex1 USD after the immutable source tree
# is unpacked. These first-write backups are therefore the pristine source for
# repository-owned USD edits; they do not exist on a fresh first run.
GENERATED_ASSET_BACKUP_SPECS = (
    (
        Path("robofinals/data/assets/g1_urdf_gripper/G1_GRIPPER.usd"),
        CONTACT_MATERIAL_BACKUP_SUFFIX,
    ),
    (
        Path("robofinals/data/assets/g1_urdf_gripper/configuration/usd_base.usd"),
        MATERIAL_BINDING_BACKUP_SUFFIX,
    ),
    (
        Path("robofinals/data/assets/g1_urdf_gripper/configuration/usd_physics.usd"),
        PHYSX_COLLISION_BACKUP_SUFFIX,
    ),
)
DIRECT_TARGET_ACTUATOR_PROFILE_MARKER = "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V5"
LEGACY_DIRECT_TARGET_ACTUATOR_PROFILE_MARKERS = (
    "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V1",
    "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V2",
    "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V3",
    "FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V4",
)


@dataclass(frozen=True)
class CameraGeometry:
    camera_name: str
    prim_path: str
    pos: str
    rot: str
    convention: str
    focal_length: str
    horizontal_aperture: str
    clipping_range: str
    update_period: str
    class_names: tuple[str, ...] | None = None
    vertical_aperture: str | None = None
    focus_distance: str = "400.0"
    tags: str = "[]"
    execute_modes: str = "[ExecuteMode.EVAL]"
    insert_if_missing: bool = False


POLICY_CAMERA_GEOMETRIES = (
    CameraGeometry(
        camera_name="first_person_camera",
        class_names=GRIPPER_CFG_CLASSES,
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/first_person_camera",
        pos=HEAD_LEFT_OPTICAL_CENTER_POS,
        rot="(0.26523914, -0.27106013, -0.66472446, 0.64367383)",
        convention="opengl",
        focal_length=HEAD_LEFT_FOCAL_LENGTH,
        horizontal_aperture=HEAD_LEFT_HORIZONTAL_APERTURE,
        vertical_aperture=HEAD_LEFT_VERTICAL_APERTURE,
        clipping_range="(0.1, 1.0e5)",
        update_period=POLICY_CAMERA_UPDATE_PERIOD,
    ),
    CameraGeometry(
        camera_name="head_right_camera",
        class_names=GRIPPER_CFG_CLASSES,
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_right_camera",
        pos=HEAD_RIGHT_OPTICAL_CENTER_POS,
        rot="(0.26523914, -0.27106013, -0.66472446, 0.64367383)",
        convention="opengl",
        focal_length=HEAD_LEFT_FOCAL_LENGTH,
        horizontal_aperture=HEAD_LEFT_HORIZONTAL_APERTURE,
        vertical_aperture=HEAD_LEFT_VERTICAL_APERTURE,
        clipping_range="(0.1, 1.0e5)",
        update_period=POLICY_CAMERA_UPDATE_PERIOD,
        insert_if_missing=True,
    ),
    CameraGeometry(
        camera_name="left_hand_camera",
        class_names=GRIPPER_CFG_CLASSES,
        prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link/left_hand_camera",
        pos=DEX1_D405_LEFT_OPTICAL_CENTER_POS,
        rot=DEX1_D405_OPTICAL_CENTER_ROT,
        convention="opengl",
        focal_length=D405_FOCAL_LENGTH,
        horizontal_aperture=D405_HORIZONTAL_APERTURE,
        vertical_aperture=D405_VERTICAL_APERTURE,
        clipping_range="(0.01, 50.0)",
        update_period=POLICY_CAMERA_UPDATE_PERIOD,
    ),
    CameraGeometry(
        camera_name="right_hand_camera",
        class_names=GRIPPER_CFG_CLASSES,
        prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link/right_hand_camera",
        pos=DEX1_D405_RIGHT_OPTICAL_CENTER_POS,
        rot=DEX1_D405_OPTICAL_CENTER_ROT,
        convention="opengl",
        focal_length=D405_FOCAL_LENGTH,
        horizontal_aperture=D405_HORIZONTAL_APERTURE,
        vertical_aperture=D405_VERTICAL_APERTURE,
        clipping_range="(0.01, 50.0)",
        update_period=POLICY_CAMERA_UPDATE_PERIOD,
    ),
    CameraGeometry(
        camera_name="global_camera",
        class_names=GLOBAL_CAMERA_CFG_CLASSES,
        prim_path="{ENV_REGEX_NS}/global_camera",
        pos=GLOBAL_CAMERA_WORLD_POS,
        rot=GLOBAL_CAMERA_WORLD_ROT,
        convention="opengl",
        focal_length="24.0",
        horizontal_aperture="90",
        clipping_range="(0.1, 1.0e5)",
        update_period="0.05",
        insert_if_missing=True,
    ),
)


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


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _env_camera_names(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_camera_tuple(name: str, default: str, size: int) -> str:
    """Return a validated Python tuple literal for a camera offset field."""

    raw = _env_str(name, default).strip()
    if raw.startswith("(") and raw.endswith(")"):
        raw = raw[1:-1].strip()
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != size:
        raise ValueError(f"{name} must contain exactly {size} finite values, got {raw!r}")
    try:
        numbers = [float(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values, got {raw!r}") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{name} must contain finite values, got {raw!r}")
    return "(" + ", ".join(f"{value:.9g}" for value in numbers) + ")"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_official_v1_robot_files(
    robofinals_root: Path,
    backup_root: Path,
    expected_sha256: dict[str, str] | None = None,
) -> dict[str, str]:
    """Restore mutable vendor files before applying repository-owned patches."""

    target_root = robofinals_root.resolve()
    source_root = backup_root.resolve()
    if target_root == source_root:
        raise ValueError("official V1 backup root must differ from the mutable RoboFinals root")

    restored: dict[str, str] = {}
    for relative in OFFICIAL_V1_ROBOT_FILES:
        source = source_root / relative
        target = target_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"official V1 source file is missing: {source}")
        source_digest = _sha256(source)
        expected = (expected_sha256 or {}).get(str(relative))
        if expected is not None and source_digest != expected:
            raise RuntimeError(
                f"official V1 source hash mismatch for {relative}: "
                f"expected={expected}, actual={source_digest}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        restored[str(relative)] = _sha256(target)
    print(
        "[flip_table] restored official V1 robot baseline: "
        + ", ".join(f"{path}={digest[:12]}" for path, digest in restored.items()),
        flush=True,
    )
    return restored


def restore_generated_v1_assets(robofinals_root: Path) -> dict[str, str]:
    """Restore generated V1 USDs from the first repository patch backup."""

    root = robofinals_root.resolve()
    restored: dict[str, str] = {}
    for relative, suffix in GENERATED_ASSET_BACKUP_SPECS:
        target = root / relative
        backup = Path(str(target) + suffix)
        if not backup.is_file():
            continue
        if not target.is_file():
            raise FileNotFoundError(f"generated V1 asset is missing: {target}")
        shutil.copy2(backup, target)
        restored[str(relative)] = _sha256(target)
    if restored:
        print(
            "[flip_table] restored generated V1 asset baseline: "
            + ", ".join(f"{path}={digest[:12]}" for path, digest in restored.items()),
            flush=True,
        )
    return restored


def _ensure_gripper_asset_cfg_text(text: str) -> str:
    """Add the Dex1 articulation config missing from the organizer V1 module."""

    if DIRECT_TARGET_ACTUATOR_PROFILE_MARKER in text:
        return text
    marker = "G1_WUJI_ASSET_PATH ="
    index = text.find(marker)
    existing = text.find("G1_GRIPPER_CFG = G1_GEARWBC_CFG.copy()")
    legacy_candidates = [
        text.find(f"# {marker}")
        for marker in LEGACY_DIRECT_TARGET_ACTUATOR_PROFILE_MARKERS
    ]
    legacy = min((value for value in legacy_candidates if value >= 0), default=-1)
    replacement_start = legacy if 0 <= legacy < existing else existing
    def positive_scale(name: str) -> float:
        raw = os.environ.get(name, "").strip()
        # This conservative startup value is not a calibrated physical
        # parameter.  Recorded-replay identification may override it at reset
        # time, but it must not be promoted to a shared default until the
        # fixed parameter set passes the held-out acceptance gate.
        if name == "FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE" and not raw:
            value = 0.5
        else:
            value = 1.0 if not raw else float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite positive scale")
        return value

    def nonnegative_value(name: str) -> float:
        raw = os.environ.get(name, "").strip()
        value = 0.0 if not raw else float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")
        return value

    stiffness_scale = positive_scale("FLIP_TABLE_CALIBRATION_ARM_STIFFNESS_SCALE")
    damping_scale = positive_scale("FLIP_TABLE_CALIBRATION_ARM_DAMPING_SCALE")
    armature_scale = positive_scale("FLIP_TABLE_CALIBRATION_ARM_ARMATURE_SCALE")
    friction_nm = nonnegative_value("FLIP_TABLE_CALIBRATION_ARM_FRICTION_NM")
    dex1_stiffness_scale = positive_scale("FLIP_TABLE_CALIBRATION_DEX1_STIFFNESS_SCALE")
    dex1_damping_scale = positive_scale("FLIP_TABLE_CALIBRATION_DEX1_DAMPING_SCALE")
    block = '''# FLIP_TABLE_DIRECT_TARGET_ACTUATOR_PROFILE_V5
G1_GRIPPER_CFG = G1_GEARWBC_CFG.copy()
G1_GRIPPER_CFG.spawn.usd_path = str(robofinals_DATA_PATH / "assets" / "g1_urdf_gripper" / "G1_GRIPPER.usd")
G1_GRIPPER_CFG.spawn.rigid_props.solver_position_iteration_count = 8
G1_GRIPPER_CFG.spawn.rigid_props.solver_velocity_iteration_count = 4
G1_GRIPPER_CFG.spawn.articulation_props.solver_position_iteration_count = 8
G1_GRIPPER_CFG.spawn.articulation_props.solver_velocity_iteration_count = 4
G1_GRIPPER_CFG.actuators.pop("hands", None)
G1_GRIPPER_CFG.actuators.pop("gearwbc_implicit_arms", None)
for _joint_name in list(G1_GRIPPER_CFG.init_state.joint_pos):
    if "_hand_" in _joint_name:
        del G1_GRIPPER_CFG.init_state.joint_pos[_joint_name]
G1_GRIPPER_CFG.init_state.joint_pos.update(
    {
        "left_dex1_finger_joint_1": 0.0245,
        "left_dex1_finger_joint_2": 0.0245,
        "right_dex1_finger_joint_1": 0.0245,
        "right_dex1_finger_joint_2": 0.0245,
    }
)
# The copied WBC arm profile allows up to 200 Nm at 25 Nm G1 arm joints. It is
# removed above so every arm joint has exactly one actuator. The inherited leg
# motor groups remain responsible for the waist; adding another waist actuator
# would likewise create overlapping drives. This embodiment receives absolute
# joint targets, like xr_teleoperate on the real robot, so use the generated
# URDF limits and the real arm-controller gains.
G1_GRIPPER_CFG.actuators["arms"] = IdealPDActuatorCfg(
    joint_names_expr=[
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    ],
    effort_limit={
        ".*_shoulder_pitch_joint": 25.0,
        ".*_shoulder_roll_joint": 25.0,
        ".*_shoulder_yaw_joint": 25.0,
        ".*_elbow_joint": 25.0,
        ".*_wrist_roll_joint": 25.0,
        ".*_wrist_pitch_joint": 13.4,
        ".*_wrist_yaw_joint": 13.4,
    },
    effort_limit_sim={
        ".*_shoulder_pitch_joint": 25.0,
        ".*_shoulder_roll_joint": 25.0,
        ".*_shoulder_yaw_joint": 25.0,
        ".*_elbow_joint": 25.0,
        ".*_wrist_roll_joint": 25.0,
        ".*_wrist_pitch_joint": 13.4,
        ".*_wrist_yaw_joint": 13.4,
    },
    velocity_limit={
        ".*_shoulder_pitch_joint": 37.0,
        ".*_shoulder_roll_joint": 37.0,
        ".*_shoulder_yaw_joint": 37.0,
        ".*_elbow_joint": 37.0,
        ".*_wrist_roll_joint": 37.0,
        ".*_wrist_pitch_joint": 27.0,
        ".*_wrist_yaw_joint": 27.0,
    },
    velocity_limit_sim={
        ".*_shoulder_pitch_joint": 37.0,
        ".*_shoulder_roll_joint": 37.0,
        ".*_shoulder_yaw_joint": 37.0,
        ".*_elbow_joint": 37.0,
        ".*_wrist_roll_joint": 37.0,
        ".*_wrist_pitch_joint": 27.0,
        ".*_wrist_yaw_joint": 27.0,
    },
    stiffness={
        ".*_shoulder_.*_joint": __FLIP_TABLE_ARM_SHOULDER_STIFFNESS__,
        ".*_elbow_joint": __FLIP_TABLE_ARM_ELBOW_STIFFNESS__,
        ".*_wrist_.*_joint": __FLIP_TABLE_ARM_WRIST_STIFFNESS__,
    },
    damping={
        ".*_shoulder_.*_joint": __FLIP_TABLE_ARM_SHOULDER_DAMPING__,
        ".*_elbow_joint": __FLIP_TABLE_ARM_ELBOW_DAMPING__,
        ".*_wrist_.*_joint": __FLIP_TABLE_ARM_WRIST_DAMPING__,
    },
    armature=__FLIP_TABLE_ARM_ARMATURE__,
    friction=__FLIP_TABLE_ARM_FRICTION__,
)
# Unitree's Dex1 Isaac Lab profile uses 800/3 position-drive gains. The URDF
# still supplies the physical 20 N and 0.2 m/s prismatic limits.
G1_GRIPPER_CFG.actuators["grippers"] = IdealPDActuatorCfg(
    joint_names_expr=[".*_dex1_finger_joint_.*"],
    effort_limit=20.0,
    effort_limit_sim=20.0,
    velocity_limit=0.2,
    velocity_limit_sim=0.2,
    stiffness=__FLIP_TABLE_DEX1_STIFFNESS__,
    damping=__FLIP_TABLE_DEX1_DAMPING__,
    armature=0.01,
    friction=0.0,
)


'''
    substitutions = {
        "__FLIP_TABLE_ARM_SHOULDER_STIFFNESS__": repr(80.0 * stiffness_scale),
        "__FLIP_TABLE_ARM_ELBOW_STIFFNESS__": repr(80.0 * stiffness_scale),
        "__FLIP_TABLE_ARM_WRIST_STIFFNESS__": repr(40.0 * stiffness_scale),
        "__FLIP_TABLE_ARM_SHOULDER_DAMPING__": repr(3.0 * damping_scale),
        "__FLIP_TABLE_ARM_ELBOW_DAMPING__": repr(3.0 * damping_scale),
        "__FLIP_TABLE_ARM_WRIST_DAMPING__": repr(1.5 * damping_scale),
        "__FLIP_TABLE_ARM_ARMATURE__": repr(0.01 * armature_scale),
        "__FLIP_TABLE_ARM_FRICTION__": repr(friction_nm),
        "__FLIP_TABLE_DEX1_STIFFNESS__": repr(800.0 * dex1_stiffness_scale),
        "__FLIP_TABLE_DEX1_DAMPING__": repr(3.0 * dex1_damping_scale),
    }
    for token, value in substitutions.items():
        block = block.replace(token, value)
    print(
        "[flip_table] selected calibrated arm IdealPD profile: "
        f"stiffness_scale={stiffness_scale:.6g}, damping_scale={damping_scale:.6g}, "
        f"armature_scale={armature_scale:.6g}, friction_nm={friction_nm:.6g}; "
        f"Dex1 stiffness_scale={dex1_stiffness_scale:.6g}, damping_scale={dex1_damping_scale:.6g}",
        flush=True,
    )
    if existing >= 0:
        replacement_end = index if index > existing else len(text)
        return text[:replacement_start] + block + text[replacement_end:]
    if index < 0:
        raise RuntimeError("Could not find G1_WUJI_ASSET_PATH in assets_cfg.py")
    return text[:index] + block + text[index:]


def patch_g1_gripper_asset_config(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "core" / "robots" / "unitree" / "assets_cfg.py"
    if not target.is_file():
        raise FileNotFoundError(f"G1 assets config not found: {target}")
    current = target.read_text(encoding="utf-8")
    updated = _ensure_gripper_asset_cfg_text(current)
    if updated == current:
        return False
    backup = Path(str(target) + ASSETS_BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(current, encoding="utf-8")
    target.write_text(updated, encoding="utf-8")
    print(f"[flip_table] added deterministic V1 Dex1 articulation config: {target}", flush=True)
    return True


def _ensure_v1_runtime_api_text(text: str) -> str:
    """Adapt the V1 G1 helper to the strategy API bundled in the same image."""

    updated = text
    ee_helper = "def _g1_ee_target_frame_path("
    if "configure_g1_hand_action_cfg(" not in updated:
        start = updated.find("def _newton_g1_gripper_action_cfg(")
        end = updated.find(ee_helper, start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not find the RoboFinals V1 G1 hand-action helper block")
        replacement = '''def _set_g1_hand_action_cfg(action_config, gripper_cfg, hand_action_mode: str) -> None:
    get_strategy(get_context().physics_backend).configure_g1_hand_action_cfg(
        action_config,
        gripper_cfg,
        hand_action_mode,
    )


def _g1_robot_frame_path(link_name: str) -> str:
    return f"{{ENV_REGEX_NS}}/Robot/{link_name}"


'''
        updated = updated[:start] + replacement + updated[end:]
    elif "def _g1_robot_frame_path(" not in updated:
        index = updated.find(ee_helper)
        if index < 0:
            raise RuntimeError("Could not find the RoboFinals V1 G1 EE-frame helper")
        updated = (
            updated[:index]
            + 'def _g1_robot_frame_path(link_name: str) -> str:\n'
            + '    return f"{{ENV_REGEX_NS}}/Robot/{link_name}"\n\n\n'
            + updated[index:]
        )

    updated = updated.replace(
        '    return f"{{ENV_REGEX_NS}}/Robot/{link_name}"\n\n\n@configclass',
        '    return _g1_robot_frame_path(link_name)\n\n\n@configclass',
        1,
    )
    updated = updated.replace(
        'prim_path="{ENV_REGEX_NS}/Robot/base_link",',
        'prim_path=_g1_robot_frame_path("pelvis"),',
    )

    strategy_call = (
        "    def customize_physics_cfg(self, env_cfg) -> None:\n"
        "        get_strategy(get_context().physics_backend)."
        "customize_g1_controller_physics_cfg(env_cfg)\n\n"
    )
    if "customize_g1_controller_physics_cfg(env_cfg)" not in updated:
        start = updated.find("    def customize_physics_cfg(self, env_cfg) -> None:")
        end = updated.find("    def __init__(", start)
        if start >= 0 and end >= 0:
            updated = updated[:start] + strategy_call + updated[end:]
    return updated


def _validate_patched_runtime_booleans() -> None:
    """Fail before startup if a boolean consumed by patched ``g1.py`` is invalid."""

    for name, default in (
        (UPPER_BODY_JOINT_ACTION_ENV, False),
        (PINK_EEF_ACTION_ENV, False),
        ("FLIP_TABLE_FIX_ROOT_LINK", False),
        (ROBOT_COLLISION_ENV, True),
        (ROBOT_SELF_COLLISION_ENV, False),
        (DISABLE_COLLISION_FILTER_ENV, False),
    ):
        _env_bool(name, default)


def _geometry_with_env_overrides(geometry: CameraGeometry) -> CameraGeometry:
    side_prefixes = {
        "first_person_camera": "FLIP_TABLE_HEAD_LEFT_CAMERA",
        "head_right_camera": "FLIP_TABLE_HEAD_RIGHT_CAMERA",
        "left_hand_camera": "FLIP_TABLE_LEFT_WRIST_CAMERA",
        "right_hand_camera": "FLIP_TABLE_RIGHT_WRIST_CAMERA",
    }
    side_prefix = side_prefixes.get(geometry.camera_name)
    if side_prefix is None:
        return geometry

    common_prefix = (
        "FLIP_TABLE_DEX1_WRIST_CAMERA"
        if geometry.camera_name in {"left_hand_camera", "right_hand_camera"}
        else side_prefix
    )

    def value(suffix: str, current: str) -> str:
        return _env_str(f"{side_prefix}_{suffix}", _env_str(f"{common_prefix}_{suffix}", current))

    return replace(
        geometry,
        pos=_env_camera_tuple(f"{side_prefix}_OFFSET_POS", value("OFFSET_POS", geometry.pos), 3),
        rot=_env_camera_tuple(f"{side_prefix}_OFFSET_ROT", value("OFFSET_ROT", geometry.rot), 4),
        focal_length=value("FOCAL_LENGTH", geometry.focal_length),
        horizontal_aperture=value("HORIZONTAL_APERTURE", geometry.horizontal_aperture),
        vertical_aperture=(
            value("VERTICAL_APERTURE", geometry.vertical_aperture)
            if geometry.vertical_aperture is not None
            else None
        ),
        clipping_range=value("CLIPPING_RANGE", geometry.clipping_range),
        update_period=value("UPDATE_PERIOD", geometry.update_period),
    )


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _class_range(lines: list[str], class_name: str) -> tuple[int, int] | None:
    target = f"class {class_name}"
    start = None
    for index, line in enumerate(lines):
        if line.startswith(target):
            start = index
            break
    if start is None:
        return None

    end = len(lines)
    indent = _line_indent(lines[start])
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if _line_indent(lines[index]) == indent and stripped.startswith("class "):
            end = index
            break
    return start, end


def _camera_block_range_from_start(
    lines: list[str], start: int, search_end: int | None = None
) -> tuple[int, int]:
    indent = _line_indent(lines[start])
    end = len(lines) if search_end is None else search_end
    for index in range(start + 1, end):
        stripped = lines[index].strip()
        if _line_indent(lines[index]) == indent and stripped.startswith('"') and stripped.endswith("{"):
            end = index
            break
    return start, end


def _camera_block_ranges(
    lines: list[str], camera_name: str, search_start: int = 0, search_end: int | None = None
) -> list[tuple[int, int]]:
    end = len(lines) if search_end is None else search_end
    target = f'"{camera_name}": {{'
    ranges = []
    for index in range(search_start, end):
        if lines[index].strip() == target:
            ranges.append(_camera_block_range_from_start(lines, index, end))
    return ranges


def _replace_camera_resolution(text: str, camera_name: str, width: int, height: int) -> str:
    lines = text.splitlines(keepends=True)
    ranges = _camera_block_ranges(lines, camera_name)
    if not ranges:
        return text

    for start, end in ranges:
        found_width = False
        found_height = False
        for index in range(start + 1, end):
            stripped = lines[index].lstrip()
            if stripped.startswith("width="):
                lines[index] = f"{_line_indent(lines[index])}width={width},\n"
                found_width = True
            elif stripped.startswith("height="):
                lines[index] = f"{_line_indent(lines[index])}height={height},\n"
                found_height = True
        if not found_width or not found_height:
            raise RuntimeError(f"Could not find G1 {camera_name} width/height lines.")

    return "".join(lines)


def _replace_line_by_prefix(
    lines: list[str], start: int, end: int, prefix: str, replacement: str
) -> int:
    matches = []
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if stripped.startswith(prefix):
            matches.append(index)
    if not matches:
        raise RuntimeError(f"Could not find camera line starting with {prefix!r}.")

    lines[matches[0]] = f"{_line_indent(lines[matches[0]])}{replacement}\n"
    for index in reversed(matches[1:]):
        del lines[index]
    return end - (len(matches) - 1)


def _replace_or_insert_spawn_field(
    lines: list[str], start: int, end: int, field: str, value: str
) -> None:
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if stripped.startswith(f"{field}="):
            lines[index] = f"{_line_indent(lines[index])}{field}={value},\n"
            return

    for index in range(start + 1, end):
        if lines[index].lstrip().startswith("spawn=sim_utils.PinholeCameraCfg("):
            indent = _line_indent(lines[index]) + "    "
            lines.insert(index + 1, f"{indent}{field}={value},\n")
            return
    raise RuntimeError(f"Could not insert spawn field {field!r}.")


def _remove_spawn_field(lines: list[str], start: int, end: int, field: str) -> None:
    for index in range(end - 1, start, -1):
        if lines[index].lstrip().startswith(f"{field}="):
            del lines[index]


def _replace_offset_cfg(lines: list[str], start: int, end: int, geometry: CameraGeometry) -> int:
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        if not stripped.startswith("offset=TiledCameraCfg.OffsetCfg("):
            continue

        indent = _line_indent(lines[index])
        next_line = index + 1
        while next_line < end and not lines[next_line].lstrip().startswith("data_types="):
            next_line += 1
        removed = max(0, next_line - index - 1)
        if removed:
            del lines[index + 1 : next_line]
        lines[index] = (
            f"{indent}offset=TiledCameraCfg.OffsetCfg("
            f'pos={geometry.pos}, rot={geometry.rot}, convention="{geometry.convention}"),\n'
        )
        return end - removed

    raise RuntimeError("Could not find camera offset block.")


def _patch_camera_geometry_block(lines: list[str], start: int, end: int, geometry: CameraGeometry) -> None:
    end = _replace_line_by_prefix(
        lines,
        start,
        end,
        "prim_path=",
        f'prim_path="{geometry.prim_path}",',
    )
    end = _replace_offset_cfg(lines, start, end, geometry)
    _replace_or_insert_spawn_field(lines, start, end, "focal_length", geometry.focal_length)
    _replace_or_insert_spawn_field(lines, start, end, "focus_distance", geometry.focus_distance)
    _replace_or_insert_spawn_field(
        lines, start, end, "horizontal_aperture", geometry.horizontal_aperture
    )
    if geometry.vertical_aperture is None:
        _remove_spawn_field(lines, start, end, "vertical_aperture")
    else:
        _replace_or_insert_spawn_field(
            lines, start, end, "vertical_aperture", geometry.vertical_aperture
        )
    _replace_or_insert_spawn_field(lines, start, end, "clipping_range", geometry.clipping_range)
    _replace_line_by_prefix(lines, start, end, "update_period=", f"update_period={geometry.update_period},")


def _format_camera_geometry_block(geometry: CameraGeometry, indent: str) -> list[str]:
    spawn_lines = [
        f"{indent}            focal_length={geometry.focal_length},\n",
        f"{indent}            focus_distance={geometry.focus_distance},\n",
        f"{indent}            horizontal_aperture={geometry.horizontal_aperture},\n",
    ]
    if geometry.vertical_aperture is not None:
        spawn_lines.append(f"{indent}            vertical_aperture={geometry.vertical_aperture},\n")
    spawn_lines.extend(
        [
            f"{indent}            clipping_range={geometry.clipping_range},\n",
            f"{indent}            lock_camera=True\n",
        ]
    )

    return [
        f'{indent}"{geometry.camera_name}": {{\n',
        f'{indent}    "camera_cfg": TiledCameraCfg(\n',
        f'{indent}        prim_path="{geometry.prim_path}",\n',
        f"{indent}        offset=TiledCameraCfg.OffsetCfg("
        f'pos={geometry.pos}, rot={geometry.rot}, convention="{geometry.convention}"),\n',
        f'{indent}        data_types=["rgb"],\n',
        f"{indent}        spawn=sim_utils.PinholeCameraCfg(\n",
        *spawn_lines,
        f"{indent}        ),\n",
        f"{indent}        width={DEFAULT_CAMERA_WIDTH},\n",
        f"{indent}        height={DEFAULT_CAMERA_HEIGHT},\n",
        f"{indent}        update_period={geometry.update_period},\n",
        f"{indent}    ),\n",
        f'{indent}    "tags": {geometry.tags},\n',
        f'{indent}    "execute_mode": {geometry.execute_modes},\n',
        f"{indent}}},\n",
    ]


def _format_gripper_controller_class() -> list[str]:
    lines = [
        "\n\n",
        "class UnitreeG1GripperControllerDecoupledWBCEnvCfg("
        "UnitreeG1ControllerDecoupledWBCEnvCfg):\n",
        "\n",
        "    def __init__(self, enable_cameras: bool = False, "
        "initial_pose: Pose | None = None):\n",
        "        super().__init__(enable_cameras, initial_pose)\n",
        '        self.name = "G1-Gripper-Controller-DecoupledWBC"\n',
        "        self.gripper_cfg = Dex1GripperCfg()\n",
        "        self.scene_config.robot = G1_GRIPPER_CFG.replace("
        'prim_path="{ENV_REGEX_NS}/Robot")\n',
        f'        if os.environ.get("{ROBOT_COLLISION_ENV}", "true").strip().lower() '
        'in {"1", "true", "yes", "on"} and self.scene_config.robot.spawn is not None:\n',
        "            self.scene_config.robot.spawn.activate_contact_sensors = True\n",
        f'        if os.environ.get("{ROBOT_USD_PATH_ENV}", "").strip() '
        "and self.scene_config.robot.spawn is not None:\n",
        f'            self.scene_config.robot.spawn.usd_path = '
        f'os.environ["{ROBOT_USD_PATH_ENV}"].strip()\n',
        f'        if os.environ.get("{ROBOT_SELF_COLLISION_ENV}", "false").strip().lower() '
        'in {"1", "true", "yes", "on"} '
        "and self.scene_config.robot.spawn.articulation_props is not None:\n",
        "            self.scene_config.robot.spawn.articulation_props.enabled_self_collisions = True\n",
        "        self.scene_config.left_gripper_contact = ContactSensorCfg(\n",
        "            prim_path=f\"{{ENV_REGEX_NS}}/Robot/"
        "{self.gripper_cfg.left_contact_body_name}\",\n",
        "            update_period=0.0,\n",
        "            history_length=1,\n",
        "            force_threshold=0.1,\n",
        "            debug_vis=False,\n",
        "            filter_prim_paths_expr=[],\n",
        "        )\n",
        "        self.scene_config.right_gripper_contact = ContactSensorCfg(\n",
        "            prim_path=f\"{{ENV_REGEX_NS}}/Robot/"
        "{self.gripper_cfg.right_contact_body_name}\",\n",
        "            update_period=0.0,\n",
        "            history_length=1,\n",
        "            force_threshold=0.1,\n",
        "            debug_vis=False,\n",
        "            filter_prim_paths_expr=[],\n",
        "        )\n",
        "        self.observation_cameras: dict = {\n",
    ]
    for geometry in POLICY_CAMERA_GEOMETRIES:
        lines.extend(_format_camera_geometry_block(geometry, "            "))
    lines.extend(
        [
            "        }\n",
            "        self.action_config = G1DecoupledWBCActionsCfg()\n",
            "        self.action_config.left_hand_action = "
            "self.gripper_cfg.left_hand_action_cfg()[self.hand_action_mode]\n",
            "        self.action_config.right_hand_action = "
            "self.gripper_cfg.right_hand_action_cfg()[self.hand_action_mode]\n",
        ]
    )
    return lines


def _ensure_gripper_controller_text(text: str) -> str:
    """Add the V1 Dex1/WBC embodiment without modifying organizer controllers."""

    lines = text.splitlines(keepends=True)
    if any(
        line.startswith("class UnitreeG1GripperControllerDecoupledWBCEnvCfg")
        for line in lines
    ):
        return text

    text = _ensure_v1_runtime_api_text(text)
    lines = text.splitlines(keepends=True)
    changed = False
    if not any("from robofinals.core.models.grippers.dex1 import Dex1GripperCfg" in line for line in lines):
        for index, line in enumerate(lines):
            if line.startswith("import robofinals.core.mdp as mdp"):
                lines[index:index] = [
                    "from robofinals.core.models.grippers.dex1 import Dex1GripperCfg\n"
                ]
                changed = True
                break
        else:
            raise RuntimeError("Could not insert Dex1GripperCfg import in g1.py")

    if not any("G1_GRIPPER_CFG" in line and "from .assets_cfg import" in line for line in lines):
        for index, line in enumerate(lines):
            if line.startswith("from .assets_cfg import ") and "G1_GEARWBC_CFG" in line:
                lines[index] = line.replace(
                    "G1_GEARWBC_CFG,",
                    "G1_GEARWBC_CFG, G1_GRIPPER_CFG,",
                    1,
                )
                changed = True
                break
        else:
            raise RuntimeError("Could not add G1_GRIPPER_CFG import in g1.py")

    if not any(
        line.startswith("class UnitreeG1GripperControllerDecoupledWBCEnvCfg")
        for line in lines
    ):
        parent_range = _class_range(lines, "UnitreeG1ControllerDecoupledWBCEnvCfg")
        if parent_range is None:
            raise RuntimeError("Could not find UnitreeG1ControllerDecoupledWBCEnvCfg in g1.py")
        _start, end = parent_range
        lines[end:end] = _format_gripper_controller_class()
        changed = True
    return "".join(lines) if changed else text


def patch_g1_gripper_controller(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "core" / "robots" / "unitree" / "g1.py"
    if not target.is_file():
        raise FileNotFoundError(f"G1 config not found: {target}")
    current = target.read_text(encoding="utf-8")
    updated = _ensure_gripper_controller_text(current)
    if updated == current:
        return False
    target.write_text(updated, encoding="utf-8")
    print(f"[flip_table] added deterministic V1 Dex1/WBC embodiment: {target}", flush=True)
    return True


def _observation_cameras_close_index(lines: list[str], class_start: int, class_end: int) -> int | None:
    for index in range(class_start, class_end):
        if "self.observation_cameras" not in lines[index] or "{" not in lines[index]:
            continue
        indent = _line_indent(lines[index])
        for close_index in range(index + 1, class_end):
            if _line_indent(lines[close_index]) == indent and lines[close_index].strip() in {"}", "},"}:
                return close_index
        return None
    return None


def _insert_camera_geometry_block(
    lines: list[str], class_start: int, class_end: int, geometry: CameraGeometry
) -> None:
    close_index = _observation_cameras_close_index(lines, class_start, class_end)
    if close_index is None:
        raise RuntimeError(f"Could not find observation_cameras dict for {geometry.camera_name}.")
    camera_indent = _line_indent(lines[close_index]) + "    "
    lines[close_index:close_index] = _format_camera_geometry_block(geometry, camera_indent)


def _patch_camera_geometry(text: str, geometry: CameraGeometry) -> str:
    lines = text.splitlines(keepends=True)

    if geometry.class_names is None:
        ranges = _camera_block_ranges(lines, geometry.camera_name)
        if not ranges:
            raise RuntimeError(f"Could not find {geometry.camera_name} in all classes.")
        for start, end in reversed(ranges):
            _patch_camera_geometry_block(lines, start, end, geometry)
    else:
        for class_name in geometry.class_names:
            class_scope = _class_range(lines, class_name)
            if class_scope is None:
                continue
            ranges = _camera_block_ranges(lines, geometry.camera_name, *class_scope)
            if not ranges and geometry.insert_if_missing:
                _insert_camera_geometry_block(lines, *class_scope, geometry)
                continue
            if not ranges:
                raise RuntimeError(f"Could not find {geometry.camera_name} in {class_name}.")
            for start, end in reversed(ranges):
                _patch_camera_geometry_block(lines, start, end, geometry)

    return "".join(lines)


def _add_camera_execute_mode(
    text: str,
    camera_name: str,
    class_names: tuple[str, ...],
    mode: str,
) -> str:
    """Add an execution mode to existing camera declarations idempotently."""

    lines = text.splitlines(keepends=True)
    for class_name in class_names:
        class_scope = _class_range(lines, class_name)
        if class_scope is None:
            continue
        for start, end in reversed(_camera_block_ranges(lines, camera_name, *class_scope)):
            for index in range(start, end):
                stripped = lines[index].lstrip()
                if not stripped.startswith('"execute_mode":'):
                    continue
                if mode in stripped:
                    break
                close = lines[index].rfind("]")
                if close < 0:
                    raise RuntimeError(f"Malformed execute_mode for {class_name}.{camera_name}")
                prefix = lines[index][:close].rstrip()
                separator = "" if prefix.endswith("[") else ", "
                lines[index] = prefix + separator + mode + lines[index][close:]
                break
    return "".join(lines)


def _remove_existing_gripper_variants(lines: list[str]) -> None:
    prefix = "G1_GRIPPER_CFG.spawn.variants"
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].lstrip().startswith(prefix):
            continue

        end = index + 1
        if "{" in lines[index] and "}" not in lines[index]:
            while end < len(lines) and "}" not in lines[end]:
                end += 1
            end = min(end + 1, len(lines))
        del lines[index:end]


def _format_gripper_variants_assignment(indent: str) -> list[str]:
    lines = [f"{indent}G1_GRIPPER_CFG.spawn.variants = {{\n"]
    for name, value in GRIPPER_VARIANTS.items():
        lines.append(f'{indent}    "{name}": "{value}",\n')
    lines.append(f"{indent}}}\n")
    return lines


def _patch_gripper_asset_variants_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    _remove_existing_gripper_variants(lines)

    target_prefix = "G1_GRIPPER_CFG.spawn.usd_path"
    for index, line in enumerate(lines):
        if not line.lstrip().startswith(target_prefix):
            continue
        indent = _line_indent(line)
        insert_index = index + 1
        lines[insert_index:insert_index] = _format_gripper_variants_assignment(indent)
        return "".join(lines)

    raise RuntimeError("Could not find G1_GRIPPER_CFG.spawn.usd_path in assets_cfg.py.")


def _format_upper_body_joint_action_class() -> list[str]:
    return [
        "\n",
        "\n",
        "@configclass\n",
        f"class {UPPER_BODY_JOINT_ACTION_CLASS}:\n",
        '    """19-D absolute upper-body joint action for Team RAMEN policy evaluation."""\n',
        "\n",
        "    waist_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(\n",
        '        asset_name="robot",\n',
        '        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],\n',
        "        scale=1.0,\n",
        "        use_default_offset=False,\n",
        "        preserve_order=True,\n",
        "    )\n",
        "    left_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(\n",
        '        asset_name="robot",\n',
        "        joint_names=[\n",
        '            "left_shoulder_pitch_joint",\n',
        '            "left_shoulder_roll_joint",\n',
        '            "left_shoulder_yaw_joint",\n',
        '            "left_elbow_joint",\n',
        '            "left_wrist_roll_joint",\n',
        '            "left_wrist_pitch_joint",\n',
        '            "left_wrist_yaw_joint",\n',
        "        ],\n",
        "        scale=1.0,\n",
        "        use_default_offset=False,\n",
        "        preserve_order=True,\n",
        "    )\n",
        "    right_arm_action: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(\n",
        '        asset_name="robot",\n',
        "        joint_names=[\n",
        '            "right_shoulder_pitch_joint",\n',
        '            "right_shoulder_roll_joint",\n',
        '            "right_shoulder_yaw_joint",\n',
        '            "right_elbow_joint",\n',
        '            "right_wrist_roll_joint",\n',
        '            "right_wrist_pitch_joint",\n',
        '            "right_wrist_yaw_joint",\n',
        "        ],\n",
        "        scale=1.0,\n",
        "        use_default_offset=False,\n",
        "        preserve_order=True,\n",
        "    )\n",
        "    left_hand_action: mdp.ActionTermCfg = None\n",
        "    right_hand_action: mdp.ActionTermCfg = None\n",
        "\n",
        "\n",
        "@configclass\n",
        f"class {PINK_EEF_ACTION_CLASS}:\n",
        '    """Absolute bimanual EEF actions solved by the organizer V1 PINK controller."""\n',
        "\n",
        "    arms_action: G1ActionCfg = G1ActionCfg(\n",
        '        asset_name="robot",\n',
        "        joint_names=[\n",
        '            "left_shoulder_pitch_joint",\n',
        '            "left_shoulder_roll_joint",\n',
        '            "left_shoulder_yaw_joint",\n',
        '            "left_elbow_joint",\n',
        '            "left_wrist_roll_joint",\n',
        '            "left_wrist_pitch_joint",\n',
        '            "left_wrist_yaw_joint",\n',
        '            "right_shoulder_pitch_joint",\n',
        '            "right_shoulder_roll_joint",\n',
        '            "right_shoulder_yaw_joint",\n',
        '            "right_elbow_joint",\n',
        '            "right_wrist_roll_joint",\n',
        '            "right_wrist_pitch_joint",\n',
        '            "right_wrist_yaw_joint",\n',
        "        ],\n",
        "    )\n",
        "    left_hand_action: mdp.ActionTermCfg = None\n",
        "    right_hand_action: mdp.ActionTermCfg = None\n",
    ]


def _insert_upper_body_joint_action_class(lines: list[str]) -> bool:
    if any(line.startswith(f"class {UPPER_BODY_JOINT_ACTION_CLASS}") for line in lines):
        return False
    for target_class in ("G1CameraCfg", "UnitreeG1GripperControllerDecoupledWBCEnvCfg"):
        for index, line in enumerate(lines):
            if not line.startswith(f"class {target_class}"):
                continue
            insert_index = index
            if index > 0 and lines[index - 1].strip() == "@configclass":
                insert_index = index - 1
            lines[insert_index:insert_index] = _format_upper_body_joint_action_class()
            return True
    lines.extend(_format_upper_body_joint_action_class())
    return True


def _repair_configclass_decorators(lines: list[str]) -> bool:
    changed = False

    index = 0
    while index < len(lines):
        if lines[index].strip() != "@configclass":
            index += 1
            continue
        next_index = index + 1
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if next_index < len(lines) and lines[next_index].strip() == "@configclass":
            del lines[index:next_index]
            changed = True
            continue
        index += 1

    for index, line in enumerate(lines):
        if not line.startswith("class G1CameraCfg"):
            continue
        prev_index = index - 1
        while prev_index >= 0 and not lines[prev_index].strip():
            prev_index -= 1
        if prev_index < 0 or lines[prev_index].strip() != "@configclass":
            lines[index:index] = ["@configclass\n"]
            changed = True
        break
    return changed


def _patch_gripper_controller_action_config(lines: list[str]) -> bool:
    class_scope = _class_range(lines, "UnitreeG1GripperControllerDecoupledWBCEnvCfg")
    if class_scope is None:
        raise RuntimeError("Could not find UnitreeG1GripperControllerDecoupledWBCEnvCfg in g1.py.")
    start, end = class_scope
    marker = PINK_EEF_ACTION_ENV
    for index in range(start, end):
        if marker in lines[index]:
            return False

    target = "self.action_config = G1DecoupledWBCActionsCfg()"
    for index in range(start, end):
        if lines[index].strip() != target:
            continue
        indent = _line_indent(lines[index])
        lines[index : index + 1] = [
            f'{indent}if os.environ.get("{UPPER_BODY_JOINT_ACTION_ENV}", "").strip().lower() '
            'in {"1", "true", "yes", "on"}:\n',
            f"{indent}    self.action_config = {UPPER_BODY_JOINT_ACTION_CLASS}()\n",
            f'{indent}elif os.environ.get("{PINK_EEF_ACTION_ENV}", "").strip().lower() '
            'in {"1", "true", "yes", "on"}:\n',
            f"{indent}    self.action_config = {PINK_EEF_ACTION_CLASS}()\n",
            f"{indent}else:\n",
            f"{indent}    self.action_config = G1DecoupledWBCActionsCfg()\n",
        ]
        return True
    return False


def _patch_gripper_controller_root_lock(lines: list[str]) -> bool:
    """Use the simulator's native fixed-root articulation for upper-body eval.

    Rewriting a floating-base pose every control step creates contact impulses
    when the feet or workbench are close to the robot.  The evaluation contract
    already fixes the lower body, so let PhysX enforce the root constraint too.
    """
    class_scope = _class_range(lines, "UnitreeG1GripperControllerDecoupledWBCEnvCfg")
    if class_scope is None:
        raise RuntimeError("Could not find UnitreeG1GripperControllerDecoupledWBCEnvCfg in g1.py.")
    start, end = class_scope
    marker = "FLIP_TABLE_FIX_ROOT_LINK"
    if any(marker in lines[index] for index in range(start, end)):
        # A previous development build used the wrong config path.  Repair it
        # in place so re-running the container overlay is deterministic even
        # when the container's mutable g1.py already contains that patch.
        changed = False
        for index in range(start, end):
            if "self.scene_config.robot.articulation_props.fix_root_link" in lines[index]:
                lines[index] = lines[index].replace(
                    "self.scene_config.robot.articulation_props.fix_root_link",
                    "self.scene_config.robot.spawn.articulation_props.fix_root_link",
                )
                changed = True
        return changed

    target = 'self.scene_config.robot = G1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")'
    for index in range(start, end):
        if lines[index].strip() != target:
            continue
        indent = _line_indent(lines[index])
        lines[index + 1:index + 1] = [
            f'{indent}if os.environ.get("{marker}", "").strip().lower() '
            'in {"1", "true", "yes", "on"} and self.scene_config.robot.spawn.articulation_props is not None:\n',
            f"{indent}    self.scene_config.robot.spawn.articulation_props.fix_root_link = True\n",
        ]
        return True
    # Older test doubles and non-PhysX controller variants may not expose the
    # robot assignment in this class.  They can still use the camera/action
    # patch; the production V1 class has the assignment and is patched above.
    return False


def _patch_g1_contact_sensor_fields(lines: list[str]) -> bool:
    """Declare four finger sensors plus four optional leg-side sensors."""

    class_scope = _class_range(lines, "G1SceneCfg")
    if class_scope is None:
        return False
    start, end = class_scope
    indent = _line_indent(lines[start]) + "    "
    desired = {
        "left_gripper_contact": (
            f"{indent}left_gripper_contact: ContactSensorCfg | None = None\n"
        ),
        "right_gripper_contact": (
            f"{indent}right_gripper_contact: ContactSensorCfg | None = None\n"
        ),
        "left_gripper_contact_2": (
            f'{indent}left_gripper_contact_2: ContactSensorCfg | None = '
            'ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/left_dex1_finger_link_2", '
            'update_period=0.0, history_length=1, force_threshold=0.1, debug_vis=False, '
            'filter_prim_paths_expr=[])\n'
        ),
        "right_gripper_contact_2": (
            f'{indent}right_gripper_contact_2: ContactSensorCfg | None = '
            'ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/right_dex1_finger_link_2", '
            'update_period=0.0, history_length=1, force_threshold=0.1, debug_vis=False, '
            'filter_prim_paths_expr=[])\n'
        ),
        **{
            f"white_leg_contact_{index}": (
                f"{indent}white_leg_contact_{index}: ContactSensorCfg | None = None\n"
            )
            for index in range(4)
        },
    }
    changed = False
    present = set()
    for index in range(start + 1, end):
        stripped = lines[index].lstrip()
        for field, replacement in desired.items():
            if not stripped.startswith(f"{field}:"):
                continue
            present.add(field)
            if lines[index] != replacement:
                lines[index] = replacement
                changed = True
            break
    missing = [replacement for field, replacement in desired.items() if field not in present]
    if missing:
        lines[start + 1:start + 1] = missing
        changed = True
    return changed


def _patch_gripper_contact_sensor_thresholds(lines: list[str]) -> bool:
    """Set the same low diagnostic threshold on both primary finger sensors."""

    class_scope = _class_range(lines, "UnitreeG1GripperControllerDecoupledWBCEnvCfg")
    if class_scope is None:
        return False
    start, end = class_scope
    changed = False
    for field in ("left_gripper_contact", "right_gripper_contact"):
        assignment = f"self.scene_config.{field} = ContactSensorCfg("
        config_start = next(
            (index for index in range(start, end) if lines[index].strip() == assignment),
            None,
        )
        if config_start is None:
            continue
        config_end = next(
            (index for index in range(config_start + 1, end) if lines[index].strip() == ")"),
            None,
        )
        if config_end is None:
            raise RuntimeError(f"unterminated ContactSensorCfg for {field}")
        threshold_index = next(
            (
                index
                for index in range(config_start + 1, config_end)
                if lines[index].strip().startswith("force_threshold=")
            ),
            None,
        )
        if threshold_index is not None:
            replacement = f"{_line_indent(lines[threshold_index])}force_threshold=0.1,\n"
            if lines[threshold_index] != replacement:
                lines[threshold_index] = replacement
                changed = True
            continue
        history_index = next(
            (
                index
                for index in range(config_start + 1, config_end)
                if lines[index].strip().startswith("history_length=")
            ),
            None,
        )
        if history_index is None:
            raise RuntimeError(f"history_length is missing from {field}")
        lines[history_index + 1:history_index + 1] = [
            f"{_line_indent(lines[history_index])}force_threshold=0.1,\n"
        ]
        end += 1
        changed = True
    return changed


def _remove_unsupported_shape_contact_filters(lines: list[str]) -> bool:
    """Keep finger sensors unfiltered; RL config adds reverse leg filters.

    PhysX GPU filtering accepts rigid-body paths, not collision-shape paths.
    Organizer legs contain 123 shapes each, so filtering a finger sensor by
    ``Leg001_Collider118`` is unsupported and silently produces zero matrices.
    The RL config instead senses each leg body against four single-shape finger
    bodies. Finger ``net_forces_w`` remains the all-surface safety signal.
    """

    changed = False
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(f"{LEGACY_WHITE_TABLE_CONTACT_FILTER_NAME} =")
        ),
        None,
    )
    if start is not None:
        end = start + 1
        while end < len(lines) and lines[end].strip() != ")":
            end += 1
        if end >= len(lines):
            raise RuntimeError("unterminated legacy white-table contact-filter tuple")
        end += 1
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
        changed = True

    class_scope = _class_range(lines, "UnitreeG1GripperControllerDecoupledWBCEnvCfg")
    if class_scope is not None:
        class_start, class_end = class_scope
        for index in range(class_start, class_end):
            if "filter_prim_paths_expr=" not in lines[index]:
                continue
            replacement = f"{_line_indent(lines[index])}filter_prim_paths_expr=[],\n"
            if lines[index] != replacement:
                lines[index] = replacement
                changed = True
    return changed


def _patch_gripper_collision_properties(lines: list[str]) -> bool:
    """Enable contact reporting while preserving the authored collision asset."""
    if any(ROBOT_USD_PATH_ENV in line for line in lines):
        return False
    class_scope = _class_range(lines, "UnitreeG1GripperControllerDecoupledWBCEnvCfg")
    if class_scope is None:
        return False
    start, end = class_scope
    target = 'self.scene_config.robot = G1_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")'
    for index in range(start, end):
        if lines[index].strip() != target:
            continue
        indent = _line_indent(lines[index])
        lines[index + 1:index + 1] = [
            f'{indent}if os.environ.get("{ROBOT_COLLISION_ENV}", "true").strip().lower() '
            'in {"1", "true", "yes", "on"} and self.scene_config.robot.spawn is not None:\n',
            f'{indent}    self.scene_config.robot.spawn.activate_contact_sensors = True\n',
            f'{indent}if os.environ.get("{ROBOT_USD_PATH_ENV}", "").strip() and self.scene_config.robot.spawn is not None:\n',
            f'{indent}    self.scene_config.robot.spawn.usd_path = os.environ["{ROBOT_USD_PATH_ENV}"].strip()\n',
            f'{indent}if os.environ.get("{ROBOT_SELF_COLLISION_ENV}", "false").strip().lower() '
            'in {"1", "true", "yes", "on"} and self.scene_config.robot.spawn.articulation_props is not None:\n',
            f'{indent}    self.scene_config.robot.spawn.articulation_props.enabled_self_collisions = True\n',
        ]
        return True
    return False


def _patch_collision_filter_config(lines: list[str]) -> bool:
    """Expose the official environment collision-filter switch for diagnosis."""
    marker = f'os.environ.get("{DISABLE_COLLISION_FILTER_ENV}"'
    if any(marker in line for line in lines):
        return False
    class_scope = _class_range(lines, "UnitreeG1ControllerEnvCfg")
    if class_scope is None:
        return False
    start, end = class_scope
    target = "        env_cfg.sim.gravity = (0.0, 0.0, -9.81)\n"
    for index in range(start, end):
        if lines[index] != target:
            continue
        lines[index + 1:index + 1] = [
            f'        if os.environ.get("{DISABLE_COLLISION_FILTER_ENV}", "false").strip().lower() '
            'in {"1", "true", "yes", "on"}:\n',
            "            env_cfg.scene.filter_collisions = False\n",
        ]
        return True
    return False


def _patch_upper_body_joint_action_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    changed = _repair_configclass_decorators(lines)
    changed = _patch_g1_contact_sensor_fields(lines) or changed
    changed = _patch_gripper_contact_sensor_thresholds(lines) or changed
    changed = _remove_unsupported_shape_contact_filters(lines) or changed
    changed = _insert_upper_body_joint_action_class(lines) or changed
    changed = _patch_gripper_controller_action_config(lines) or changed
    changed = _patch_gripper_controller_root_lock(lines) or changed
    changed = _patch_gripper_collision_properties(lines) or changed
    changed = _patch_collision_filter_config(lines) or changed
    changed = _repair_configclass_decorators(lines) or changed
    return "".join(lines) if changed else text


def patch_g1_upper_body_joint_action(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "core" / "robots" / "unitree" / "g1.py"
    if not target.exists():
        raise FileNotFoundError(f"G1 config not found: {target}")

    current = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(current, encoding="utf-8")

    updated = _patch_upper_body_joint_action_text(current)
    if updated == current:
        print(
            "[flip_table] G1 ACT upper-body joint action switch already available: "
            f"{target}",
            flush=True,
        )
        return False

    target.write_text(updated, encoding="utf-8")
    print(
        "[flip_table] added G1 ACT upper-body joint action switch "
        f"({UPPER_BODY_JOINT_ACTION_ENV}): {target}",
        flush=True,
    )
    return True


def patch_g1_gripper_asset_variants(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "core" / "robots" / "unitree" / "assets_cfg.py"
    if not target.exists():
        raise FileNotFoundError(f"G1 assets config not found: {target}")

    current = target.read_text(encoding="utf-8")
    backup = Path(str(target) + ASSETS_BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(current, encoding="utf-8")

    updated = _patch_gripper_asset_variants_text(current)
    if updated == current:
        print(
            "[flip_table] G1_GRIPPER USD variants already selected: "
            f"{GRIPPER_VARIANTS}: {target}",
            flush=True,
        )
        return False

    target.write_text(updated, encoding="utf-8")
    print(
        f"[flip_table] selected generated G1_GRIPPER USD variants "
        f"{GRIPPER_VARIANTS}: {target}",
        flush=True,
    )
    return True


def patch_g1_gripper_material_binding_api(robofinals_root: Path) -> bool:
    """Make RTX honor the generated G1 and Dex1 material bindings.

    RoboFinals V1's URDF-converted base layer has valid ``material:binding``
    relationships, but the 39 owning visual prims are missing
    ``UsdShade.MaterialBindingAPI``. Unitree's official Isaac Lab G1/Dex1 USD
    applies that schema. Add only the missing schema and preserve the existing
    materials, targets, meshes, and physics.
    """

    target = (
        robofinals_root
        / "robofinals"
        / "data"
        / "assets"
        / "g1_urdf_gripper"
        / "configuration"
        / "usd_base.usd"
    )
    if not target.exists():
        raise FileNotFoundError(f"G1 gripper base USD not found: {target}")

    backup = Path(str(target) + MATERIAL_BINDING_BACKUP_SUFFIX)
    if not backup.exists():
        import shutil

        shutil.copy2(target, backup)

    try:
        from pxr import Usd, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "pxr is required to repair G1_GRIPPER material bindings; run this "
            "patch with the RoboFinals conda Python."
        ) from exc

    stage = Usd.Stage.Open(str(target))
    if stage is None:
        raise RuntimeError(f"Could not open G1 gripper base USD: {target}")

    repaired_paths: list[str] = []
    for prim in stage.TraverseAll():
        binding = prim.GetRelationship("material:binding")
        if not binding or not binding.GetTargets():
            continue
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim)
        repaired_paths.append(str(prim.GetPath()))

    if not repaired_paths:
        print(
            f"[flip_table] G1_GRIPPER material bindings already valid: {target}",
            flush=True,
        )
        return False

    stage.GetRootLayer().Save()
    print(
        "[flip_table] repaired G1_GRIPPER material bindings "
        f"on {len(repaired_paths)} visual prims: {target}",
        flush=True,
    )
    return True


def patch_g1_gripper_unitree_material_values(robofinals_root: Path) -> bool:
    """Apply Unitree's official G1/Dex1 OmniPBR appearance to RoboFinals."""

    target = (
        robofinals_root
        / "robofinals"
        / "data"
        / "assets"
        / "g1_urdf_gripper"
        / "configuration"
        / "usd_base.usd"
    )
    if not target.exists():
        raise FileNotFoundError(f"G1 gripper base USD not found: {target}")

    backup = Path(str(target) + MATERIAL_BINDING_BACKUP_SUFFIX)
    if not backup.exists():
        import shutil

        shutil.copy2(target, backup)

    try:
        from pxr import Gf, Sdf, Usd, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "pxr is required to apply Unitree G1 material values; run this "
            "patch with the RoboFinals conda Python."
        ) from exc

    stage = Usd.Stage.Open(str(target))
    if stage is None:
        raise RuntimeError(f"Could not open G1 gripper base USD: {target}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim or not default_prim.IsValid():
        raise RuntimeError(f"G1 gripper base USD has no default prim: {target}")

    def values_match(current, expected) -> bool:
        try:
            return all(abs(float(a) - float(b)) <= 1.0e-6 for a, b in zip(current, expected))
        except TypeError:
            return current is not None and abs(float(current) - float(expected)) <= 1.0e-6

    changed = False
    for name, (color, metallic, roughness, specular) in UNITREE_G1_PBR_MATERIALS.items():
        shader_prim = stage.GetPrimAtPath(f"{default_prim.GetPath()}/Looks/{name}/Shader")
        if not shader_prim or not shader_prim.IsValid():
            raise RuntimeError(f"G1 gripper material shader is missing: {name}")
        shader = UsdShade.Shader(shader_prim)
        values = (
            ("diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*color)),
            ("metallic_constant", Sdf.ValueTypeNames.Float, metallic),
            ("reflection_roughness_constant", Sdf.ValueTypeNames.Float, roughness),
            ("specular_level", Sdf.ValueTypeNames.Float, specular),
        )
        for input_name, value_type, expected in values:
            shader_input = shader.CreateInput(input_name, value_type)
            if not values_match(shader_input.Get(), expected):
                shader_input.Set(expected)
                changed = True

    if not changed:
        print(
            f"[flip_table] official Unitree G1/Dex1 material values already active: {target}",
            flush=True,
        )
        return False

    stage.GetRootLayer().Save()
    print(
        f"[flip_table] applied official Unitree G1/Dex1 OmniPBR material values: {target}",
        flush=True,
    )
    return True


def patch_g1_gripper_contact_material(robofinals_root: Path) -> bool:
    """Pre-bind the nominal physics material to every Dex1 hand collider.

    This baseline is required even when per-episode contact randomization is
    disabled. The runtime randomizer only changes the authored coefficients;
    it must not decide whether the hand has a physics material at all.
    """

    target = (
        robofinals_root
        / "robofinals"
        / "data"
        / "assets"
        / "g1_urdf_gripper"
        / "G1_GRIPPER.usd"
    )
    if not target.exists():
        raise FileNotFoundError(f"G1 gripper USD not found: {target}")
    backup = Path(str(target) + CONTACT_MATERIAL_BACKUP_SUFFIX)
    if not backup.exists():
        import shutil

        shutil.copy2(target, backup)

    try:
        from pxr import Sdf, Usd, UsdPhysics, UsdShade
    except ImportError as exc:
        raise RuntimeError(
            "pxr is required to pre-bind the G1 contact material; run this "
            "patch with the RoboFinals conda Python."
        ) from exc

    stage = Usd.Stage.Open(str(target), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open G1 gripper USD: {target}")
    root = stage.GetDefaultPrim()
    if not root or not root.IsValid():
        raise RuntimeError(f"G1 gripper USD has no default prim: {target}")

    material = UsdShade.Material.Define(stage, f"{root.GetPath()}/Looks/flip_table_contact_hand")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(0.875)
    material_api.CreateDynamicFrictionAttr().Set(0.675)
    material_api.CreateRestitutionAttr().Set(0.07)
    material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    material.GetPrim().CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("average")
    material.GetPrim().CreateAttribute(
        "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("average")

    suffixes = (
        "left_wrist_yaw_link/collisions",
        "right_wrist_yaw_link/collisions",
        "left_dex1_finger_link_1/collisions",
        "left_dex1_finger_link_2/collisions",
        "right_dex1_finger_link_1/collisions",
        "right_dex1_finger_link_2/collisions",
    )
    bound = []
    for suffix in suffixes:
        prim = stage.GetPrimAtPath(f"{root.GetPath()}/{suffix}")
        if not prim or not prim.IsValid():
            raise RuntimeError(f"G1 hand collision scope is missing: {suffix}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
        bound.append(str(prim.GetPath()))

    stage.GetRootLayer().Save()
    print(
        f"[flip_table] pre-bound Dex1 contact material to {len(bound)} collision scopes: {target}",
        flush=True,
    )
    return True


def patch_g1_gripper_physx_collisions(robofinals_root: Path) -> bool:
    """Author the Dex1 contact envelope on the non-instanced source colliders.

    Applying ``CollisionPropertiesCfg`` at the articulation root cannot edit
    the generated USD's instance proxies.  The four source prims below are the
    exact official Unitree collision meshes referenced by both hands, so this
    changes only PhysX contact generation and leaves their geometry untouched.
    """

    target = (
        robofinals_root
        / "robofinals"
        / "data"
        / "assets"
        / "g1_urdf_gripper"
        / "configuration"
        / "usd_physics.usd"
    )
    if not target.is_file():
        raise FileNotFoundError(f"G1 gripper physics USD not found: {target}")
    backup = Path(str(target) + PHYSX_COLLISION_BACKUP_SUFFIX)
    if not backup.exists():
        import shutil

        shutil.copy2(target, backup)

    try:
        from pxr import Sdf, Usd, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "pxr PhysX schemas are required to configure Dex1 collision contact"
        ) from exc

    stage = Usd.Stage.Open(str(target), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Could not open G1 gripper physics USD: {target}")

    paths = tuple(
        f"/colliders/{side}_dex1_finger_link_{finger}/dex1_col_{finger}/node_STL_BINARY_"
        for side in ("left", "right")
        for finger in (1, 2)
    )
    changed = False
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"official Dex1 source collider is missing: {path}")
        prim.AddAppliedSchema("PhysxCollisionAPI")
        contact_attr = prim.CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        )
        rest_attr = prim.CreateAttribute(
            "physxCollision:restOffset", Sdf.ValueTypeNames.Float
        )
        if contact_attr.Get() != 0.002:
            contact_attr.Set(0.002)
            changed = True
        if rest_attr.Get() != 0.0:
            rest_attr.Set(0.0)
            changed = True

    if changed:
        stage.GetRootLayer().Save()
        status = "configured"
    else:
        status = "already configured"
    print(
        f"[flip_table] {status} PhysX contact offsets on {len(paths)} official Dex1 colliders: "
        f"contact_offset=0.002, rest_offset=0.0, asset={target}",
        flush=True,
    )
    return changed


def patch_g1_cameras(robofinals_root: Path) -> bool:
    target = robofinals_root / "robofinals" / "core" / "robots" / "unitree" / "g1.py"
    if not target.exists():
        raise FileNotFoundError(f"G1 config not found: {target}")

    width = _env_int("FLIP_TABLE_CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH)
    height = _env_int("FLIP_TABLE_CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT)
    if (width, height) != (DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT):
        raise ValueError(
            "policy and recording cameras must remain 640x480 to match the real dataset; "
            f"got {width}x{height}"
        )
    camera_names = _env_camera_names("FLIP_TABLE_CAMERA_RESOLUTION_NAMES", DEFAULT_CAMERA_NAMES)
    normalize_policy_cameras = _env_bool("FLIP_TABLE_NORMALIZE_G1_POLICY_CAMERAS", True)

    current = target.read_text(encoding="utf-8")
    backup = Path(str(target) + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(current, encoding="utf-8")

    updated = current
    if normalize_policy_cameras:
        for geometry in POLICY_CAMERA_GEOMETRIES:
            updated = _patch_camera_geometry(updated, _geometry_with_env_overrides(geometry))
    if _env_bool(RL_CAMERA_ENV, False):
        for camera_name in (
            "first_person_camera",
            "head_right_camera",
            "left_hand_camera",
            "right_hand_camera",
        ):
            updated = _add_camera_execute_mode(
                updated,
                camera_name,
                GRIPPER_CFG_CLASSES,
                "ExecuteMode.TRAIN",
            )
    for camera_name in camera_names:
        updated = _replace_camera_resolution(updated, camera_name, width, height)

    if updated == current:
        print(
            "[flip_table] G1 camera config already normalized "
            f"with resolution={width}x{height}: {target}",
            flush=True,
        )
        return False

    target.write_text(updated, encoding="utf-8")
    geometry_msg = (
        "normalized policy camera geometry/intrinsics and "
        if normalize_policy_cameras
        else ""
    )
    print(
        f"[flip_table] {geometry_msg}set resolution={width}x{height} "
        f"for {','.join(camera_names)}: {target}",
        flush=True,
    )
    return True


def patch_camera_config_field_generation(robofinals_root: Path) -> bool:
    """Expose all selected V1 cameras as dataclass fields.

    Isaac Lab discovers sensors by iterating declared fields, not attributes
    added with ``setattr``.  The organizer registry uses the latter, which
    silently drops the added head-right stereo camera.
    """

    target = robofinals_root / "robofinals" / "core" / "robots" / "robot_arena_base.py"
    if not target.is_file():
        # Keep the camera/asset patch unit-testable against a minimal G1 tree.
        # Every supported RoboFinals V1 image contains this registry.
        print(f"[flip_table] camera registry unavailable; stereo registry patch skipped: {target}", flush=True)
        return False

    current = target.read_text(encoding="utf-8")
    if CAMERA_CONFIG_FIELDS_MARKER in current:
        return False

    import_line = "from isaaclab_arena.utils.configclass import combine_configclass_instances\n"
    replacement_import = (
        "from isaaclab_arena.utils.configclass import "
        "combine_configclass_instances, make_configclass\n"
    )
    if import_line not in current:
        raise RuntimeError("Could not find RoboFinals V1 configclass import for camera registry patch")

    old_method = '''    def _setup_camera_config(self, task_type):
        for cam_name, cam_info in self.observation_cameras.items():
            if self.context.execute_mode in cam_info["execute_mode"]:
                cam_info = self.observation_cameras[cam_name]
                setattr(self.camera_config, cam_name, cam_info["camera_cfg"])
        if self.enable_cameras and self.camera_config and self.add_camera_to_observation:
            camera_observation_config = make_camera_observation_cfg(self.camera_config)
            camera_observations = camera_observation_config.camera_obs
            for field_name in camera_observations.__dataclass_fields__:
                field = camera_observations.__dataclass_fields__[field_name]
                if isinstance(field.type, type) and issubclass(field.type, ObsTerm):
                    self.active_observation_camera_names.append(field_name)
            self.policy_observation_config = combine_configclass_instances(
                "PolicyObservationCfg",
                camera_observations,
                self.policy_observation_config,
            )
'''
    new_method = '''    def _setup_camera_config(self, task_type):
        # FLIP_TABLE_CAMERA_CONFIG_FIELDS_V1
        # Build declared fields so the right stereo eye is a real sensor.
        camera_fields = []
        for cam_name, cam_info in self.observation_cameras.items():
            if self.context.execute_mode in cam_info["execute_mode"]:
                camera_cfg = cam_info["camera_cfg"]
                camera_fields.append((cam_name, type(camera_cfg), camera_cfg))
        self.camera_config = (
            make_configclass(f"{type(self).__name__}CameraCfg", camera_fields)()
            if camera_fields
            else None
        )
        if self.enable_cameras and self.camera_config and self.add_camera_to_observation:
            camera_observation_config = make_camera_observation_cfg(self.camera_config)
            camera_observations = camera_observation_config.camera_obs
            for field_name in camera_observations.__dataclass_fields__:
                field = camera_observations.__dataclass_fields__[field_name]
                if isinstance(field.type, type) and issubclass(field.type, ObsTerm):
                    self.active_observation_camera_names.append(field_name)
            self.policy_observation_config = combine_configclass_instances(
                "PolicyObservationCfg",
                camera_observations,
                self.policy_observation_config,
            )
'''
    if old_method not in current:
        raise RuntimeError("RoboFinals V1 camera registry method changed; cannot patch safely")

    target.write_text(
        current.replace(import_line, replacement_import, 1).replace(old_method, new_method, 1),
        encoding="utf-8",
    )
    print(f"[flip_table] rebuilt V1 camera fields for true stereo: {target}", flush=True)
    return True


def main() -> None:
    if not _env_bool("FLIP_TABLE_PATCH_G1_GLOBAL_CAMERA", True):
        print("[flip_table] G1 camera patch disabled.", flush=True)
        return
    robofinals_root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
    _validate_patched_runtime_booleans()
    backup_root_raw = os.environ.get(OFFICIAL_V1_BACKUP_ROOT_ENV, "").strip()
    restore_enabled = _env_bool(RESTORE_OFFICIAL_V1_ENV, bool(backup_root_raw))
    if restore_enabled:
        if not backup_root_raw:
            raise ValueError(
                f"{OFFICIAL_V1_BACKUP_ROOT_ENV} is required when "
                f"{RESTORE_OFFICIAL_V1_ENV}=true"
            )
        restore_official_v1_robot_files(
            robofinals_root,
            Path(backup_root_raw),
            expected_sha256=OFFICIAL_V1_SOURCE_SHA256,
        )
    restore_generated_v1_assets(robofinals_root)
    patch_g1_gripper_asset_config(robofinals_root)
    patch_g1_gripper_controller(robofinals_root)
    patch_g1_cameras(robofinals_root)
    patch_camera_config_field_generation(robofinals_root)
    if _env_bool("FLIP_TABLE_PATCH_G1_UPPER_BODY_JOINT_ACTION", True):
        patch_g1_upper_body_joint_action(robofinals_root)
    if _env_bool("FLIP_TABLE_PATCH_G1_GRIPPER_VARIANTS", True):
        patch_g1_gripper_asset_variants(robofinals_root)
    if _env_bool(MATERIAL_BINDING_PATCH_ENV, True):
        patch_g1_gripper_material_binding_api(robofinals_root)
    if _env_bool(UNITREE_MATERIAL_VALUES_ENV, True):
        patch_g1_gripper_unitree_material_values(robofinals_root)
    if _env_bool(CONTACT_MATERIAL_PATCH_ENV, True):
        patch_g1_gripper_contact_material(robofinals_root)
    if _env_bool(ROBOT_COLLISION_ENV, True):
        patch_g1_gripper_physx_collisions(robofinals_root)


if __name__ == "__main__":
    main()
