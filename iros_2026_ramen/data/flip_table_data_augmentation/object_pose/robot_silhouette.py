"""Project the pinned G1 + Dex1-1 visual model into a rectified source image."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np

from ..fk_audit import G1_BODY_JOINT_ORDER
from ..io_utils import sha256_file


DEX1_FINGER_JOINT_ORDER = (
    "left_dex1_finger_joint_1",
    "left_dex1_finger_joint_2",
    "right_dex1_finger_joint_1",
    "right_dex1_finger_joint_2",
)
DEX1_OPEN_POSITION_M = 0.0245
DEX1_CLOSED_POSITION_M = -0.02
DEMO_HAND_OPEN = 4.5
_UPPER_BODY_VISUAL_TOKENS = ("shoulder", "elbow", "wrist", "dex1")


@dataclass(frozen=True)
class RobotSilhouetteMetrics:
    projected_visuals: int
    visible_visuals: int
    mask_pixels_before_dilation: int
    mask_pixels: int
    mask_fraction: float

    def to_json(self) -> dict[str, int | float]:
        return asdict(self)


def robot_silhouette_coverage_is_plausible(mask_fraction: float) -> bool:
    """Accept an off-screen robot while rejecting a broken full-frame projection."""

    return bool(np.isfinite(mask_fraction) and 0.0 <= mask_fraction <= 0.75)


def demo_hand_to_dex1_joint_position(hand_state: np.ndarray) -> np.ndarray:
    """Expand real two-hand motor state into the four coupled Dex1 joints."""

    hand = np.asarray(hand_state, dtype=np.float64)
    if hand.shape != (2,) or not np.isfinite(hand).all():
        raise ValueError("hand_state must contain two finite values")
    position = DEX1_CLOSED_POSITION_M + (hand / DEMO_HAND_OPEN) * (
        DEX1_OPEN_POSITION_M - DEX1_CLOSED_POSITION_M
    )
    if np.any((position < -0.03) | (position > 0.035)):
        raise ValueError("hand_state maps outside the audited Dex1 joint range")
    return np.repeat(position, 2)


def projected_convex_hull_mask(
    points_root: np.ndarray,
    *,
    root_from_camera: np.ndarray,
    intrinsic_matrix: np.ndarray,
    width: int,
    height: int,
    near_m: float = 0.05,
) -> np.ndarray:
    """Rasterize a conservative convex silhouette for one visual mesh."""

    points = np.asarray(points_root, dtype=np.float64)
    transform = np.asarray(root_from_camera, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_root must be a finite Nx3 array")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("root_from_camera must be a finite 4x4 transform")
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_matrix must be a finite 3x3 matrix")
    if width <= 0 or height <= 0 or near_m <= 0.0:
        raise ValueError("image dimensions and near plane must be positive")

    camera_from_root = np.linalg.inv(transform)
    camera_points = (
        camera_from_root[:3, :3] @ points.T
    ).T + camera_from_root[:3, 3]
    camera_points = camera_points[camera_points[:, 2] > near_m]
    output = np.zeros((height, width), dtype=bool)
    if len(camera_points) < 3:
        return output
    pixels_h = (intrinsic @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    margin = float(max(width, height))
    pixels[:, 0] = np.clip(pixels[:, 0], -margin, width - 1 + margin)
    pixels[:, 1] = np.clip(pixels[:, 1], -margin, height - 1 + margin)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    if hull is not None and len(hull) >= 3:
        cv2.fillConvexPoly(output.view(np.uint8), hull, 1)
    return output


class RobotSilhouetteRenderer:
    """Render upper-body occupancy from source joints and the official Dex1 URDF."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        expected_sha256: str,
        dilation_px: int,
    ) -> None:
        import pinocchio as pin
        import trimesh

        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)
        if sha256_file(self.urdf_path) != expected_sha256:
            raise ValueError("G1 + Dex1-1 visual URDF SHA-256 differs from config")
        if dilation_px <= 0 or dilation_px % 2:
            raise ValueError("robot silhouette dilation must be a positive even integer")
        self.dilation_px = dilation_px
        self._pin = pin
        wrapper = pin.RobotWrapper.BuildFromURDF(
            str(self.urdf_path), package_dirs=[str(self.urdf_path.parent)]
        )
        self._model = wrapper.model
        self._data = wrapper.data
        self._visual_model = wrapper.visual_model
        self._visual_data = pin.GeometryData(self._visual_model)
        required = (*G1_BODY_JOINT_ORDER, *DEX1_FINGER_JOINT_ORDER)
        missing = [name for name in required if not self._model.existJointName(name)]
        if missing:
            raise ValueError(f"G1 + Dex1-1 visual URDF is missing joints: {missing}")
        self._body_q_indices = np.asarray(
            [self._model.joints[self._model.getJointId(name)].idx_q for name in G1_BODY_JOINT_ORDER],
            dtype=np.int64,
        )
        self._finger_q_indices = np.asarray(
            [self._model.joints[self._model.getJointId(name)].idx_q for name in DEX1_FINGER_JOINT_ORDER],
            dtype=np.int64,
        )
        visuals: list[tuple[int, np.ndarray]] = []
        for index, geometry in enumerate(self._visual_model.geometryObjects):
            lowered = geometry.name.lower()
            if not any(token in lowered for token in _UPPER_BODY_VISUAL_TOKENS):
                continue
            mesh = trimesh.load(geometry.meshPath, force="mesh", process=False)
            vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(
                geometry.meshScale, dtype=np.float64
            )
            if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
                raise ValueError(f"invalid robot visual mesh: {geometry.meshPath}")
            visuals.append((index, vertices))
        if not visuals:
            raise ValueError("G1 + Dex1-1 visual URDF has no upper-body meshes")
        self._visuals = tuple(visuals)

    def render(
        self,
        *,
        robot_q_current: np.ndarray,
        hand_state: np.ndarray,
        root_from_camera: np.ndarray,
        intrinsic_matrix: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, RobotSilhouetteMetrics]:
        body = np.asarray(robot_q_current, dtype=np.float64)
        if body.shape != (36,) or not np.isfinite(body).all():
            raise ValueError("robot_q_current must be a finite 36D vector")
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[self._body_q_indices] = body[7:]
        q[self._finger_q_indices] = demo_hand_to_dex1_joint_position(hand_state)
        self._pin.forwardKinematics(self._model, self._data, q)
        self._pin.updateGeometryPlacements(
            self._model,
            self._data,
            self._visual_model,
            self._visual_data,
            q,
        )

        mask = np.zeros((height, width), dtype=bool)
        visible_visuals = 0
        for index, vertices in self._visuals:
            placement = self._visual_data.oMg[index]
            points_root = (
                np.asarray(placement.rotation) @ vertices.T
            ).T + np.asarray(placement.translation)
            visual_mask = projected_convex_hull_mask(
                points_root,
                root_from_camera=root_from_camera,
                intrinsic_matrix=intrinsic_matrix,
                width=width,
                height=height,
            )
            if visual_mask.any():
                visible_visuals += 1
                mask |= visual_mask
        before = int(mask.sum())
        kernel_size = 2 * self.dilation_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0
        metrics = RobotSilhouetteMetrics(
            projected_visuals=len(self._visuals),
            visible_visuals=visible_visuals,
            mask_pixels_before_dilation=before,
            mask_pixels=int(mask.sum()),
            mask_fraction=float(mask.mean()),
        )
        return mask, metrics
