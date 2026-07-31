"""Monocular CAD-based localization of the assembled white tabletop.

Only RGB pixels and immutable camera calibration are consumed here. Simulator
poses are deliberately absent from this module so the same estimator can run on
the real G1 head-left stream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class WristShaftObservation:
    detected: bool
    center_px: tuple[float, float] | None
    confidence: float
    white_fraction: float
    vertical_support: float
    bounding_box_px: tuple[int, int, int, int] | None


@dataclass(frozen=True)
class WristTabletopEdgeObservation:
    detected: bool
    edge_y_px: float | None
    confidence: float
    white_fraction: float


class WristTabletopEdgeDetector:
    """Locate the tabletop's lower white edge between the Dex1 fingers."""

    _ROI_X = (0.18, 0.82)
    _ROI_Y = (0.05, 0.88)

    def detect(self, rgb: np.ndarray) -> WristTabletopEdgeObservation:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("wrist edge detector requires an HxWx3 RGB image")
        image = image[..., :3].astype(np.uint8, copy=False)
        height, width = image.shape[:2]
        x0, x1 = (round(width * value) for value in self._ROI_X)
        y0, y1 = (round(height * value) for value in self._ROI_Y)
        roi = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        white = np.asarray(
            (hsv[..., 1] <= 105) & (hsv[..., 2] >= 115), dtype=np.uint8
        ) * 255
        white = cv2.morphologyEx(
            white,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
        )
        white_fraction = float(np.count_nonzero(white)) / max(1, white.size)
        row_indices = np.arange(roi.shape[0], dtype=np.int32)[:, None]
        bottom_by_column = np.max(np.where(white > 0, row_indices, -1), axis=0)
        valid = bottom_by_column[
            (bottom_by_column >= round(0.15 * roi.shape[0]))
            & (bottom_by_column <= round(0.96 * roi.shape[0]))
        ]
        support = float(valid.size) / max(1, roi.shape[1])
        if valid.size < round(0.35 * roi.shape[1]) or white_fraction < 0.08:
            return WristTabletopEdgeObservation(False, None, 0.0, white_fraction)
        edge_local_y = float(np.median(valid))
        dispersion = float(np.median(np.abs(valid - edge_local_y)))
        confidence = float(
            np.clip(
                0.55 * min(1.0, support / 0.70)
                + 0.30 * min(1.0, white_fraction / 0.35)
                + 0.15 * max(0.0, 1.0 - dispersion / (0.08 * height)),
                0.0,
                1.0,
            )
        )
        return WristTabletopEdgeObservation(
            True,
            y0 + edge_local_y,
            confidence,
            white_fraction,
        )


class WristShaftDetector:
    """Detect the white table leg inside the Dex1 finger corridor."""

    _ROI_X = (0.35, 0.65)
    _ROI_Y = (0.42, 0.94)
    _MAX_CENTER_ERROR_FRACTION = 0.11
    _MIN_AREA_FRACTION = 0.018
    _MIN_VERTICAL_SUPPORT = 0.30
    _MIN_SHAFT_WIDTH_FRACTION = 0.025
    _MAX_SHAFT_WIDTH_FRACTION = 0.35

    def detect(
        self,
        rgb: np.ndarray,
        *,
        require_centered: bool = True,
    ) -> WristShaftObservation:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("wrist shaft detector requires an HxWx3 RGB image")
        image = image[..., :3].astype(np.uint8, copy=False)
        height, width = image.shape[:2]
        roi_x = self._ROI_X if require_centered else (0.05, 0.95)
        x0, x1 = (round(width * value) for value in roi_x)
        y0, y1 = (round(height * value) for value in self._ROI_Y)
        roi = image[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        white = np.asarray(
            (hsv[..., 1] <= 105) & (hsv[..., 2] >= 115), dtype=np.uint8
        ) * 255
        white = cv2.morphologyEx(
            white,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 13)),
        )
        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        target_x = 0.5 * width
        white_fraction = float(np.count_nonzero(white)) / roi_area
        row_indices = np.arange(roi.shape[0], dtype=np.int32)[:, None]
        bottom_by_column = np.max(
            np.where(white > 0, row_indices, -1), axis=0
        )
        deepest = int(np.max(bottom_by_column, initial=-1))
        if deepest < round(self._MIN_VERTICAL_SUPPORT * roi.shape[0]):
            return WristShaftObservation(
                False, None, 0.0, white_fraction, 0.0, None
            )

        # The leg is often connected to the tabletop in the binary mask. Its
        # lower end nevertheless protrudes farther between the fingers than
        # the broad tabletop underside. Extract narrow runs close to the
        # deepest white pixel instead of treating the full assembly as one
        # connected component.
        protrusion_tolerance = max(4, round(0.08 * roi.shape[0]))
        protruding_columns = np.flatnonzero(
            bottom_by_column >= deepest - protrusion_tolerance
        )
        runs = np.split(
            protruding_columns,
            np.flatnonzero(np.diff(protruding_columns) > 1) + 1,
        )
        candidates: list[tuple[float, np.ndarray, int, int, float, float]] = []
        for run in runs:
            if run.size == 0:
                continue
            run_width = int(run[-1] - run[0] + 1)
            if not (
                self._MIN_SHAFT_WIDTH_FRACTION * roi.shape[1]
                <= run_width
                <= self._MAX_SHAFT_WIDTH_FRACTION * roi.shape[1]
            ):
                continue
            shaft_pixels = white[:, run] > 0
            supported_rows = np.flatnonzero(np.any(shaft_pixels, axis=1))
            if supported_rows.size == 0:
                continue
            vertical_support = float(
                supported_rows[-1] - supported_rows[0] + 1
            ) / max(1, roi.shape[0])
            fill_fraction = float(np.count_nonzero(shaft_pixels)) / max(
                1, shaft_pixels.size
            )
            area_fraction = float(np.count_nonzero(shaft_pixels)) / roi_area
            center_x = x0 + 0.5 * float(run[0] + run[-1])
            center_error = abs(center_x - target_x) / width
            if (
                vertical_support < self._MIN_VERTICAL_SUPPORT
                or area_fraction < self._MIN_AREA_FRACTION
                or (
                    require_centered
                    and center_error > self._MAX_CENTER_ERROR_FRACTION
                )
            ):
                continue
            score = (
                0.55 * min(1.0, vertical_support / self._MIN_VERTICAL_SUPPORT)
                + 0.30 * fill_fraction
                - 1.5 * center_error
            )
            candidates.append(
                (
                    score,
                    run,
                    int(supported_rows[0]),
                    int(supported_rows[-1]),
                    vertical_support,
                    fill_fraction,
                )
            )
        if not candidates:
            return WristShaftObservation(
                False, None, 0.0, white_fraction, 0.0, None
            )
        _, run, top, bottom, vertical_support, fill_fraction = max(
            candidates, key=lambda candidate: candidate[0]
        )
        left = int(run[0])
        box_width = int(run[-1] - run[0] + 1)
        box_height = int(bottom - top + 1)
        center_x = x0 + 0.5 * float(run[0] + run[-1])
        center_y = y0 + 0.5 * float(top + bottom)
        center_score = max(
            0.0,
            1.0
            - abs(center_x - target_x)
            / (width * self._MAX_CENTER_ERROR_FRACTION),
        )
        confidence = float(
            np.clip(
                0.45 * fill_fraction
                + 0.35
                * min(1.0, vertical_support / self._MIN_VERTICAL_SUPPORT)
                + 0.20 * center_score,
                0.0,
                1.0,
            )
        )
        return WristShaftObservation(
            True,
            (center_x, center_y),
            confidence,
            white_fraction,
            vertical_support,
            (x0 + left, y0 + top, box_width, box_height),
        )


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic_matrix: np.ndarray
    distortion: np.ndarray
    root_from_camera: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.intrinsic_matrix).shape != (3, 3):
            raise ValueError("intrinsic_matrix must be [3,3]")
        if np.asarray(self.root_from_camera).shape != (4, 4):
            raise ValueError("root_from_camera must be [4,4]")

    @classmethod
    def _g1_head_left_sim_intrinsics(cls) -> tuple[np.ndarray, np.ndarray]:
        """Return the pinhole intrinsics authored for the simulated left eye."""

        focal_px = 24.0 * 640.0 / 45.56883749280177
        intrinsic = np.asarray(
            ((focal_px, 0.0, 320.0), (0.0, focal_px, 240.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        return intrinsic, np.zeros(5, dtype=np.float64)

    @classmethod
    def g1_head_left_real_raw_intrinsics(cls) -> tuple[np.ndarray, np.ndarray]:
        """Return the measured raw head-left calibration for real deployment."""

        intrinsic = np.asarray(
            (
                (337.5311318539417, 0.0, 316.5285046932812),
                (0.0, 336.61378142923456, 232.50620475777816),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        )
        distortion = np.asarray(
            (0.06635329597971165, -0.07841619072258442, -0.0032837567734969727,
             -0.0010816865229956933, 0.021030073866954904),
            dtype=np.float64,
        )
        return intrinsic, distortion

    @classmethod
    def g1_head_left_torso_from_camera(cls) -> np.ndarray:
        """Return the organizer's torso-link to OpenCV head-left transform."""

        # The organizer offset is an OpenGL camera pose relative to torso_link.
        # OpenCV uses +x right, +y down, +z forward, hence diag(1,-1,-1).
        quaternion_xyzw = np.asarray(
            (0.26523914, -0.27106013, -0.66472446, 0.64367383), dtype=np.float64
        )
        x, y, z, w = quaternion_xyzw
        root_from_gl_rotation = np.asarray(
            (
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ),
            dtype=np.float64,
        )
        torso_from_camera = np.eye(4, dtype=np.float64)
        torso_from_camera[:3, :3] = root_from_gl_rotation @ np.diag((1.0, -1.0, -1.0))
        torso_from_camera[:3, 3] = (0.10209156, 0.02077481159355057, 0.42446595)
        return torso_from_camera

    @classmethod
    def g1_head_left_from_torso(cls, root_from_torso: np.ndarray) -> "CameraCalibration":
        """Compose measured torso FK with the fixed physical camera mount."""

        root_from_torso = np.asarray(root_from_torso, dtype=np.float64)
        if root_from_torso.shape != (4, 4) or not np.isfinite(root_from_torso).all():
            raise ValueError("root_from_torso must be finite [4,4]")
        intrinsic, distortion = cls._g1_head_left_sim_intrinsics()
        root_from_camera = root_from_torso @ cls.g1_head_left_torso_from_camera()
        return cls(intrinsic, distortion, root_from_camera)

    @classmethod
    def g1_head_left(cls) -> "CameraCalibration":
        """Return RoboFinals V1's fixed root-to-head-left calibration.

        The V1 G1 gripper articulation keeps ``torso_link`` at this fixed
        transform relative to the policy root while the lower body is locked.
        This is intentionally distinct from a physical robot's torso FK.
        """

        root_from_torso = np.eye(4, dtype=np.float64)
        root_from_torso[:3, 3] = (-0.0039635, 0.0, 0.044)
        return cls.g1_head_left_from_torso(root_from_torso)


@dataclass(frozen=True)
class TabletopEstimate:
    root_from_table: np.ndarray
    camera_from_table: np.ndarray
    corners_px: np.ndarray
    mask: np.ndarray
    confidence: float
    reprojection_error_px: float
    area_fraction: float

    @property
    def center_root_m(self) -> np.ndarray:
        return self.root_from_table[:3, 3].copy()

    @property
    def yaw_root_rad(self) -> float:
        axis = self.root_from_table[:3, 0]
        return math.atan2(float(axis[1]), float(axis[0]))


@dataclass(frozen=True)
class LegDetection:
    side: str
    endpoints_px: np.ndarray
    confidence: float
    tracked: bool = False
    vertical_alignment: float = 0.0
    inferred_from_tabletop: bool = False
    cad_projected_axis: bool = False
    attachment_root_m: np.ndarray | None = None


@dataclass(frozen=True)
class _LegAxisCandidate:
    endpoints_px: np.ndarray
    attachment_px: np.ndarray
    attachment_root_m: np.ndarray
    confidence: float
    vertical_alignment: float
    score: float


class TableLegDetector:
    """Detect the two front table legs from RGB edges near the tabletop."""

    _MIN_VERTICAL_ALIGNMENT = math.cos(math.radians(28.0))
    _TABLE_LENGTH_M = 0.58
    _TABLE_DEPTH_M = 0.42
    _LEG_INSET_M = 0.035
    # Measured from RoboFinals-IKEA-V1's assembled Table001 visual mesh in the
    # Table001_01 body frame. The body origin is not the tabletop mid-plane.
    _CAD_LEG_CENTER_X_M = 0.261315
    _CAD_LEG_CENTER_Y_M = 0.181380
    _CAD_LEG_ROOT_Z_M = 0.016341
    _MAX_NEAR_LEG_CORRECTION_M = 0.12

    @staticmethod
    def _project_nearest_corner_to_axis(
        endpoints: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray:
        """Intersect an observed leg axis with its nearest tabletop corner."""

        start, end = np.asarray(endpoints, dtype=np.float64).reshape(2, 2)
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator < 1.0:
            raise ValueError("degenerate table-leg axis")
        candidates = []
        for corner in np.asarray(corners, dtype=np.float64).reshape(-1, 2):
            fraction = float(np.dot(corner - start, direction) / denominator)
            projection = start + fraction * direction
            perpendicular_distance = float(np.linalg.norm(corner - projection))
            candidates.append((perpendicular_distance, projection))
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _center_line_on_white_shaft(
        mask: np.ndarray,
        endpoints: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Move a Hough edge to the center of its narrow white shaft."""

        points = np.asarray(endpoints, dtype=np.float64).reshape(2, 2)
        direction = points[1] - points[0]
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            return points.copy(), 0, 0.0
        direction /= length
        perpendicular = np.asarray((-direction[1], direction[0]))
        offsets = np.arange(-32, 33, dtype=np.float64)
        center_offsets: list[float] = []
        shaft_widths: list[float] = []
        height, width = mask.shape
        for fraction in (0.25, 0.40, 0.55, 0.70, 0.85):
            base = (1.0 - fraction) * points[0] + fraction * points[1]
            samples = np.rint(base + offsets[:, None] * perpendicular).astype(np.int32)
            valid = (
                (samples[:, 0] >= 0)
                & (samples[:, 0] < width)
                & (samples[:, 1] >= 0)
                & (samples[:, 1] < height)
            )
            white = np.zeros(len(offsets), dtype=bool)
            white[valid] = mask[samples[valid, 1], samples[valid, 0]] > 0
            indices = np.flatnonzero(white)
            if not len(indices):
                continue
            split_at = np.flatnonzero(np.diff(indices) > 1) + 1
            runs = np.split(indices, split_at)
            run = min(
                runs,
                key=lambda candidate: float(np.min(np.abs(offsets[candidate]))),
            )
            run_width = float(len(run))
            # A Dex1 table leg projects to a narrow shaft. Broad runs belong to
            # the tabletop, braces, robot body, floor, or wall.
            if not 5.0 <= run_width <= 28.0:
                continue
            center_offsets.append(float(np.mean(offsets[run])))
            shaft_widths.append(run_width)
        if len(center_offsets) < 2:
            return points.copy(), len(center_offsets), 0.0
        center_shift = float(np.median(center_offsets))
        centered = points + center_shift * perpendicular
        return centered, len(center_offsets), float(np.median(shaft_widths))

    @staticmethod
    def _pixel_to_plane(
        pixel: np.ndarray,
        calibration: CameraCalibration,
        plane_z_m: float,
    ) -> np.ndarray:
        normalized = cv2.undistortPoints(
            np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
            calibration.intrinsic_matrix,
            calibration.distortion,
        ).reshape(2)
        ray_root = calibration.root_from_camera[:3, :3] @ np.asarray(
            (normalized[0], normalized[1], 1.0)
        )
        if abs(float(ray_root[2])) < 1.0e-6:
            raise ValueError("RGB ray is parallel to the tabletop plane")
        distance = (float(plane_z_m) - calibration.root_from_camera[2, 3]) / ray_root[2]
        if distance <= 0.0:
            raise ValueError("RGB tabletop point lies behind the camera")
        return calibration.root_from_camera[:3, 3] + distance * ray_root

    @staticmethod
    def _project_root_point(point: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
        camera_from_root = np.linalg.inv(calibration.root_from_camera)
        point_camera = (
            camera_from_root[:3, :3] @ np.asarray(point, dtype=np.float64)
            + camera_from_root[:3, 3]
        )
        if point_camera[2] <= 0.0:
            raise ValueError("root point lies behind the camera")
        projected, _ = cv2.projectPoints(
            point_camera.reshape(1, 3),
            np.zeros(3),
            np.zeros(3),
            calibration.intrinsic_matrix,
            calibration.distortion,
        )
        return projected.reshape(2)

    def detect(
        self,
        rgb: np.ndarray,
        tabletop: TabletopEstimate,
        calibration: CameraCalibration,
        tabletop_z_m: float,
        previous: dict[str, LegDetection] | None = None,
    ) -> dict[str, LegDetection]:
        expected = self.estimate_near_leg_centers(tabletop.root_from_table)
        try:
            candidates = self._axis_candidates(
                rgb, tabletop, calibration, tabletop_z_m
            )
        except ValueError:
            candidates = []
        detections: dict[str, LegDetection] = {}
        selected_indices: set[int] = set()
        for side in ("left", "right"):
            options: list[tuple[float, int, _LegAxisCandidate]] = []
            for index, candidate in enumerate(candidates):
                if index in selected_indices:
                    continue
                distance = float(
                    np.linalg.norm(
                        candidate.attachment_root_m[:2] - expected[side][:2]
                    )
                )
                if distance <= self._MAX_NEAR_LEG_CORRECTION_M:
                    options.append((distance, index, candidate))
            if options:
                _, index, candidate = min(
                    options,
                    key=lambda item: (item[0], -item[2].score),
                )
                selected_indices.add(index)
                # Hough returns only the visible section of a leg, often
                # stopping at an underside brace.  Its endpoints are therefore
                # not physical attachment/tip points and must never become a
                # grasp target.  The registered tabletop CAD fixes the full
                # Dex1-relevant leg axis; the RGB segment validates that axis.
                attachment = expected[side].copy()
                attachment_px = self._project_root_point(attachment, calibration)
                tip_px = self._project_root_point(
                    attachment + np.asarray((0.0, 0.0, 0.40)), calibration
                )
                detections[side] = LegDetection(
                    side,
                    np.stack((attachment_px, tip_px)),
                    candidate.confidence,
                    vertical_alignment=candidate.vertical_alignment,
                    cad_projected_axis=True,
                    attachment_root_m=attachment,
                )
                continue

            # A hand or brace can fully occlude a shaft. The tabletop corners,
            # known CAD dimensions, and calibrated camera still determine all
            # four attachment points. Keep that geometric inference explicit in
            # diagnostics instead of relabelling a visible rear leg as a front
            # leg.
            attachment = expected[side].copy()
            attachment_px = self._project_root_point(attachment, calibration)
            tip_px = self._project_root_point(
                attachment + np.asarray((0.0, 0.0, 0.40)), calibration
            )
            detections[side] = LegDetection(
                side,
                np.stack((attachment_px, tip_px)),
                max(0.25, 0.75 * tabletop.confidence),
                vertical_alignment=1.0,
                inferred_from_tabletop=True,
                cad_projected_axis=True,
                attachment_root_m=attachment.copy(),
            )
        return detections

    def _axis_candidates(
        self,
        rgb: np.ndarray,
        tabletop: TabletopEstimate,
        calibration: CameraCalibration,
        tabletop_z_m: float,
    ) -> list[_LegAxisCandidate]:
        """Return RGB shaft axes with their support-plane attachment points."""

        mask = TabletopPoseEstimator.segment_table_assembly(rgb)
        edges = cv2.Canny(mask, 40, 120)
        lines = cv2.HoughLinesP(
            edges, 1.0, np.pi / 180.0, threshold=20, minLineLength=30, maxLineGap=25
        )
        if lines is None:
            raise ValueError("no table-leg line candidates found")

        height, width = mask.shape
        candidates: list[_LegAxisCandidate] = []
        for raw in lines[:, 0, :]:
            raw_endpoints = raw.reshape(2, 2).astype(np.float64)
            centered, shaft_support, shaft_width = self._center_line_on_white_shaft(
                mask, raw_endpoints
            )
            if shaft_support < 2:
                continue
            direction = centered[1] - centered[0]
            length = float(np.linalg.norm(direction))
            if length < 30.0:
                continue

            best_alignment = -1.0
            attachment_px = None
            attachment_root = None
            for attachment_index in (0, 1):
                candidate_attachment = centered[attachment_index]
                candidate_tip = centered[1 - attachment_index]
                try:
                    candidate_root = self._pixel_to_plane(
                        candidate_attachment, calibration, tabletop_z_m
                    )
                    vertical_tip_px = self._project_root_point(
                        candidate_root + np.asarray((0.0, 0.0, 0.40)), calibration
                    )
                except ValueError:
                    continue
                observed_direction = candidate_tip - candidate_attachment
                expected_direction = vertical_tip_px - candidate_attachment
                denominator = float(
                    np.linalg.norm(observed_direction) * np.linalg.norm(expected_direction)
                )
                if denominator <= 1.0e-6:
                    continue
                alignment = float(
                    np.dot(observed_direction, expected_direction) / denominator
                )
                if alignment > best_alignment:
                    best_alignment = alignment
                    attachment_px = candidate_attachment
                    attachment_root = candidate_root
            if (
                attachment_px is None
                or attachment_root is None
                or best_alignment < self._MIN_VERTICAL_ALIGNMENT
            ):
                continue

            # Hough edges commonly continue into the white corner bracket, so
            # their endpoint is not the physical leg/table attachment. Refine
            # it by intersecting the shaft axis with the nearest broad-tabletop
            # corner measured in the same RGB frame.
            refined_attachment_px = self._project_nearest_corner_to_axis(
                centered, tabletop.corners_px
            )
            if float(np.linalg.norm(refined_attachment_px - attachment_px)) <= 90.0:
                try:
                    refined_attachment_root = self._pixel_to_plane(
                        refined_attachment_px, calibration, tabletop_z_m
                    )
                except ValueError:
                    pass
                else:
                    attachment_px = refined_attachment_px
                    attachment_root = refined_attachment_root

            corner_distance = min(
                float(np.linalg.norm(attachment_px - corner))
                for corner in tabletop.corners_px
            )
            if corner_distance > 110.0:
                continue
            if attachment_px[0] < 0.04 * width or attachment_px[0] > 0.96 * width:
                continue
            score = (
                1.50 * best_alignment
                + max(0.0, 1.0 - corner_distance / 110.0)
                + min(length / (0.30 * height), 1.0)
                + 0.08 * shaft_support
                + max(0.0, 1.0 - abs(shaft_width - 16.0) / 16.0) * 0.20
            )
            confidence = float(np.clip((score - 1.5) / 1.8, 0.0, 1.0))
            candidates.append(
                _LegAxisCandidate(
                    endpoints_px=centered,
                    attachment_px=attachment_px,
                    attachment_root_m=attachment_root,
                    confidence=confidence,
                    vertical_alignment=best_alignment,
                    score=score,
                )
            )
        if not candidates:
            raise ValueError("no calibrated table-leg shaft candidates found")
        return candidates

    def detect_near_leg_attachment_points(
        self,
        rgb: np.ndarray,
        tabletop: TabletopEstimate,
        calibration: CameraCalibration,
        table_frame: np.ndarray,
        *,
        tabletop_z_m: float,
    ) -> dict[str, np.ndarray]:
        """Reconstruct the reachable pair from the RGB tabletop frame.

        White underside braces can look like leg shafts and bias a direct
        ray-plane intersection by several centimetres. Once the calibrated
        0.58 x 0.42 m tabletop frame is known, all four attachment centers are
        fixed by construction. Shaft detections remain part of table-frame
        validation upstream but must not displace these CAD-consistent targets.
        """

        del rgb, tabletop, calibration
        expected = self.estimate_near_leg_centers(table_frame)
        selected = {side: point.copy() for side, point in expected.items()}
        separation = float(
            np.linalg.norm(selected["left"][:2] - selected["right"][:2])
        )
        if not 0.22 <= separation <= 0.58:
            raise ValueError(
                f"implausible direct RGB near-leg separation: {separation:.3f} m"
            )
        return selected

    @staticmethod
    def estimate_leg_attachment_points(
        tabletop: TabletopEstimate,
        detections: dict[str, LegDetection],
        calibration: CameraCalibration,
        *,
        tabletop_z_m: float,
    ) -> dict[str, np.ndarray]:
        """Intersect detected leg axes with the calibrated tabletop plane."""

        if set(detections) != {"left", "right"}:
            raise ValueError("both RGB front-leg detections are required")
        def pixel_to_plane(pixel: np.ndarray) -> np.ndarray:
            return TableLegDetector._pixel_to_plane(pixel, calibration, tabletop_z_m)

        image_center_x = float(tabletop.corners_px[:, 0].mean())
        attachments: dict[str, np.ndarray] = {}
        for side, detection in detections.items():
            if detection.attachment_root_m is not None:
                attachment_root = np.asarray(
                    detection.attachment_root_m, dtype=np.float64
                )
                if attachment_root.shape != (3,) or not np.isfinite(attachment_root).all():
                    raise ValueError(f"invalid {side} leg attachment")
                attachments[side] = attachment_root.copy()
                continue
            start, end = np.asarray(detection.endpoints_px, dtype=np.float64)
            direction = end - start
            denominator = float(np.dot(direction, direction))
            if denominator < 1.0:
                raise ValueError(f"degenerate {side} leg detection")
            side_corners = tabletop.corners_px[
                tabletop.corners_px[:, 0] < image_center_x
                if side == "left"
                else tabletop.corners_px[:, 0] >= image_center_x
            ]
            attachment_corner = max(side_corners, key=lambda corner: float(corner[1]))
            fraction = float(np.dot(attachment_corner - start, direction) / denominator)
            attachment_px = start + fraction * direction
            attachments[side] = pixel_to_plane(attachment_px)

        return attachments

    @staticmethod
    def estimate_table_frame(
        tabletop: TabletopEstimate,
        detections: dict[str, LegDetection],
        calibration: CameraCalibration,
        *,
        tabletop_z_m: float,
    ) -> np.ndarray:
        """Fuse the PnP tabletop pose and front-leg RGB detections on its plane.

        ``TabletopPoseEstimator`` owns the metric table center and orientation.
        The leg detector only resolves the front-edge direction and validates the
        result.  Reconstructing the center by intersecting an image ray with a
        fixed root-height plane silently mixed two coordinate estimates and
        failed whenever the robot root height changed.
        """

        attachments = TableLegDetector.estimate_leg_attachment_points(
            tabletop,
            detections,
            calibration,
            tabletop_z_m=tabletop_z_m,
        )
        attachment_axis = attachments["left"] - attachments["right"]
        attachment_axis[2] = 0.0
        separation = float(np.linalg.norm(attachment_axis))
        if not 0.20 <= separation <= 1.20:
            raise ValueError(f"implausible RGB front-leg separation: {separation:.3f} m")
        attachment_axis /= separation

        center = tabletop.root_from_table[:3, 3].astype(np.float64).copy()
        if not np.isfinite(center).all():
            raise ValueError("non-finite RGB tabletop PnP center")
        # ``tabletop_z_m`` is the PnP center height supplied by the caller.
        # Keep the explicit assignment to make the projection-plane contract
        # unambiguous for the leg attachment reconstruction above.
        center[2] = float(tabletop_z_m)
        # Preserve the known 0.58 x 0.42 m PnP axes. At arbitrary yaw the two
        # root-nearest legs can share either a long or a short edge; assuming a
        # short edge would alias every long-edge presentation by 90 degrees.
        long_axis = tabletop.root_from_table[:3, 0].astype(np.float64).copy()
        long_axis[2] = 0.0
        long_norm = float(np.linalg.norm(long_axis))
        if long_norm <= 1.0e-6:
            raise ValueError("degenerate RGB tabletop long axis")
        long_axis /= long_norm
        down = np.asarray((0.0, 0.0, -1.0))
        short_axis = np.cross(down, long_axis)
        expected_long_span = (
            TableLegDetector._TABLE_LENGTH_M - 2.0 * TableLegDetector._LEG_INSET_M
        )
        expected_short_span = (
            TableLegDetector._TABLE_DEPTH_M - 2.0 * TableLegDetector._LEG_INSET_M
        )
        pair_is_long_edge = abs(separation - expected_long_span) < abs(
            separation - expected_short_span
        )
        signed_alignment = float(
            np.dot(long_axis if pair_is_long_edge else short_axis, attachment_axis)
        )
        if abs(signed_alignment) < 0.50:
            raise ValueError("RGB leg pair is inconsistent with tabletop PnP axes")
        if signed_alignment < 0.0:
            long_axis *= -1.0
            short_axis *= -1.0
        root_from_table = np.eye(4, dtype=np.float64)
        root_from_table[:3, 0] = long_axis
        root_from_table[:3, 1] = short_axis
        root_from_table[:3, 2] = (0.0, 0.0, -1.0)
        root_from_table[:3, 3] = center
        return root_from_table

    @classmethod
    def estimate_near_leg_centers(cls, root_from_table: np.ndarray) -> dict[str, np.ndarray]:
        """Reconstruct the reachable leg pair from RGB axes and known dimensions."""

        all_legs = cls.estimate_all_leg_centers(root_from_table)
        near = sorted(all_legs.values(), key=lambda point: float(point[0]))[:2]
        left, right = (near[0], near[1]) if near[0][1] >= near[1][1] else (near[1], near[0])
        return {"left": left.copy(), "right": right.copy()}

    @classmethod
    def select_arm_reachable_leg_centers(
        cls,
        root_from_table: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Assign distinct CAD legs to the left and right arm workspaces.

        A one- or two-hand tabletop rotation can translate the assembly as it
        yaws. Selecting the two root-nearest corners after that motion can put
        both targets on the same side of G1 and force a cross-body grasp. The
        nominal workspace anchors below are expressed in the robot root frame;
        they approximate comfortable Dex1 grasp centers and require only the
        RGB-estimated table frame and known table dimensions.
        """

        points = list(cls.estimate_all_leg_centers(root_from_table).values())
        anchors = {
            "left": np.asarray((0.42, 0.24), dtype=np.float64),
            "right": np.asarray((0.42, -0.24), dtype=np.float64),
        }
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for left_index, left in enumerate(points):
            for right_index, right in enumerate(points):
                if left_index == right_index:
                    continue
                score = float(
                    np.sum((left[:2] - anchors["left"]) ** 2)
                    + np.sum((right[:2] - anchors["right"]) ** 2)
                )
                candidate = (score, left, right)
                if best is None or score < best[0]:
                    best = candidate
        assert best is not None
        return {"left": best[1].copy(), "right": best[2].copy()}

    @classmethod
    def estimate_all_leg_centers(cls, root_from_table: np.ndarray) -> dict[str, np.ndarray]:
        """Return all four CAD leg attachments in the RGB-estimated table frame."""

        table = np.asarray(root_from_table, dtype=np.float64)
        if table.shape != (4, 4) or not np.isfinite(table).all():
            raise ValueError("root_from_table must be finite [4,4]")
        long_axis = table[:3, 0].copy()
        long_axis[2] = 0.0
        long_axis /= np.linalg.norm(long_axis)
        short_axis = table[:3, 1].copy()
        short_axis[2] = 0.0
        short_axis /= np.linalg.norm(short_axis)
        points = {}
        for long_sign in (-1.0, 1.0):
            for short_sign in (-1.0, 1.0):
                points[f"{int(long_sign):+d}_{int(short_sign):+d}"] = (
                    table[:3, :3]
                    @ np.asarray(
                        (
                            long_sign * cls._CAD_LEG_CENTER_X_M,
                            short_sign * cls._CAD_LEG_CENTER_Y_M,
                            cls._CAD_LEG_ROOT_Z_M,
                        ),
                        dtype=np.float64,
                    )
                    + table[:3, 3]
                )
        return points

    @staticmethod
    def render_debug(
        rgb: np.ndarray,
        tabletop: TabletopEstimate,
        detections: dict[str, LegDetection],
    ) -> np.ndarray:
        output = TabletopPoseEstimator.render_debug(rgb, tabletop)
        colors = {"left": (255, 64, 64), "right": (64, 128, 255)}
        for side, detection in detections.items():
            points = np.rint(detection.endpoints_px).astype(np.int32)
            cv2.line(output, tuple(points[0]), tuple(points[1]), colors[side], 5, cv2.LINE_AA)
            anchor = tuple(points.mean(axis=0).astype(int))
            if detection.inferred_from_tabletop:
                source = "CAD inferred"
            elif detection.cad_projected_axis:
                source = "CAD + RGB"
            else:
                source = "tracked" if detection.tracked else "detected"
            cv2.putText(
                output,
                f"{side} leg {detection.confidence:.2f} vertical={detection.vertical_alignment:.2f} {source}",
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[side], 2,
            )
        return output


def _order_clockwise(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def _quadrilateral_from_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the outer tabletop quadrilateral, excluding legs and floor leaks."""

    height, width = mask.shape
    # The floor can be bright under domain randomization and occasionally leaks
    # through the black-workbench mask.  The assembled tabletop is the broad,
    # central component above the workbench front edge, never a bottom-border
    # component.  These are image geometry constraints, not simulator state.
    expected_center = np.asarray((0.5 * width, 0.42 * height), dtype=np.float64)

    # Preserve an already clean, filled four-sided target. This path is useful
    # for calibration checks and overexposed real frames, where opening first
    # would round the true perspective corners and bias PnP depth.
    raw_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(raw_contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < 0.012 * width * height:
            continue
        hull = cv2.convexHull(contour)
        rectangle_area = float(cv2.minAreaRect(hull)[1][0] * cv2.minAreaRect(hull)[1][1])
        if rectangle_area <= 1.0 or area / rectangle_area < 0.82:
            continue
        polygon = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
        if len(polygon) == 4 and cv2.isContourConvex(polygon.astype(np.float32)):
            return _order_clockwise(polygon.astype(np.float64)), area

    # The panel width changes with table/robot placement and the randomized
    # camera intrinsics. Try the strongest leg-removal scale first, then relax
    # only when that scale leaves no physically plausible tabletop component.
    for kernel_size in (31, 21, 15):
        body_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        )
        contours, _ = cv2.findContours(
            body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        best_contour: np.ndarray | None = None
        best_score = -math.inf
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 0.012 * width * height:
                continue
            _, y, _, component_height = cv2.boundingRect(contour)
            if y + component_height >= round(0.90 * height):
                continue
            rectangle = cv2.minAreaRect(contour)
            rectangle_area = float(rectangle[1][0] * rectangle[1][1])
            if rectangle_area <= 1.0:
                continue
            fill_ratio = area / rectangle_area
            if fill_ratio < 0.22:
                continue
            center_distance = np.linalg.norm(
                np.asarray(rectangle[0], dtype=np.float64) - expected_center
            )
            center_score = max(0.10, 1.0 - center_distance / math.hypot(width, height))
            score = area * min(1.0, fill_ratio / 0.55) * center_score
            if score > best_score:
                best_score = score
                best_contour = contour
        if best_contour is None:
            continue

        hull = cv2.convexHull(best_contour)
        perimeter = cv2.arcLength(hull, True)
        candidates: list[np.ndarray] = []
        # Do not take the first four-corner simplification. Inner braces can
        # produce a small quadrilateral at low epsilon; maximize the supported
        # outer area among convex four-corner candidates instead.
        for epsilon_fraction in (0.012, 0.018, 0.025, 0.035, 0.05, 0.07, 0.10):
            candidate = cv2.approxPolyDP(
                hull, epsilon_fraction * perimeter, True
            ).reshape(-1, 2)
            if len(candidate) == 4 and cv2.isContourConvex(candidate.astype(np.float32)):
                candidates.append(candidate.astype(np.float64))
        candidates.append(cv2.boxPoints(cv2.minAreaRect(best_contour)).astype(np.float64))

        best_polygon: np.ndarray | None = None
        best_polygon_score = -math.inf
        component_area = float(cv2.contourArea(best_contour))
        for polygon in candidates:
            polygon_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
            # A uniformly visible tabletop can itself be a filled rectangle;
            # accept its near-equal fitted envelope instead of requiring an
            # empty rim/interior gap.
            if polygon_area < 0.90 * component_area:
                continue
            support = component_area / polygon_area
            if not 0.20 <= support <= 1.05:
                continue
            center_distance = np.linalg.norm(polygon.mean(axis=0) - expected_center)
            center_score = max(0.10, 1.0 - center_distance / math.hypot(width, height))
            # Prefer an outer envelope only while its edges remain supported.
            # A pure min-area box expands a perspective quadrilateral and
            # biases metric PnP depth; the cubic support term keeps a genuinely
            # filled tabletop faithful to its observed corners.
            score = polygon_area * support**3 * center_score
            if score > best_polygon_score:
                best_polygon_score = score
                best_polygon = polygon
        if best_polygon is not None:
            return _order_clockwise(best_polygon), component_area
    raise ValueError("no supported tabletop quadrilateral found")


class TabletopPoseEstimator:
    """Estimate the 0.58 x 0.42 m tabletop pose from one head-left RGB frame."""

    def __init__(
        self,
        calibration: CameraCalibration | None = None,
        *,
        tabletop_length_m: float = 0.58,
        tabletop_depth_m: float = 0.42,
        cad_body_center_z_candidates_m: tuple[float, ...] | None = None,
    ) -> None:
        self.calibration = calibration or CameraCalibration.g1_head_left()
        if tabletop_depth_m <= 0.0 or tabletop_length_m <= tabletop_depth_m:
            raise ValueError("tabletop dimensions must be positive with length > depth")
        self.tabletop_length_m = float(tabletop_length_m)
        self.tabletop_depth_m = float(tabletop_depth_m)
        if cad_body_center_z_candidates_m is None:
            # Keep online CV behaviour fixed to the measured V1 fixture.  The
            # source-RGB calibration path supplies an explicit bounded grid
            # because its floating-base height is not the V1 reset height.
            candidates = (self._TABLE_BODY_CENTER_Z_M,)
        else:
            candidates = tuple(float(value) for value in cad_body_center_z_candidates_m)
            if not candidates or not np.isfinite(candidates).all():
                raise ValueError("cad_body_center_z_candidates_m must be non-empty and finite")
            if min(candidates) < -0.10 or max(candidates) > 0.10:
                raise ValueError("cad_body_center_z_candidates_m must stay within [-0.10, 0.10] m")
        self._cad_body_center_z_candidates_m = tuple(sorted(set(candidates)))

    # Exact V1 assembled-mesh bounds in the Table001_01 body frame. The table
    # is upside down initially, so its local +z tabletop surface faces down in
    # the robot root frame while the local -z legs rise upward.
    _CAD_OUTER_X_M = 0.289993405
    _CAD_OUTER_Y_M = 0.209995389
    _CAD_TABLETOP_Z_M = 0.020159841
    _CAD_LEG_ROOT_Z_M = 0.016341166
    _CAD_LEG_TIP_Z_M = -0.407381144
    # The stationary workbench top is 0.762294 m and the locked G1 root is
    # 0.780000 m above the floor. The assembled body's measured origin is
    # therefore 0.00543 m in the robot-root frame. This is a physical fixture
    # calibration, not an object pose read from the simulator.
    _TABLE_BODY_CENTER_Z_M = 0.00543

    @staticmethod
    def _line_support(mask: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        """Return bright-mask support around a finite image line segment."""

        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        length = float(np.linalg.norm(end - start))
        if length < 8.0:
            return 0.0
        count = max(12, int(round(length / 3.0)))
        fractions = np.linspace(0.0, 1.0, count)[:, None]
        pixels = np.rint((1.0 - fractions) * start + fractions * end).astype(np.int32)
        height, width = mask.shape
        supported = 0
        for x, y in pixels:
            x0, x1 = max(0, x - 3), min(width, x + 4)
            y0, y1 = max(0, y - 3), min(height, y + 4)
            if x0 < x1 and y0 < y1 and np.any(mask[y0:y1, x0:x1] > 0):
                supported += 1
        return float(supported) / float(count)

    @staticmethod
    def _sample_luminance(luminance: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Sample a small, clipped luminance patch at each image point."""

        height, width = luminance.shape
        samples: list[float] = []
        for x, y in np.rint(points).astype(np.int32):
            x0, x1 = max(0, x - 2), min(width, x + 3)
            y0, y1 = max(0, y - 2), min(height, y + 3)
            if x0 >= x1 or y0 >= y1:
                samples.append(0.0)
            else:
                samples.append(float(np.median(luminance[y0:y1, x0:x1])))
        return np.asarray(samples, dtype=np.float64)

    @classmethod
    def _outer_rim_boundary_score(
        cls,
        mask: np.ndarray,
        luminance: np.ndarray,
        corners_px: np.ndarray,
    ) -> float:
        """Score a projected rim by its outside-to-inside visual transition.

        The underside has multiple bright rectangular braces.  A line-distance
        score alone happily locks onto one of those braces.  The real tabletop
        rim is different: moving outward across it reaches the dark workbench
        (or room background), while moving inward reaches the white tabletop
        assembly.  This test uses only the RGB image and is therefore valid for
        both simulation and the physical robot.
        """

        corners = np.asarray(corners_px, dtype=np.float64)
        if corners.shape != (4, 2):
            return 0.0
        center = corners.mean(axis=0)
        edge_scores: list[float] = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            direction = end - start
            length = float(np.linalg.norm(direction))
            if length < 20.0:
                return 0.0
            normal = np.asarray((-direction[1], direction[0]), dtype=np.float64) / length
            midpoint = 0.5 * (start + end)
            # Orient the normal away from the rectangle centre.
            if float(np.dot(normal, midpoint - center)) < 0.0:
                normal *= -1.0
            fractions = np.linspace(0.12, 0.88, max(8, int(length / 20.0)))[:, None]
            rim = (1.0 - fractions) * start + fractions * end
            outside = rim + 9.0 * normal
            inside = rim - 9.0 * normal
            rim_luminance = cls._sample_luminance(luminance, rim)
            outside_luminance = cls._sample_luminance(luminance, outside)
            inside_luminance = cls._sample_luminance(luminance, inside)
            bright_rim = rim_luminance >= 110.0
            exterior_contrast = (rim_luminance - outside_luminance) >= 22.0
            interior_not_darker = inside_luminance >= outside_luminance + 6.0
            # This is deliberately a point-neighborhood test, not
            # ``_line_support``: the latter rejects segments shorter than
            # eight pixels, whereas each rim sample is a single point.
            height, width = mask.shape
            support_values: list[bool] = []
            for x, y in np.rint(rim).astype(np.int32):
                x0, x1 = max(0, x - 4), min(width, x + 5)
                y0, y1 = max(0, y - 4), min(height, y + 5)
                support_values.append(
                    x0 < x1 and y0 < y1 and bool(np.any(mask[y0:y1, x0:x1] > 0))
                )
            support = np.asarray(support_values, dtype=bool)
            edge_scores.append(
                float(np.mean(bright_rim & exterior_contrast & interior_not_darker & support))
            )
        # A rim may be occluded by a leg or hand.  Two clean edges are enough,
        # but an inner brace must not win by explaining only a single segment.
        return float(np.mean(sorted(edge_scores, reverse=True)[:3]))

    def _estimate_from_cad_wireframe(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float] | None:
        """Register the CAD rim and four leg axes directly to RGB evidence.

        A full white component contains braces, rim, and legs, so its convex
        hull alone cannot distinguish the outer tabletop from a smaller inner
        brace.  This fitter renders only the known outer rim and four upright
        leg axes, then minimizes their distance to the RGB segmentation.  It
        uses neither simulator object state nor a learned segmentation label.
        """

        # A white-filled mask gives zero distance everywhere inside the
        # tabletop, so it cannot distinguish the outer rim from a parallel
        # inner brace.  Register the rim to physical RGB edges and retain the
        # white-mask distance only for the narrow leg shafts.
        distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
        grayscale = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
        rgb_edges = cv2.Canny(cv2.GaussianBlur(grayscale, (5, 5), 0), 35, 105)
        edge_distance = cv2.distanceTransform((rgb_edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
        outer_object_corners = np.asarray(
            (
                (-self._CAD_OUTER_X_M, -self._CAD_OUTER_Y_M, self._CAD_TABLETOP_Z_M),
                (self._CAD_OUTER_X_M, -self._CAD_OUTER_Y_M, self._CAD_TABLETOP_Z_M),
                (self._CAD_OUTER_X_M, self._CAD_OUTER_Y_M, self._CAD_TABLETOP_Z_M),
                (-self._CAD_OUTER_X_M, self._CAD_OUTER_Y_M, self._CAD_TABLETOP_Z_M),
            ),
            dtype=np.float64,
        )
        rim_samples: list[np.ndarray] = []
        for start, end in zip(outer_object_corners, np.roll(outer_object_corners, -1, axis=0)):
            rim_samples.extend((1.0 - fraction) * start + fraction * end for fraction in np.linspace(0.0, 1.0, 16))
        leg_samples: list[np.ndarray] = []
        for x in (-TableLegDetector._CAD_LEG_CENTER_X_M, TableLegDetector._CAD_LEG_CENTER_X_M):
            for y in (-TableLegDetector._CAD_LEG_CENTER_Y_M, TableLegDetector._CAD_LEG_CENTER_Y_M):
                leg_samples.extend(
                    np.asarray((x, y, z), dtype=np.float64)
                    for z in np.linspace(self._CAD_LEG_ROOT_Z_M, self._CAD_LEG_TIP_Z_M, 18)
                )
        samples = np.asarray([*rim_samples, *leg_samples], dtype=np.float64)
        rim_sample_count = len(rim_samples)
        camera_from_root = np.linalg.inv(self.calibration.root_from_camera)
        intrinsic = self.calibration.intrinsic_matrix
        height, width = mask.shape

        luminance = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)[..., 0]

        def evaluate(center_x: float, center_y: float, center_z: float, yaw: float):
            cosine, sine = math.cos(yaw), math.sin(yaw)
            root_from_table = np.eye(4, dtype=np.float64)
            # Rz(yaw) @ Rx(pi): long/short axes remain horizontal and the
            # tabletop's local +z normal is downward before the first flip.
            root_from_table[:3, :3] = (
                (cosine, sine, 0.0),
                (sine, -cosine, 0.0),
                (0.0, 0.0, -1.0),
            )
            root_from_table[:3, 3] = (center_x, center_y, center_z)
            points_root = samples @ root_from_table[:3, :3].T + root_from_table[:3, 3]
            points_camera = points_root @ camera_from_root[:3, :3].T + camera_from_root[:3, 3]
            visible = points_camera[:, 2] > 0.05
            pixels = np.empty((len(points_camera), 2), dtype=np.float64)
            pixels[:, 0] = intrinsic[0, 0] * points_camera[:, 0] / points_camera[:, 2] + intrinsic[0, 2]
            pixels[:, 1] = intrinsic[1, 1] * points_camera[:, 1] / points_camera[:, 2] + intrinsic[1, 2]
            visible &= (
                (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] < width)
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] < height)
            )
            if int(np.count_nonzero(visible)) < int(0.80 * len(samples)):
                return None
            integer_pixels = np.rint(pixels[visible]).astype(np.int32)
            integer_pixels[:, 0] = np.clip(integer_pixels[:, 0], 0, width - 1)
            integer_pixels[:, 1] = np.clip(integer_pixels[:, 1], 0, height - 1)
            visible_indices = np.flatnonzero(visible)
            rim_indices = np.flatnonzero(visible_indices < rim_sample_count)
            leg_indices = np.flatnonzero(visible_indices >= rim_sample_count)
            if len(rim_indices) < int(0.80 * rim_sample_count) or not len(leg_indices):
                return None
            mask_errors = distance[integer_pixels[:, 1], integer_pixels[:, 0]]
            rim_errors = edge_distance[integer_pixels[rim_indices, 1], integer_pixels[rim_indices, 0]]
            leg_errors = mask_errors[leg_indices]
            support = float(np.mean(mask_errors <= 5.0))
            # The rim dominates table pose.  The leg term prevents a table-like
            # floor rectangle from winning when a real leg remains visible.
            error = 0.76 * float(np.mean(rim_errors)) + 0.24 * float(np.mean(leg_errors))
            return error, support, root_from_table

        def candidate_score(candidate: tuple[float, float, np.ndarray]) -> tuple[float, float]:
            """Combine geometric support with the outer-rim visual evidence."""

            error, _, frame = candidate
            camera_from_table = camera_from_root @ frame
            rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
            projected, _ = cv2.projectPoints(
                outer_object_corners,
                rotation_vector,
                camera_from_table[:3, 3],
                intrinsic,
                self.calibration.distortion,
            )
            rim_score = self._outer_rim_boundary_score(
                mask,
                luminance,
                projected.reshape(-1, 2),
            )
            # Keep the dense CAD/leg registration primary, but require it to
            # explain an actual outer boundary rather than an inner brace.
            return error - 12.0 * rim_score, rim_score

        # The raw distance field is intentionally permissive: retain several
        # candidates, then use the RGB outside/inside boundary cue to select
        # the actual rim.  Refining only the single raw winner reintroduced the
        # inner-brace local minimum under strong shadows.
        coarse_candidates: list[tuple[float, float, np.ndarray]] = []
        for center_x in np.arange(0.30, 1.06, 0.05):
            for center_y in np.arange(-0.45, 0.46, 0.05):
                for center_z in self._cad_body_center_z_candidates_m:
                    for yaw in np.arange(0.0, 2.0 * math.pi, math.radians(10.0)):
                        result = evaluate(center_x, center_y, center_z, yaw)
                        if result is not None:
                            coarse_candidates.append(result)
        if not coarse_candidates:
            return None
        # Use a bounded set for local refinement.  The later rim score is only
        # evaluated on its best geometric candidates, keeping control updates
        # bounded at the 30 Hz camera rate.
        coarse_candidates.sort(key=lambda item: item[0])
        refined: list[tuple[float, float, np.ndarray]] = []
        for _, _, coarse_frame in coarse_candidates[:4]:
            coarse_center = coarse_frame[:3, 3]
            coarse_yaw = math.atan2(float(coarse_frame[1, 0]), float(coarse_frame[0, 0]))
            refine_z = (
                (float(coarse_center[2]),)
                if len(self._cad_body_center_z_candidates_m) == 1
                else tuple(
                    float(value)
                    for value in np.arange(coarse_center[2] - 0.008, coarse_center[2] + 0.0081, 0.002)
                    if self._cad_body_center_z_candidates_m[0] <= value <= self._cad_body_center_z_candidates_m[-1]
                )
            )
            for center_x in np.arange(coarse_center[0] - 0.04, coarse_center[0] + 0.041, 0.01):
                for center_y in np.arange(coarse_center[1] - 0.04, coarse_center[1] + 0.041, 0.01):
                    for center_z in refine_z:
                        for yaw in np.arange(
                            coarse_yaw - math.radians(10.0),
                            coarse_yaw + math.radians(10.1),
                            math.radians(2.0),
                        ):
                            result = evaluate(center_x, center_y, center_z, yaw)
                            if result is not None:
                                refined.append(result)
        if not refined:
            return None
        refined.sort(key=lambda item: item[0])
        best = min(refined[:64], key=lambda item: candidate_score(item)[0])
        # Finish at 2 mm / 0.5 degree resolution around the selected physical
        # rim rather than allowing the 1 cm grid to visibly cut across it.
        selected_center = best[2][:3, 3]
        selected_yaw = math.atan2(float(best[2][1, 0]), float(best[2][0, 0]))
        precise_z = (
            (float(selected_center[2]),)
            if len(self._cad_body_center_z_candidates_m) == 1
            else tuple(
                float(value)
                for value in np.arange(selected_center[2] - 0.004, selected_center[2] + 0.0041, 0.001)
                if self._cad_body_center_z_candidates_m[0] <= value <= self._cad_body_center_z_candidates_m[-1]
            )
        )
        precise: list[tuple[float, float, np.ndarray]] = []
        for center_x in np.arange(selected_center[0] - 0.008, selected_center[0] + 0.0081, 0.002):
            for center_y in np.arange(selected_center[1] - 0.008, selected_center[1] + 0.0081, 0.002):
                for center_z in precise_z:
                    for yaw in np.arange(
                        selected_yaw - math.radians(2.0),
                        selected_yaw + math.radians(2.01),
                        math.radians(0.5),
                    ):
                        result = evaluate(center_x, center_y, center_z, yaw)
                        if result is not None:
                            precise.append(result)
        # The boundary cue selects the physical rim at the coarse scale.  At
        # the final 2 mm scale, optimize only the geometric registration so a
        # dense re-evaluation of RGB patches cannot stall the 50 Hz servo loop.
        best = min(precise or [best], key=lambda item: item[0])
        error, support, root_from_table = best
        _, rim_score = candidate_score(best)
        # Reject unsupported floor/grid alignments rather than emitting a
        # plausible-looking but false table frame.
        if error > 18.0 or support < 0.42 or rim_score < 0.18:
            return None
        camera_from_table = camera_from_root @ root_from_table
        rotation_vector, _ = cv2.Rodrigues(camera_from_table[:3, :3])
        projected, _ = cv2.projectPoints(
            outer_object_corners,
            rotation_vector,
            camera_from_table[:3, 3],
            intrinsic,
            self.calibration.distortion,
        )
        confidence = float(
            np.clip(support * math.exp(-error / 18.0) * (0.55 + 0.45 * rim_score), 0.0, 1.0)
        )
        return root_from_table, camera_from_table, projected.reshape(-1, 2), error, confidence

    @staticmethod
    def segment_table_assembly(rgb: np.ndarray) -> np.ndarray:
        """Segment white table pixels while preserving narrow leg shafts."""

        image = np.asarray(rgb)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"head-left RGB must be uint8 [480,640,3], got {image.shape} {image.dtype}")
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        chroma = np.linalg.norm(lab[..., 1:3].astype(np.float32) - 128.0, axis=2)
        # First localize the real black workbench in RGB.  Restricting the white
        # target to its convex image footprint prevents bright randomized floors
        # and walls from joining the target component.
        dark = ((lab[..., 0] <= 90) & (hsv[..., 1] <= 180)).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        dark_contours, _ = cv2.findContours(
            dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        workbench_roi = np.zeros(dark.shape, dtype=np.uint8)
        valid_dark_contours = [
            contour
            for contour in dark_contours
            if cv2.contourArea(contour) >= 0.08 * dark.size
        ]
        if valid_dark_contours:
            workbench = max(valid_dark_contours, key=cv2.contourArea)
            cv2.fillConvexPoly(workbench_roi, cv2.convexHull(workbench), 255)
        # Separate the white assembly from the black workbench by the observed
        # bimodal luminance distribution *inside* the detected workbench.  A
        # fixed high ``white`` threshold loses the shaded outside rim while
        # leaving only the bright underside braces; that is exactly the source
        # of the inner-rectangle failure this detector must avoid.  Otsu's
        # threshold adapts to exposure and room lighting but is bounded so a
        # nearly dark/bright DR scene cannot make the decision degenerate.
        workbench_luminance = lab[..., 0][workbench_roi > 0]
        if workbench_luminance.size:
            otsu_luminance, _ = cv2.threshold(
                workbench_luminance.reshape(-1, 1),
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            otsu_luminance = 90.0
        luminance_threshold = int(np.clip(otsu_luminance + 12.0, 70.0, 135.0))
        mask = (
            (lab[..., 0] >= luminance_threshold)
            & (chroma <= 45.0)
            & (hsv[..., 1] <= 125)
        ).astype(np.uint8) * 255
        # The target always lies on the workbench in the central forward view.
        roi = np.zeros_like(mask)
        # The manipulation surface occupies the lower image. Excluding the top
        # band prevents bright randomized floors and walls from becoming the
        # dominant white component while retaining the tabletop and front legs.
        roi[100:405, 20:620] = 255
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.bitwise_and(mask, workbench_roi)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        )
        return mask

    @staticmethod
    def segment_tabletop(rgb: np.ndarray) -> np.ndarray:
        mask = TabletopPoseEstimator.segment_table_assembly(rgb)
        # Remove narrow legs while retaining the broad tabletop/underside body.
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        return mask

    def estimate(self, rgb: np.ndarray) -> TabletopEstimate:
        # First try direct CAD registration against the complete assembly.  A
        # four-corner contour is an unreliable precondition here: the bright
        # underside braces can hide the outer rim exactly in the frames where
        # the CAD model and leg geometry are most useful.
        mask = self.segment_table_assembly(rgb)
        cad_fit = self._estimate_from_cad_wireframe(mask, rgb)
        if cad_fit is not None:
            (
                root_from_table,
                camera_from_table,
                selected_corners,
                reprojection,
                confidence,
            ) = cad_fit
            return TabletopEstimate(
                root_from_table=root_from_table,
                camera_from_table=camera_from_table,
                corners_px=selected_corners,
                mask=mask,
                confidence=confidence,
                reprojection_error_px=reprojection,
                area_fraction=float(np.count_nonzero(mask)) / float(mask.size),
            )

        # ``_quadrilateral_from_mask`` performs its own scale-aware leg
        # suppression and remains as the PnP fallback for a featureless or
        # partially occluded tabletop.
        try:
            corners, contour_area = _quadrilateral_from_mask(mask)
        except ValueError:
            # A featureless synthetic/overexposed view can have no separate
            # narrow shafts. The tabletop-only path remains a valid fallback
            # for that case and preserves the estimator's calibrated contract.
            mask = self.segment_tabletop(rgb)
            corners, contour_area = _quadrilateral_from_mask(mask)
        half_x = 0.5 * self.tabletop_length_m
        half_y = 0.5 * self.tabletop_depth_m
        object_corners = np.asarray(
            ((-half_x, -half_y, 0.0), (half_x, -half_y, 0.0),
             (half_x, half_y, 0.0), (-half_x, half_y, 0.0)),
            dtype=np.float64,
        )

        candidates: list[tuple[float, np.ndarray, float]] = []
        # A rectangle is ambiguous under cyclic ordering and reflection. Test all
        # physical corner assignments, then prefer the pre-flip downward normal.
        ordered_variants = []
        for reverse in (False, True):
            sequence = corners[::-1] if reverse else corners
            for shift in range(4):
                ordered_variants.append(np.roll(sequence, shift, axis=0))
        for image_corners in ordered_variants:
            ok, rvec, tvec = cv2.solvePnP(
                object_corners,
                image_corners,
                self.calibration.intrinsic_matrix,
                self.calibration.distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if not ok or float(tvec[2, 0]) <= 0.15:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            camera_from_table = np.eye(4, dtype=np.float64)
            camera_from_table[:3, :3] = rotation
            camera_from_table[:3, 3] = tvec.reshape(3)
            root_from_table = self.calibration.root_from_camera @ camera_from_table
            projected, _ = cv2.projectPoints(
                object_corners, rvec, tvec,
                self.calibration.intrinsic_matrix, self.calibration.distortion,
            )
            reprojection = float(
                np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image_corners) ** 2, axis=1)))
            )
            normal = root_from_table[:3, 2]
            normal_error = 1.0 - float(np.clip(np.dot(normal, (0.0, 0.0, -1.0)), -1.0, 1.0))
            upright_error = abs(float(root_from_table[2, 0])) + abs(float(root_from_table[2, 1]))
            # Do not bias yaw toward the organizer's original orientation. The
            # 0.58 x 0.42 m rectangle resolves long/short axes through
            # reprojection error and must remain free to cover the full circle.
            score = reprojection + 30.0 * normal_error + 10.0 * upright_error
            candidates.append((score, root_from_table, reprojection, camera_from_table, image_corners))
        if not candidates:
            raise ValueError("PnP found no positive-depth tabletop pose")
        _, root_from_table, reprojection, camera_from_table, selected_corners = min(
            candidates, key=lambda item: item[0]
        )
        area_fraction = contour_area / float(mask.size)
        confidence = float(
            np.clip((area_fraction / 0.12) * math.exp(-reprojection / 12.0), 0.0, 1.0)
        )
        return TabletopEstimate(
            root_from_table=root_from_table,
            camera_from_table=camera_from_table,
            corners_px=selected_corners,
            mask=mask,
            confidence=confidence,
            reprojection_error_px=reprojection,
            area_fraction=area_fraction,
        )

    @staticmethod
    def render_debug(rgb: np.ndarray, estimate: TabletopEstimate | None, error: str = "") -> np.ndarray:
        output = np.asarray(rgb).copy()
        if estimate is not None:
            corners = np.rint(estimate.corners_px).astype(np.int32)
            cv2.polylines(output, [corners], True, (0, 255, 0), 3, cv2.LINE_AA)
            # Image-coordinate corner averaging is not a perspective-invariant
            # centre.  The intersection of the two projected diagonals is the
            # actual tabletop centre and coincides with the central underside
            # brace intersection in the pre-flip assembly.
            first_diagonal = np.cross(
                np.append(estimate.corners_px[0], 1.0),
                np.append(estimate.corners_px[2], 1.0),
            )
            second_diagonal = np.cross(
                np.append(estimate.corners_px[1], 1.0),
                np.append(estimate.corners_px[3], 1.0),
            )
            center_homogeneous = np.cross(first_diagonal, second_diagonal)
            if abs(float(center_homogeneous[2])) > 1.0e-9:
                center_px = center_homogeneous[:2] / center_homogeneous[2]
            else:
                center_px = estimate.corners_px.mean(axis=0)
            center = tuple(np.rint(center_px).astype(int))
            cv2.circle(output, center, 6, (255, 0, 0), -1)
            text = f"conf={estimate.confidence:.2f} reproj={estimate.reprojection_error_px:.1f}px"
            cv2.putText(output, text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if error:
            cv2.putText(output, error[:80], (16, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 64, 64), 2)
        return output
