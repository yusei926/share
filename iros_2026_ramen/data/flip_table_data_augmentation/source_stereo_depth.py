"""Deterministic metric depth from the pinned source head-stereo calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from model.flip_table_reinforcement_learning.teacher.source_stereo_calibration import (
    StereoCalibration,
)


STEREO_DEPTH_SCHEMA_VERSION = "team_ramen_fast_foundationstereo_depth/v2"


@dataclass(frozen=True)
class FastFoundationStereoParameters:
    valid_iterations: int = 8
    max_disparity_px: int = 192
    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 2.0
    optimize_build_volume: str = "pytorch1"
    normalize_feature_volume: bool = True
    maximum_left_right_error_px: float = 1.5

    def validate(self) -> None:
        if self.valid_iterations <= 0:
            raise ValueError("valid_iterations must be positive")
        if self.max_disparity_px <= 0 or self.max_disparity_px % 32:
            raise ValueError("max_disparity_px must be a positive multiple of 32")
        if not 0.0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("stereo depth range must be positive and ordered")
        if self.optimize_build_volume != "pytorch1":
            raise ValueError("release inference requires the audited pytorch1 volume builder")
        if not self.normalize_feature_volume:
            raise ValueError("the pinned model requires normalized feature volumes")
        if not 0.0 < self.maximum_left_right_error_px <= 5.0:
            raise ValueError("maximum_left_right_error_px must be in (0, 5]")


def left_right_consistency_mask(
    left_disparity_px: np.ndarray,
    right_disparity_px: np.ndarray,
    *,
    maximum_error_px: float,
) -> np.ndarray:
    """Validate a left disparity against a positive right-coordinate disparity."""

    left = np.asarray(left_disparity_px, dtype=np.float32)
    right = np.asarray(right_disparity_px, dtype=np.float32)
    if (
        left.ndim != 2
        or right.shape != left.shape
        or not math.isfinite(maximum_error_px)
        or maximum_error_px <= 0.0
    ):
        raise ValueError("left-right consistency inputs are invalid")
    height, width = left.shape
    columns = np.broadcast_to(
        np.arange(width, dtype=np.float32)[None, :], (height, width)
    )
    corresponding_x = columns - left
    valid = (
        np.isfinite(left)
        & (left > 0.0)
        & np.isfinite(corresponding_x)
        & (corresponding_x >= 0.0)
        & (corresponding_x <= width - 1)
    )
    finite_x = np.where(np.isfinite(corresponding_x), corresponding_x, 0.0)
    lower_x = np.floor(finite_x).astype(np.int64)
    lower_x = np.clip(lower_x, 0, width - 1)
    upper_x = np.minimum(lower_x + 1, width - 1)
    weight = finite_x - lower_x
    rows = np.arange(height, dtype=np.int64)[:, None]
    sampled_right = (
        (1.0 - weight) * right[rows, lower_x]
        + weight * right[rows, upper_x]
    )
    return (
        valid
        & np.isfinite(sampled_right)
        & (sampled_right > 0.0)
        & (np.abs(left - sampled_right) <= maximum_error_px)
    )


class FastFoundationStereoDepthEstimator:
    """Metric depth from the pinned NVIDIA Fast FoundationStereo checkpoint."""

    def __init__(
        self,
        calibration: StereoCalibration,
        *,
        source_root: Path,
        model_path: Path,
        image_size: tuple[int, int] = (640, 480),
        parameters: FastFoundationStereoParameters = FastFoundationStereoParameters(),
        device: str = "cuda",
    ):
        parameters.validate()
        self.calibration = calibration
        self.width, self.height = image_size
        if (self.width, self.height) != (640, 480):
            raise ValueError("source head stereo must remain 640x480")
        self.parameters = parameters
        self.source_root = Path(source_root).resolve(strict=True)
        self.model_path = Path(model_path).resolve(strict=True)
        self.intrinsic_matrix = np.asarray(calibration.projection_left[:, :3], dtype=np.float64)
        focal = float(self.intrinsic_matrix[0, 0])
        if focal <= 0.0 or not math.isclose(
            focal, float(self.intrinsic_matrix[1, 1]), rel_tol=1e-9
        ):
            raise ValueError("rectified stereo projection must use square pixels")
        self.baseline_m = abs(
            float(calibration.projection_right[0, 3] - calibration.projection_left[0, 3])
        ) / focal
        if not 0.04 <= self.baseline_m <= 0.08:
            raise ValueError("source head stereo baseline lies outside the physical rig range")
        size = (self.width, self.height)
        self._left_maps = cv2.initUndistortRectifyMap(
            calibration.camera_matrix_left,
            calibration.dist_coeffs_left,
            calibration.rectification_left,
            self.intrinsic_matrix,
            size,
            cv2.CV_32FC1,
        )
        self._right_maps = cv2.initUndistortRectifyMap(
            calibration.camera_matrix_right,
            calibration.dist_coeffs_right,
            calibration.rectification_right,
            np.asarray(calibration.projection_right[:, :3], dtype=np.float64),
            size,
            cv2.CV_32FC1,
        )
        self._torch, self._padder_type, amp_dtype = self._load_runtime_modules()
        if device != "cuda" or not self._torch.cuda.is_available():
            raise ValueError("release Fast FoundationStereo inference requires CUDA")
        self.device = self._torch.device(device)
        self._amp_dtype = amp_dtype
        self._torch.manual_seed(0)
        self._torch.cuda.manual_seed_all(0)
        self._torch.set_grad_enabled(False)
        self._model = self._torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        self._model.args.valid_iters = parameters.valid_iterations
        self._model.args.max_disp = parameters.max_disparity_px
        # The serialized February 2026 checkpoint predates this explicit field.
        # NVIDIA's current ONNX path defines the missing value as True.
        self._model.args.normalize = parameters.normalize_feature_volume
        self._model.to(self.device).eval()

    def _load_runtime_modules(self):
        for module_name in ("core", "Utils"):
            module = sys.modules.get(module_name)
            module_file = None if module is None else getattr(module, "__file__", None)
            if module_file is not None and self.source_root not in Path(module_file).resolve().parents:
                raise RuntimeError(
                    f"module {module_name} was loaded from a conflicting runtime: {module_file}"
                )
        source = str(self.source_root)
        if source not in sys.path:
            sys.path.insert(0, source)
        try:
            from core.utils.utils import InputPadder
            from Utils import AMP_DTYPE
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "pinned Fast FoundationStereo source or dependencies are unavailable"
            ) from exc
        return torch, InputPadder, AMP_DTYPE

    def _image(self, value: np.ndarray, label: str) -> np.ndarray:
        image = np.asarray(value)
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(f"{label} must be uint8 {(self.height, self.width, 3)} RGB")
        return image

    def rectify(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left = self._image(left_rgb, "left_rgb")
        right = self._image(right_rgb, "right_rgb")
        return (
            cv2.remap(left, *self._left_maps, interpolation=cv2.INTER_LINEAR),
            cv2.remap(right, *self._right_maps, interpolation=cv2.INTER_LINEAR),
        )

    def estimate(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return rectified RGB, metric depth, and auditable diagnostics."""

        rectified, depth, diagnostics, _ = self.estimate_with_confidence(
            left_rgb, right_rgb
        )
        return rectified, depth, diagnostics

    def _infer_disparity(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray
    ) -> np.ndarray:
        left_tensor = (
            self._torch.as_tensor(left_rgb, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
        )
        right_tensor = (
            self._torch.as_tensor(right_rgb, device=self.device)
            .float()[None]
            .permute(0, 3, 1, 2)
        )
        padder = self._padder_type(left_tensor.shape, divis_by=32, force_square=False)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        with self._torch.inference_mode(), self._torch.amp.autocast(
            "cuda", enabled=True, dtype=self._amp_dtype
        ):
            disparity = self._model.forward(
                left_tensor,
                right_tensor,
                iters=self.parameters.valid_iterations,
                test_mode=True,
                optimize_build_volume=self.parameters.optimize_build_volume,
            )
        return (
            padder.unpad(disparity.float())
            .detach()
            .cpu()
            .numpy()
            .reshape(self.height, self.width)
        )

    def estimate_with_confidence(
        self, left_rgb: np.ndarray, right_rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
        """Return depth plus a per-pixel left-right stereo consistency mask."""

        left, right = self.rectify(left_rgb, right_rgb)
        disparity = self._infer_disparity(left, right)
        reverse_disparity = np.flip(
            self._infer_disparity(
                np.ascontiguousarray(np.flip(right, axis=1)),
                np.ascontiguousarray(np.flip(left, axis=1)),
            ),
            axis=1,
        )
        consistency = left_right_consistency_mask(
            disparity,
            reverse_disparity,
            maximum_error_px=self.parameters.maximum_left_right_error_px,
        )
        columns = np.arange(self.width, dtype=np.float32)[None, :]
        valid_disparity = np.isfinite(disparity) & (disparity > 0.0) & (
            columns - disparity >= 0.0
        )
        depth = np.zeros(disparity.shape, dtype=np.float32)
        focal = float(self.intrinsic_matrix[0, 0])
        depth[valid_disparity] = focal * self.baseline_m / disparity[valid_disparity]
        valid = (
            valid_disparity
            & np.isfinite(depth)
            & (depth >= self.parameters.minimum_depth_m)
            & (depth <= self.parameters.maximum_depth_m)
        )
        depth[~valid] = 0.0
        positive = depth[valid]
        valid_count = int(valid.sum())
        consistent_valid = consistency & valid
        diagnostics = {
            "schema_version": STEREO_DEPTH_SCHEMA_VERSION,
            "backend": "NVlabs/Fast-FoundationStereo",
            "valid_fraction": float(valid.mean()),
            "valid_pixel_count": valid_count,
            "left_right_consistent_pixel_count": int(consistent_valid.sum()),
            "left_right_consistent_fraction_of_valid_depth": (
                float(consistent_valid.sum() / valid_count) if valid_count else 0.0
            ),
            "maximum_left_right_error_px": (
                self.parameters.maximum_left_right_error_px
            ),
            "baseline_m": self.baseline_m,
            "intrinsic_matrix_px": self.intrinsic_matrix.reshape(-1).tolist(),
            "depth_p05_m": float(np.percentile(positive, 5)) if positive.size else None,
            "depth_median_m": float(np.median(positive)) if positive.size else None,
            "depth_p95_m": float(np.percentile(positive, 95)) if positive.size else None,
            "valid_iterations": self.parameters.valid_iterations,
            "max_disparity_px": self.parameters.max_disparity_px,
            "normalize_feature_volume": self.parameters.normalize_feature_volume,
            "optimize_build_volume": self.parameters.optimize_build_volume,
        }
        return left, depth, diagnostics, consistency


def depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0.0):
        raise ValueError("depth_m must be a finite non-negative image")
    if float(depth.max(initial=0.0)) >= np.iinfo(np.uint16).max / 1000.0:
        raise ValueError("depth exceeds uint16 millimetre storage")
    return np.rint(depth * 1000.0).astype(np.uint16)
