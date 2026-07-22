#!/usr/bin/env python3
"""Verify compiled object-pose dependencies and record their exact runtime."""

from __future__ import annotations

import argparse
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.config import DEFAULT_CONFIG_PATH, load_pipeline_config
from data.flip_table_data_augmentation.io_utils import atomic_write_json, sha256_file


SCHEMA_VERSION = "team_ramen_object_pose_compiled_runtime/v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--observed-image-digest", required=True)
    return parser.parse_args()


def _revision(path: Path) -> str:
    repository = path.resolve()
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def main() -> None:
    args = parse_args()

    import cv2
    import iopath
    import kornia
    import kornia_rs
    import nvdiffrast
    import pybind11
    import pytorch3d
    import ruamel.yaml
    import torch
    import transformations
    import transformers
    import timm
    import mycpp

    config = load_pipeline_config(args.config)
    pose = config.object_pose_runtime
    root = args.runtime_root.resolve()
    if args.observed_image_digest != config.runtime.container_digest:
        raise ValueError("compiled object-pose runtime uses a different V1 image")
    versions = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "opencv": cv2.__version__,
        "transformers": distribution_version("transformers"),
        "kornia": distribution_version("kornia"),
        "kornia-rs": distribution_version("kornia-rs"),
        "iopath": distribution_version("iopath"),
        "pytorch3d": distribution_version("pytorch3d"),
        "nvdiffrast": distribution_version("nvdiffrast"),
        "pybind11": distribution_version("pybind11"),
        "ruamel.yaml": distribution_version("ruamel.yaml"),
        "transformations": distribution_version("transformations"),
        "timm": distribution_version("timm"),
    }
    expected = {
        "torch": "2.10.0+cu128",
        "torch_cuda": "12.8",
        "transformers": pose.transformers_version,
        "kornia": "0.8.3",
        "kornia-rs": "0.1.14",
        "iopath": "0.1.10",
        "pytorch3d": "0.7.9",
        "nvdiffrast": "0.4.0",
        "pybind11": "3.0.4",
        "ruamel.yaml": "0.19.1",
        "transformations": "2026.1.18",
        "timm": pose.timm_version,
    }
    mismatches = {
        name: {"expected": value, "observed": versions.get(name)}
        for name, value in expected.items()
        if versions.get(name) != value
    }
    repositories = {
        "FoundationPose": _revision(root / "FoundationPose"),
        "pytorch3d": _revision(root / "pytorch3d"),
        "nvdiffrast": _revision(root / "nvdiffrast"),
        "Fast-FoundationStereo": _revision(root / "Fast-FoundationStereo"),
    }
    expected_repositories = {
        "FoundationPose": pose.foundationpose_revision,
        "pytorch3d": pose.pytorch3d_revision,
        "nvdiffrast": pose.nvdiffrast_revision,
        "Fast-FoundationStereo": pose.fast_stereo_revision,
    }
    if repositories != expected_repositories:
        raise ValueError(f"compiled source revisions differ: {repositories}")
    if mismatches:
        raise ValueError(f"compiled dependency versions differ: {mismatches}")
    stereo_files = {
        pose.fast_stereo_model_filename: pose.fast_stereo_model_sha256,
        pose.fast_stereo_config_filename: pose.fast_stereo_config_sha256,
    }
    stereo_root = root / "hf" / "fast-foundation-stereo"
    observed_stereo_files = {
        name: sha256_file(stereo_root / name) for name in stereo_files
    }
    if observed_stereo_files != stereo_files:
        raise ValueError(f"Fast FoundationStereo artifacts differ: {observed_stereo_files}")
    source_manifest = root / "runtime-manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config.digest,
        "source_runtime_manifest_sha256": sha256_file(source_manifest),
        "container_digest": args.observed_image_digest,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "versions": versions,
        "repositories": repositories,
        "fast_foundationstereo": {
            "model_repo": pose.fast_stereo_model_repo,
            "model_revision": pose.fast_stereo_model_revision,
            "files": observed_stereo_files,
            "valid_iterations": pose.fast_stereo_valid_iterations,
            "max_disparity_px": pose.fast_stereo_max_disparity_px,
            "normalize_feature_volume": True,
        },
    }
    path = root / "compiled-runtime-manifest.json"
    atomic_write_json(path, payload)
    print(json.dumps({"manifest": str(path), "cuda": payload["cuda_device_name"]}, sort_keys=True))


if __name__ == "__main__":
    main()
