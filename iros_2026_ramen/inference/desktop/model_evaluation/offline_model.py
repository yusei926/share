"""Run a sealed GPU model on a recorded observation bundle without DDS."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from inference.desktop.upper_policy.worker_protocol import (
    receive_message,
    send_message,
)

from .adapters import CanonicalObservation, adapter_for
from .artifacts import checkpoint_path, validate_prepared_artifacts
from .registry import ModelSpec


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PYTHON = REPO_ROOT / "model/subtask_policy_training/.venv/bin/python"


def load_bundle(path: Path, spec: ModelSpec) -> CanonicalObservation:
    root = path.expanduser().resolve()
    document = json.loads((root / "observation.json").read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("observation.json must contain an object")
    camera_files = document.get("camera_jpeg")
    if not isinstance(camera_files, Mapping):
        raise ValueError("observation.json camera_jpeg must map roles to files")
    if set(camera_files) != set(spec.camera_roles):
        raise ValueError(
            f"bundle camera roles must be exactly {spec.camera_roles}, "
            f"got {sorted(camera_files)}"
        )
    cameras: dict[str, bytes] = {}
    for role, relative in camera_files.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents:
            raise ValueError(f"camera path escapes bundle: {relative}")
        payload = candidate.read_bytes()
        if not payload:
            raise ValueError(f"empty camera payload: {role}")
        cameras[str(role)] = payload
    return CanonicalObservation(
        body_joint_position_rad=np.asarray(
            document.get("body_joint_position_rad"), dtype=np.float64
        ),
        dex1_opening_fraction=np.asarray(
            document.get("dex1_opening_fraction"), dtype=np.float64
        ),
        eef_xyz_euler=(
            None
            if document.get("eef_xyz_euler") is None
            else np.asarray(document.get("eef_xyz_euler"), dtype=np.float64)
        ),
        camera_jpeg=cameras,
    )


def run_offline_model(
    spec: ModelSpec,
    *,
    local_dir: Path,
    bundle: Path,
    device: str,
    seed: int = 42,
) -> dict[str, Any]:
    """Load and infer once. No Unitree/CycloneDDS/camera transport is imported."""
    validate_prepared_artifacts(local_dir, spec)
    if not MODEL_PYTHON.is_file():
        raise FileNotFoundError("model/subtask_policy_training/.venv is unavailable")
    observation = load_bundle(bundle, spec)
    adapter = adapter_for(spec)
    state = adapter.model_state(observation)
    argv = [
        str(MODEL_PYTHON),
        str(REPO_ROOT / spec.worker),
        "--checkpoint",
        str(checkpoint_path(local_dir, spec)),
        "--device",
        device,
        "--seed",
        str(seed),
        "--model-repo-id",
        spec.repo_id,
        "--model-revision",
        spec.revision,
        "--task",
        spec.task,
    ]
    if spec.expected_model_sha256 is not None:
        argv += ["--expected-model-sha256", spec.expected_model_sha256]
    process = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        ready = receive_message(process.stdout)
        if ready.get("type") != "ready":
            raise RuntimeError(f"model worker did not become ready: {ready}")
        contract = ready.get("contract") or {}
        expected_identity = {
            "model_repo_id": spec.repo_id,
            "model_revision": spec.revision,
            "task": spec.task,
        }
        mismatch = {
            key: (contract.get(key), value)
            for key, value in expected_identity.items()
            if contract.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"model worker identity mismatch: {mismatch}")
        request = adapter.offline_request(observation, state)
        send_message(process.stdin, request)
        response = receive_message(process.stdout)
        if response.get("type") == "error":
            raise RuntimeError(f"model worker failed: {response.get('error')}")
        if response.get("type") != "prediction" or response.get("request_id") != 1:
            raise RuntimeError(f"unexpected model response: {response}")
        native = np.asarray(response.get("actions"), dtype=np.float64)
        canonical = adapter.canonical_action(native, observation)
        send_message(process.stdin, {"type": "close"})
        closed = receive_message(process.stdout)
        if closed.get("type") != "closed":
            raise RuntimeError(f"model worker did not close cleanly: {closed}")
        process.wait(timeout=10)
    except Exception:
        process.terminate()
        process.wait(timeout=10)
        raise
    forbidden = sorted(
        name
        for name in sys.modules
        if name.startswith(("unitree_sdk2py", "cyclonedds"))
    )
    if forbidden:
        raise RuntimeError(f"offline model dry-run imported transport: {forbidden}")
    return {
        "model_id": spec.model_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "family": spec.family,
        "state_shape": list(state.shape),
        "native_action_shape": list(native.shape),
        "canonical_action_shape": list(canonical.shape),
        "canonical_arm_min_rad": float(canonical[:, :14].min()),
        "canonical_arm_max_rad": float(canonical[:, :14].max()),
        "canonical_dex1_min_fraction": float(canonical[:, 14:].min()),
        "canonical_dex1_max_fraction": float(canonical[:, 14:].max()),
        "inference_ms": float(response["inference_ms"]),
        "model_weights_loaded": True,
        "robot_command_sent": False,
        "dds_initialized": False,
        "physical_transport_imported": False,
    }


def _request(
    spec: ModelSpec,
    observation: CanonicalObservation,
    state: np.ndarray,
) -> dict[str, Any]:
    return adapter_for(spec).offline_request(observation, state)
