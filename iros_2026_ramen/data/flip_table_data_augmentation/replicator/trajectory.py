"""Validated access to accepted Mimic trajectories and synchronized 30 Hz samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..provenance import CandidateLedger
from ..io_utils import sha256_file
from ..source_contract import NUMERIC_FEATURES


@dataclass(frozen=True)
class AcceptedTrajectory:
    hdf5_path: Path
    demo_name: str
    candidate_id: str
    source_episode_indices: tuple[int, ...]
    trajectory_seed: int
    trajectory_sha256: str
    config_sha256: str
    runtime_digest: str
    source_frame_count: int


def sample_indices(source_frame_count: int, source_hz: int = 50, target_hz: int = 30) -> np.ndarray:
    """Nearest-neighbor sample indices on an integer rational time grid."""

    for name, value in (
        ("source_frame_count", source_frame_count),
        ("source_hz", source_hz),
        ("target_hz", target_hz),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if target_hz > source_hz:
        raise ValueError("this replay path does not synthesize temporal upsampling")
    target_frame_count = ((source_frame_count - 1) * target_hz) // source_hz + 1
    ordinals = np.arange(target_frame_count, dtype=np.int64)
    # floor(k * source_hz / target_hz + 0.5), implemented exactly with integers.
    indices = (2 * ordinals * source_hz + target_hz) // (2 * target_hz)
    if indices[0] != 0 or indices[-1] >= source_frame_count:
        raise RuntimeError("internal sampling-grid error")
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("sampling grid generated duplicate source frames")
    return indices


def _trajectory_group_sha256(group: h5py.Group) -> str:
    # Keep one implementation in the V1 runtime module; importing it here would
    # pull Isaac Lab into dataset-only validation processes.
    import hashlib

    digest = hashlib.sha256()

    def update(current: h5py.Group, prefix: str) -> None:
        for name in sorted(current.keys()):
            value = current[name]
            path = f"{prefix}/{name}"
            digest.update(path.encode("utf-8"))
            if isinstance(value, h5py.Group):
                digest.update(b"group\0")
                update(value, path)
            else:
                array = np.asarray(value)
                digest.update(b"dataset\0")
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(array).tobytes())

    for name in ("num_samples", "seed", "success"):
        if name in group.attrs:
            digest.update(name.encode("ascii"))
            digest.update(str(group.attrs[name]).encode("utf-8"))
    update(group, "")
    return digest.hexdigest()


def _numeric_group(group: h5py.Group) -> h5py.Group:
    value = group.get("dataset_numeric")
    if not isinstance(value, h5py.Group):
        raise ValueError("accepted trajectory lacks dataset_numeric")
    return value


def _dataset_by_feature(group: h5py.Group, key: str) -> h5py.Dataset:
    direct = group.get(key)
    if isinstance(direct, h5py.Dataset):
        return direct
    current: h5py.Group | h5py.Dataset = group
    for component in key.split("."):
        if not isinstance(current, h5py.Group) or component not in current:
            raise ValueError(f"accepted trajectory lacks numeric feature {key}")
        current = current[component]
    if not isinstance(current, h5py.Dataset):
        raise ValueError(f"numeric feature {key} is not a dataset")
    return current


def _frame_count(group: h5py.Group) -> int:
    lengths = set()
    numeric = _numeric_group(group)
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        dataset = _dataset_by_feature(numeric, key)
        if dataset.ndim != 2 or dataset.shape[1] != width:
            raise ValueError(f"numeric feature {key} has invalid shape {dataset.shape}")
        if not np.issubdtype(dataset.dtype, np.floating):
            raise ValueError(f"numeric feature {key} must be floating point")
        lengths.add(int(dataset.shape[0]))
    if len(lengths) != 1:
        raise ValueError("numeric features are not frame-synchronized")
    frame_count = lengths.pop()
    if frame_count <= 0:
        raise ValueError("accepted trajectory is empty")
    states = group.get("states")
    if not isinstance(states, h5py.Group):
        raise ValueError("accepted trajectory lacks post-step states")
    state_lengths = {
        int(value.shape[0])
        for value in _walk_datasets(states)
        if value.ndim > 0
    }
    if state_lengths != {frame_count}:
        raise ValueError(
            f"post-step states are not synchronized with numeric trace: {sorted(state_lengths)}"
        )
    if int(group.attrs.get("num_samples", -1)) != frame_count:
        raise ValueError("num_samples does not match the accepted trajectory arrays")
    return frame_count


def _walk_datasets(group: h5py.Group):
    for value in group.values():
        if isinstance(value, h5py.Group):
            yield from _walk_datasets(value)
        elif isinstance(value, h5py.Dataset):
            yield value


def inspect_accepted_trajectory(
    hdf5_path: str | Path,
    candidate_id: str,
    ledger: CandidateLedger,
) -> AcceptedTrajectory:
    """Cross-check one HDF5 episode against its validated candidate record."""

    path = Path(hdf5_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    candidate = ledger.load(candidate_id)
    if candidate.status not in {"validated", "rendered", "exported"}:
        raise ValueError(f"candidate {candidate_id} is not physically validated")
    demo_name = candidate.payload.get("accepted_hdf5_demo")
    if not isinstance(demo_name, str) or not demo_name.startswith("demo_"):
        raise ValueError("candidate lacks accepted_hdf5_demo")
    with h5py.File(path, "r") as stream:
        data = stream.get("data")
        if not isinstance(data, h5py.Group) or demo_name not in data:
            raise ValueError(f"accepted HDF5 does not contain {demo_name}")
        group = data[demo_name]
        expected_attrs = {
            "candidate_id": candidate.candidate_id,
            "trajectory_seed": candidate.trajectory_seed,
            "config_sha256": candidate.config_sha256,
            "runtime_digest": candidate.runtime_digest,
            "trajectory_sha256": candidate.payload.get("trajectory_sha256"),
        }
        for name, expected in expected_attrs.items():
            actual = group.attrs.get(name)
            if str(actual) != str(expected):
                raise ValueError(f"{demo_name} attribute {name} differs from the ledger")
        source_indices = tuple(
            int(value) for value in json.loads(group.attrs["source_episode_indices_json"])
        )
        if source_indices != candidate.source_episode_indices:
            raise ValueError("accepted HDF5 source lineage differs from the ledger")
        actual_hash = _trajectory_group_sha256(group)
        if actual_hash != candidate.payload["trajectory_sha256"]:
            raise ValueError("accepted physical trajectory SHA-256 changed")
        frame_count = _frame_count(group)
    return AcceptedTrajectory(
        hdf5_path=path,
        demo_name=demo_name,
        candidate_id=candidate.candidate_id,
        source_episode_indices=candidate.source_episode_indices,
        trajectory_seed=candidate.trajectory_seed,
        trajectory_sha256=candidate.payload["trajectory_sha256"],
        config_sha256=candidate.config_sha256,
        runtime_digest=candidate.runtime_digest,
        source_frame_count=frame_count,
    )


def read_numeric_trace(
    trajectory: AcceptedTrajectory, indices: np.ndarray
) -> dict[str, np.ndarray]:
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= trajectory.source_frame_count):
        raise ValueError("numeric sample indices are out of range")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("numeric sample indices must be strictly increasing")
    output = {}
    with h5py.File(trajectory.hdf5_path, "r") as stream:
        numeric = _numeric_group(stream["data"][trajectory.demo_name])
        for key, (_dtype, width) in NUMERIC_FEATURES.items():
            array = np.asarray(_dataset_by_feature(numeric, key)[indices], dtype=np.float32)
            if array.shape != (len(indices), width) or not np.isfinite(array).all():
                raise ValueError(f"sampled numeric feature {key} is invalid")
            output[key] = array
    return output


def read_state_at(trajectory: AcceptedTrajectory, frame_index: int) -> dict[str, Any]:
    """Read one nested Isaac Lab post-step state as NumPy arrays."""

    if frame_index < 0 or frame_index >= trajectory.source_frame_count:
        raise IndexError(frame_index)

    def read(group: h5py.Group) -> dict[str, Any]:
        output = {}
        for name, value in group.items():
            output[name] = read(value) if isinstance(value, h5py.Group) else np.asarray(value[frame_index])
        return output

    with h5py.File(trajectory.hdf5_path, "r") as stream:
        return read(stream["data"][trajectory.demo_name]["states"])


def write_numeric_parquet(path: str | Path, trace: dict[str, np.ndarray]) -> str:
    """Write source-compatible float32 fixed-size lists and return file SHA-256."""

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write the numeric replay trace") from exc
    if tuple(trace) != tuple(NUMERIC_FEATURES):
        raise ValueError("numeric trace keys or order differ from the source contract")
    arrays = {}
    row_count = None
    for key, (_dtype, width) in NUMERIC_FEATURES.items():
        value = np.asarray(trace[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"numeric trace feature {key} is invalid")
        if row_count is None:
            row_count = value.shape[0]
        elif value.shape[0] != row_count:
            raise ValueError("numeric trace features have different row counts")
        flat = pa.array(value.reshape(-1), type=pa.float32())
        arrays[key] = pa.FixedSizeListArray.from_arrays(flat, width)
    if not row_count:
        raise ValueError("numeric trace must contain at least one frame")
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    pq.write_table(pa.table(arrays), temporary, compression="zstd")
    temporary.replace(output)
    return sha256_file(output)
