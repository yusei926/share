"""Pin and audit the raw-record provenance behind the LeRobot source slices."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .config import PipelineConfig
from .io_utils import sha256_file


RAW_BINDING_SCHEMA_VERSION = "team_ramen_flip_table_raw_binding_audit/v1"
_TIME_TOLERANCE_S = 1.0e-9


def snapshot_download_raw_contract(
    config: PipelineConfig,
    *,
    local_dir: Path | None = None,
) -> Path:
    """Download only raw annotations and camera calibrations at the pinned revision."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download raw calibration metadata") from exc
    path = snapshot_download(
        repo_id=config.raw_source.repo_id,
        repo_type="dataset",
        revision=config.raw_source.revision,
        local_dir=None if local_dir is None else str(local_dir),
        allow_patterns=(
            "*/info.json",
            "*/calibration/params/head_camera_params.yaml",
            "*/calibration/params/camera_*.json",
        ),
    )
    return Path(path).resolve()


def _raw_info_paths(raw_root: Path, config: PipelineConfig) -> list[Path]:
    paths = sorted(raw_root.glob("episode_*/episode_*/info.json"))
    if len(paths) != config.raw_source.episodes:
        raise ValueError(
            f"raw source has {len(paths)} info files, expected {config.raw_source.episodes}"
        )
    return paths


def _source_episode_rows(source_root: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to audit source episode metadata") from exc
    paths = sorted((source_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(source_root / "meta" / "episodes")
    rows = pq.read_table([str(path) for path in paths]).to_pylist()
    rows.sort(key=lambda row: int(row["episode_index"]))
    indices = [int(row["episode_index"]) for row in rows]
    if indices != list(range(len(rows))) or len(rows) != config.source.episodes:
        raise ValueError("LeRobot source episode metadata is incomplete or non-contiguous")
    return rows


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _flip_interval(info: dict[str, Any]) -> tuple[float, float] | None:
    start_ns = int(info["start_timestamp_ns"])
    end_ns = int(info["end_timestamp_ns"])
    annotations = sorted(info.get("subtasks", []), key=lambda item: int(item["timestamp_ns"]))
    matches: list[tuple[float, float]] = []
    for index, annotation in enumerate(annotations):
        if annotation.get("task") != "flip table":
            continue
        annotation_start_ns = int(annotation["timestamp_ns"])
        annotation_end_ns = (
            int(annotations[index + 1]["timestamp_ns"])
            if index + 1 < len(annotations)
            else end_ns
        )
        if not start_ns <= annotation_start_ns < annotation_end_ns <= end_ns:
            raise ValueError("raw flip-table annotation lies outside its episode")
        matches.append(
            (
                (annotation_start_ns - start_ns) / 1_000_000_000.0,
                (annotation_end_ns - start_ns) / 1_000_000_000.0,
            )
        )
    if len(matches) > 1:
        raise ValueError("a raw episode contains more than one flip-table interval")
    return matches[0] if matches else None


def _validate_head_calibration(path: Path, config: PipelineConfig) -> dict[str, Any]:
    digest = sha256_file(path)
    if digest != config.raw_source.head_stereo_calibration_sha256:
        raise ValueError(f"unexpected head-stereo calibration content: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError(f"head-stereo calibration is not successful: {path}")
    if payload.get("image_size") != [640, 480]:
        raise ValueError(f"head-stereo calibration is not 640x480: {path}")
    baseline_mm = float(payload.get("baseline", math.nan))
    rms_px = float(payload.get("rms_error", math.nan))
    expected_baseline_mm = 1000.0 * config.raw_source.head_stereo_baseline_m
    if not math.isfinite(baseline_mm) or not math.isclose(
        baseline_mm, expected_baseline_mm, abs_tol=1.0e-9
    ):
        raise ValueError(f"head-stereo baseline differs from the pinned calibration: {path}")
    if not math.isfinite(rms_px) or not math.isclose(
        rms_px, config.raw_source.head_stereo_rms_error_px, abs_tol=1.0e-12
    ):
        raise ValueError(f"head-stereo RMS differs from the pinned calibration: {path}")
    return {
        "relative_path": str(path),
        "sha256": digest,
        "image_size": [640, 480],
        "baseline_mm": baseline_mm,
        "rms_error_px": rms_px,
    }


def _validate_wrist_calibrations(directory: Path, config: PipelineConfig) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("camera_*.json"))
    if len(paths) != 2:
        raise ValueError(f"expected exactly two D405 calibration files in {directory}")
    records: list[dict[str, Any]] = []
    found_serials: set[str] = set()
    for path in paths:
        payload = _load_object(path, "D405 calibration")
        serial = str(payload.get("serial_number", ""))
        expected = config.raw_source.wrist_calibration_sha256_by_serial.get(serial)
        digest = sha256_file(path)
        if expected is None or digest != expected or path.name != f"camera_{serial}.json":
            raise ValueError(f"unexpected D405 calibration identity: {path}")
        color = payload.get("color")
        intrinsics = color.get("intrinsics") if isinstance(color, dict) else None
        if not isinstance(intrinsics, dict):
            raise ValueError(f"D405 color intrinsics are missing: {path}")
        if (intrinsics.get("width"), intrinsics.get("height")) != (640, 480):
            raise ValueError(f"D405 color calibration is not 640x480: {path}")
        focal = (float(intrinsics.get("fx", math.nan)), float(intrinsics.get("fy", math.nan)))
        principal = (float(intrinsics.get("ppx", math.nan)), float(intrinsics.get("ppy", math.nan)))
        coefficients = intrinsics.get("coeffs")
        if (
            not all(math.isfinite(value) and value > 0.0 for value in focal + principal)
            or not isinstance(coefficients, list)
            or len(coefficients) != 5
            or not all(math.isfinite(float(value)) for value in coefficients)
        ):
            raise ValueError(f"D405 color intrinsics are invalid: {path}")
        found_serials.add(serial)
        records.append(
            {
                "relative_path": str(path),
                "serial_number": serial,
                "sha256": digest,
                "image_size": [640, 480],
                "fx_fy_px": list(focal),
                "ppx_ppy_px": list(principal),
                "distortion_model": intrinsics.get("model"),
                "distortion_coefficients": [float(value) for value in coefficients],
            }
        )
    if found_serials != set(config.raw_source.wrist_calibration_sha256_by_serial):
        raise ValueError("raw D405 serial set does not match the pinned source contract")
    return records


def _episode_provenance(
    *,
    raw_root: Path,
    raw_info_path: Path,
    raw_episode_index: int,
    source_row: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    info = _load_object(raw_info_path, "raw episode info")
    interval = _flip_interval(info)
    if interval is None:
        raise ValueError(f"source episode points to a raw record without a flip label: {raw_info_path}")
    source_index = int(source_row["source_episode_index"])
    if source_index != raw_episode_index:
        raise ValueError("source_episode_index does not match the sorted raw info-file index")
    if str(source_row["source_episode_name"]) != str(info.get("episode_name")):
        raise ValueError("source_episode_name does not match raw info.json")
    start_s, end_s = interval
    if not math.isclose(float(source_row["source_start_sec"]), start_s, abs_tol=_TIME_TOLERANCE_S):
        raise ValueError("source flip start time does not match raw annotation")
    if not math.isclose(float(source_row["source_end_sec"]), end_s, abs_tol=_TIME_TOLERANCE_S):
        raise ValueError("source flip end time does not match raw annotation")

    calibration_dir = raw_info_path.parent / "calibration" / "params"
    head_path = calibration_dir / "head_camera_params.yaml"
    if not head_path.is_file():
        raise FileNotFoundError(head_path)
    head = _validate_head_calibration(head_path, config)
    wrists = _validate_wrist_calibrations(calibration_dir, config)
    for record in (head, *wrists):
        record["relative_path"] = str(Path(record["relative_path"]).relative_to(raw_root))
    return {
        "raw_episode_index": raw_episode_index,
        "raw_info": {
            "relative_path": str(raw_info_path.relative_to(raw_root)),
            "sha256": sha256_file(raw_info_path),
        },
        "raw_episode_name": str(info["episode_name"]),
        "flip_interval_sec": [start_s, end_s],
        "head_stereo_calibration": head,
        "wrist_d405_calibrations": wrists,
    }


def audit_raw_source_bindings(
    raw_root: str | Path,
    source_root: str | Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    """Prove that every source slice maps to one pinned raw flip annotation."""

    raw = Path(raw_root).expanduser().resolve()
    source = Path(source_root).expanduser().resolve()
    raw_paths = _raw_info_paths(raw, config)
    source_rows = _source_episode_rows(source, config)
    by_raw_index: dict[int, dict[str, Any]] = {}
    for row in source_rows:
        raw_index = int(row["source_episode_index"])
        if raw_index in by_raw_index:
            raise ValueError(f"multiple source episodes map to raw episode {raw_index}")
        by_raw_index[raw_index] = row

    bindings: list[dict[str, Any]] = []
    omitted: list[dict[str, Any]] = []
    head_hashes: Counter[str] = Counter()
    wrist_hashes: Counter[str] = Counter()
    for raw_index, info_path in enumerate(raw_paths):
        info = _load_object(info_path, "raw episode info")
        interval = _flip_interval(info)
        source_row = by_raw_index.get(raw_index)
        if interval is None:
            if source_row is not None:
                raise ValueError(f"raw episode {raw_index} has no flip label but appears in source")
            omitted.append(
                {
                    "raw_episode_index": raw_index,
                    "raw_info_relative_path": str(info_path.relative_to(raw)),
                    "reason": "no_flip_table_annotation",
                }
            )
            continue
        if source_row is None:
            raise ValueError(f"raw episode {raw_index} has a flip label but is absent from source")
        provenance = _episode_provenance(
            raw_root=raw,
            raw_info_path=info_path,
            raw_episode_index=raw_index,
            source_row=source_row,
            config=config,
        )
        head_hashes[provenance["head_stereo_calibration"]["sha256"]] += 1
        for wrist in provenance["wrist_d405_calibrations"]:
            wrist_hashes[wrist["sha256"]] += 1
        bindings.append(
            {
                "episode_index": int(source_row["episode_index"]),
                "source_episode_index": raw_index,
                **provenance,
            }
        )

    if len(bindings) != config.source.episodes or len(bindings) + len(omitted) != config.raw_source.episodes:
        raise ValueError("raw/source binding cardinality is inconsistent")
    return {
        "schema_version": RAW_BINDING_SCHEMA_VERSION,
        "source_repo_id": config.source.repo_id,
        "source_revision": config.source.revision,
        "raw_repo_id": config.raw_source.repo_id,
        "raw_revision": config.raw_source.revision,
        "config_sha256": config.digest,
        "passed": True,
        "counts": {
            "raw_episodes": len(raw_paths),
            "source_flip_episodes": len(bindings),
            "omitted_without_flip": len(omitted),
            "head_calibration_variants": len(head_hashes),
            "wrist_calibration_variants": len(wrist_hashes),
        },
        "head_calibration_hash_counts": dict(sorted(head_hashes.items())),
        "wrist_calibration_hash_counts": dict(sorted(wrist_hashes.items())),
        "omitted_raw_episodes": omitted,
        "bindings": bindings,
    }
