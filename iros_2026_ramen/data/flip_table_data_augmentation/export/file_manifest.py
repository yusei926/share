"""Content-addressed file manifests used before and after Hub upload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..io_utils import atomic_write_json, sha256_file


FILE_MANIFEST_SCHEMA_VERSION = "team_ramen_dataset_file_manifest/v1"
DEFAULT_MANIFEST_NAME = "dataset_file_manifest.json"


def build_file_manifest(
    root: str | Path,
    *,
    excluded_relative_paths: Iterable[str] = (DEFAULT_MANIFEST_NAME,),
) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    excluded = set(excluded_relative_paths)
    files = []
    total_bytes = 0
    for path in sorted(item for item in dataset_root.rglob("*") if item.is_file()):
        relative = path.relative_to(dataset_root).as_posix()
        if relative in excluded:
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": relative, "size_bytes": size, "sha256": sha256_file(path)})
    if not files:
        raise ValueError("dataset file manifest cannot be empty")
    return {
        "schema_version": FILE_MANIFEST_SCHEMA_VERSION,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def write_file_manifest(root: str | Path, filename: str = DEFAULT_MANIFEST_NAME) -> Path:
    dataset_root = Path(root).expanduser().resolve()
    output = dataset_root / filename
    payload = build_file_manifest(dataset_root, excluded_relative_paths=(filename,))
    atomic_write_json(output, payload)
    return output


def verify_file_manifest(root: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else dataset_root / DEFAULT_MANIFEST_NAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != FILE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported dataset file manifest")
    expected = payload.get("files")
    if not isinstance(expected, list):
        raise ValueError("dataset file manifest files must be a list")
    actual = build_file_manifest(
        dataset_root,
        excluded_relative_paths=(manifest.relative_to(dataset_root).as_posix(),),
    )
    if actual != payload:
        raise ValueError("dataset files differ from the content-addressed manifest")
    return payload
