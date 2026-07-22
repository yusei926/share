#!/usr/bin/env python3
"""Extract one digest-pinned Skopeo directory image with OCI whiteouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.flip_table_data_augmentation.io_utils import sha256_file


MARKER_NAME = ".flip-table-augmentation-rootfs.json"
MARKER_SCHEMA_VERSION = 1


def _safe_member_name(name: str) -> PurePosixPath:
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"unsafe OCI layer member: {name!r}")
    return normalized


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _apply_whiteouts(root: Path, members: list[tarfile.TarInfo]) -> set[str]:
    skipped: set[str] = set()
    for member in members:
        relative = _safe_member_name(member.name)
        basename = relative.name
        if not basename.startswith(".wh."):
            continue
        skipped.add(member.name)
        parent = root.joinpath(*relative.parent.parts)
        if basename == ".wh..wh..opq":
            if parent.is_dir():
                for child in parent.iterdir():
                    _remove_path(child)
            elif parent.exists() or parent.is_symlink():
                raise RuntimeError(f"opaque whiteout parent is not a directory: {parent}")
            else:
                parent.mkdir(parents=True)
        else:
            _remove_path(parent / basename.removeprefix(".wh."))
    return skipped


def _layer_path(layout: Path, digest: str) -> Path:
    algorithm, separator, value = digest.partition(":")
    if algorithm != "sha256" or not separator or len(value) != 64:
        raise ValueError(f"unsupported OCI layer digest: {digest!r}")
    path = layout / value
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != value:
        raise ValueError(
            f"OCI layer digest mismatch for {path.name}: expected {value}, observed {observed}"
        )
    return path


def _extract_layer(layer: Path, root: Path) -> None:
    with tarfile.open(layer, mode="r:*") as archive:
        members = archive.getmembers()
        skipped = _apply_whiteouts(root, members)
        selected = []
        for member in members:
            _safe_member_name(member.name)
            if member.name not in skipped:
                selected.append(member)
        # The source is an immutable organizer image whose manifest and every
        # layer are verified above. fully_trusted preserves its Linux rootfs
        # links and modes; path traversal names are rejected before extraction.
        archive.extractall(root, members=selected, filter="fully_trusted")


def _read_manifest(layout: Path, expected_digest: str) -> tuple[dict[str, Any], str]:
    manifest_path = layout / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    observed = f"sha256:{sha256_file(manifest_path)}"
    if observed != expected_digest:
        raise ValueError(
            "OCI manifest digest mismatch: "
            f"expected {expected_digest}, observed {observed}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schemaVersion") != 2:
        raise ValueError("only OCI/Docker schemaVersion 2 manifests are supported")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("OCI manifest has no layers")
    return manifest, observed


def _expected_marker(manifest: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "manifest_digest": digest,
        "config_digest": manifest["config"]["digest"],
        "layer_digests": [entry["digest"] for entry in manifest["layers"]],
    }


def extract(*, layout: Path, output: Path, expected_digest: str) -> dict[str, Any]:
    manifest, observed_digest = _read_manifest(layout, expected_digest)
    marker = _expected_marker(manifest, observed_digest)
    marker_path = output / MARKER_NAME
    if marker_path.is_file():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != marker:
            raise ValueError(f"existing rootfs marker does not match {expected_digest}")
        required = output / "workspace" / "robofinals"
        if not required.is_dir():
            raise FileNotFoundError(required)
        return marker
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite unverified rootfs directory without {MARKER_NAME}: {output}"
        )

    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for index, entry in enumerate(manifest["layers"], start=1):
            layer = _layer_path(layout, entry["digest"])
            print(f"extracting layer {index}/{len(manifest['layers'])}: {entry['digest']}", flush=True)
            _extract_layer(layer, temporary)
        marker_text = json.dumps(marker, indent=2, sort_keys=True) + "\n"
        (temporary / MARKER_NAME).write_text(marker_text, encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return marker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-digest", required=True)
    args = parser.parse_args()
    marker = extract(
        layout=args.layout.resolve(),
        output=args.output.resolve(),
        expected_digest=args.expected_manifest_digest,
    )
    print(json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
