"""Build a bounded, seek-friendly RGB cache for a local LeRobot training view.

The immutable Hugging Face source remains untouched.  This creates only a
320x240, short-GOP derivative under an experiment output directory and relinks
the local training view to it.  A short GOP prevents random chunk sampling from
repeatedly decoding hundreds of source frames.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
ENCODE_ARGS = (
    "-vf",
    "scale=320:240:flags=lanczos",
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "20",
    "-g",
    "8",
    "-keyint_min",
    "8",
    "-sc_threshold",
    "0",
    "-an",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_videos(training_view: Path) -> list[tuple[str, Path, Path]]:
    videos_root = training_view / "videos"
    result: list[tuple[str, Path, Path]] = []
    for camera_key in CAMERA_KEYS:
        source_dir = videos_root / camera_key
        if not source_dir.is_dir():
            raise FileNotFoundError(f"missing source video directory: {source_dir}")
        for source_path in sorted(source_dir.rglob("*.mp4")):
            result.append((camera_key, source_path, source_path.relative_to(source_dir)))
    if not result:
        raise FileNotFoundError(f"no RGB MP4 files under {videos_root}")
    return result


def expected_manifest(training_view: Path, files: list[tuple[str, Path, Path]]) -> dict:
    return {
        "format": "flip_table_rgb_cache_v1",
        "image_size": [320, 240],
        "gop": 8,
        "encoder": "libx264 veryfast crf20",
        "sources": [
            {
                "camera_key": camera_key,
                "relative_path": str(relative_path),
                "sha256": sha256(source_path.resolve()),
            }
            for camera_key, source_path, relative_path in files
        ],
    }


def cache_is_valid(cache_root: Path, manifest: dict) -> bool:
    manifest_path = cache_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if recorded != manifest:
        return False
    return all(
        (cache_root / source["camera_key"] / source["relative_path"]).is_file()
        for source in manifest["sources"]
    )


def encode_one(source_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".partial.mp4")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path), *ENCODE_ARGS, str(temporary_path)]
    subprocess.run(command, check=True)
    temporary_path.replace(output_path)


def relink_training_view(training_view: Path, cache_root: Path) -> None:
    for camera_key in CAMERA_KEYS:
        link_path = training_view / "videos" / camera_key
        target_path = cache_root / camera_key
        if not target_path.is_dir():
            raise FileNotFoundError(f"missing cached camera directory: {target_path}")
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.exists():
            shutil.rmtree(link_path)
        link_path.symlink_to(target_path, target_is_directory=True)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    training_view = args.training_view.resolve()
    cache_root = args.cache_root.resolve()
    files = source_videos(training_view)
    manifest = expected_manifest(training_view, files)
    if args.force and cache_root.exists():
        shutil.rmtree(cache_root)
    if not cache_is_valid(cache_root, manifest):
        cache_root.mkdir(parents=True, exist_ok=True)
        tasks = [
            (source_path.resolve(), cache_root / camera_key / relative_path)
            for camera_key, source_path, relative_path in files
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            futures = [pool.submit(encode_one, source, target) for source, target in tasks]
            for future in futures:
                future.result()
        (cache_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    relink_training_view(training_view, cache_root)
    print(json.dumps({"cache_root": str(cache_root), "files": len(files), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
