"""Build a pixel-exact, seek-efficient video cache for a LeRobot training view."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CACHE_SCHEMA = "flip_table_lossless_intra_video_cache_v1"
TRAINING_VIEW_MARKER = Path("meta/team_ramen_training_view.json")
ENCODE_ARGS = (
    "-map",
    "0:v:0",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-qp",
    "0",
    "-g",
    "1",
    "-keyint_min",
    "1",
    "-sc_threshold",
    "0",
    "-pix_fmt",
    "yuv420p",
    "-threads",
    "8",
    "-movflags",
    "+faststart",
)


@dataclass(frozen=True)
class VideoSource:
    policy_camera: str
    source_camera: str
    source_path: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--space-multiplier",
        type=float,
        default=32.0,
        help="Required free bytes per uncached source byte before encoding.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary_path = Path(stream.name)
    temporary_path.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def probe_video(path: Path) -> dict[str, Any]:
    value = json.loads(
        command_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                (
                    "stream=codec_name,pix_fmt,width,height,r_frame_rate,"
                    "avg_frame_rate,duration,nb_frames,nb_read_packets"
                ),
                "-of",
                "json",
                str(path),
            ]
        )
    )
    streams = value.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected exactly one video stream: {path}")
    stream = streams[0]
    frame_count_raw = stream.get("nb_frames") or stream.get("nb_read_packets")
    if frame_count_raw in {None, "N/A"}:
        raise ValueError(f"video frame count is unavailable: {path}")
    return {
        "codec_name": str(stream["codec_name"]),
        "pix_fmt": str(stream["pix_fmt"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "r_frame_rate": str(stream["r_frame_rate"]),
        "avg_frame_rate": str(stream["avg_frame_rate"]),
        "duration": float(stream["duration"]),
        "frame_count": int(frame_count_raw),
    }


def decoded_md5(path: Path) -> str:
    output = command_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "md5",
            "-",
        ]
    )
    prefix = "MD5="
    if not output.startswith(prefix) or len(output) != len(prefix) + 32:
        raise ValueError(f"unexpected ffmpeg MD5 output for {path}: {output!r}")
    return output[len(prefix) :]


def keyframe_count(path: Path) -> int:
    output = command_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=flags",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return sum("K" in line for line in output.splitlines())


def load_sources(training_view: Path) -> tuple[dict[str, Any], list[VideoSource]]:
    marker = read_json(training_view / TRAINING_VIEW_MARKER)
    source_root = Path(marker["source_root"]).expanduser().resolve()
    camera_map = marker.get("camera_map")
    if not isinstance(camera_map, dict) or not camera_map:
        raise ValueError("training-view marker has no camera_map")

    sources: list[VideoSource] = []
    for policy_camera, source_camera_value in sorted(camera_map.items()):
        source_camera = str(source_camera_value)
        source_dir = source_root / "videos" / source_camera
        if not source_dir.is_dir():
            raise FileNotFoundError(f"source camera directory is missing: {source_dir}")
        camera_files = sorted(path for path in source_dir.rglob("*.mp4") if path.is_file())
        if not camera_files:
            raise FileNotFoundError(f"source camera has no MP4 files: {source_dir}")
        sources.extend(
            VideoSource(
                policy_camera=str(policy_camera),
                source_camera=source_camera,
                source_path=path.resolve(),
                relative_path=path.relative_to(source_dir),
            )
            for path in camera_files
        )
    return marker, sources


def file_contract(source: VideoSource) -> dict[str, Any]:
    return {
        "policy_camera": source.policy_camera,
        "source_camera": source.source_camera,
        "relative_path": source.relative_path.as_posix(),
        "source_path": str(source.source_path),
        "source_bytes": source.source_path.stat().st_size,
        "source_sha256": sha256(source.source_path),
        "source_video": probe_video(source.source_path),
        "encoder_args": list(ENCODE_ARGS),
    }


def cached_report_if_valid(
    output_path: Path,
    sidecar_path: Path,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    if not output_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        report = read_json(sidecar_path)
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if report.get("source") != contract:
        return None
    if report.get("cache_bytes") != output_path.stat().st_size:
        return None
    if report.get("cache_sha256") != sha256(output_path):
        return None
    if not report.get("pixel_exact") or not report.get("all_frames_keyframes"):
        return None
    return report


def encode_one(
    source: VideoSource,
    cache_root: Path,
) -> tuple[dict[str, Any], bool]:
    contract = file_contract(source)
    output_path = cache_root / "videos" / source.policy_camera / source.relative_path
    sidecar_path = output_path.with_suffix(f"{output_path.suffix}.json")
    cached = cached_report_if_valid(output_path, sidecar_path, contract)
    if cached is not None:
        return cached, True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}-{os.getpid() ^ id(source)}.mp4"
    )
    temporary_path.unlink(missing_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-i",
                str(source.source_path),
                *ENCODE_ARGS,
                str(temporary_path),
            ],
            check=True,
        )
        source_md5 = decoded_md5(source.source_path)
        cache_md5 = decoded_md5(temporary_path)
        if source_md5 != cache_md5:
            raise ValueError(
                f"decoded pixels changed for {source.policy_camera}/{source.relative_path}"
            )
        source_video = contract["source_video"]
        cache_video = probe_video(temporary_path)
        matching_fields = (
            "pix_fmt",
            "width",
            "height",
            "r_frame_rate",
            "avg_frame_rate",
            "frame_count",
        )
        mismatches = {
            name: (source_video[name], cache_video[name])
            for name in matching_fields
            if source_video[name] != cache_video[name]
        }
        if abs(source_video["duration"] - cache_video["duration"]) > 1e-6:
            mismatches["duration"] = (source_video["duration"], cache_video["duration"])
        if mismatches:
            raise ValueError(
                f"video contract changed for {source.policy_camera}/{source.relative_path}: "
                f"{mismatches}"
            )
        cache_keyframes = keyframe_count(temporary_path)
        if cache_keyframes != cache_video["frame_count"]:
            raise ValueError(
                f"cache is not all-I: keyframes={cache_keyframes}, "
                f"frames={cache_video['frame_count']}"
            )
        temporary_path.replace(output_path)
        report = {
            "source": contract,
            "cache_path": str(output_path),
            "cache_bytes": output_path.stat().st_size,
            "cache_sha256": sha256(output_path),
            "cache_video": cache_video,
            "decoded_frame_md5": source_md5,
            "pixel_exact": True,
            "all_frames_keyframes": True,
        }
        write_json_atomic(sidecar_path, report)
        return report, False
    finally:
        temporary_path.unlink(missing_ok=True)


def ensure_space(
    sources: list[VideoSource],
    cache_root: Path,
    *,
    multiplier: float,
    workers: int,
) -> None:
    if multiplier <= 0:
        raise ValueError("--space-multiplier must be positive")
    cache_root.mkdir(parents=True, exist_ok=True)
    concurrent_encodes = 1 if (cache_root / "manifest.json").is_file() else workers
    largest_sources = sorted(
        (source.source_path.stat().st_size for source in sources),
        reverse=True,
    )[:concurrent_encodes]
    required = int(sum(largest_sources) * multiplier)
    free = shutil.disk_usage(cache_root).free
    if free < required:
        raise OSError(
            f"insufficient free space for lossless cache: required={required}, free={free}"
        )


def relink_training_view(
    training_view: Path,
    cache_root: Path,
    policy_cameras: set[str],
) -> None:
    videos_root = training_view / "videos"
    for policy_camera in sorted(policy_cameras):
        link_path = videos_root / policy_camera
        target_path = (cache_root / "videos" / policy_camera).resolve()
        if not target_path.is_dir():
            raise FileNotFoundError(target_path)
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            raise ValueError(
                f"refusing to replace non-symlink training-view camera directory: {link_path}"
            )
        link_path.symlink_to(target_path, target_is_directory=True)


def build_cache(
    training_view: Path,
    cache_root: Path,
    *,
    workers: int,
    space_multiplier: float,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FileNotFoundError(f"{executable} is required")

    training_view = training_view.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    marker, sources = load_sources(training_view)
    ensure_space(
        sources,
        cache_root,
        multiplier=space_multiplier,
        workers=min(workers, len(sources)),
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(sources))
    ) as executor:
        results = list(executor.map(lambda source: encode_one(source, cache_root), sources))
    reports = [report for report, _reused in results]
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "pixel_exact": True,
        "source": {
            "repo_id": marker["source_repo_id"],
            "revision": marker["source_revision"],
            "fingerprint_sha256": marker["source_fingerprint_sha256"],
            "camera_map": marker["camera_map"],
        },
        "encoder_args": list(ENCODE_ARGS),
        "files": sorted(
            reports,
            key=lambda report: (
                report["source"]["policy_camera"],
                report["source"]["relative_path"],
            ),
        ),
        "summary": {
            "file_count": len(reports),
            "encoded_count": sum(not reused for _report, reused in results),
            "reused_count": sum(reused for _report, reused in results),
            "source_bytes": sum(report["source"]["source_bytes"] for report in reports),
            "cache_bytes": sum(report["cache_bytes"] for report in reports),
        },
    }
    write_json_atomic(cache_root / "manifest.json", manifest)
    relink_training_view(
        training_view,
        cache_root,
        {source.policy_camera for source in sources},
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_cache(
        args.training_view,
        args.cache_root,
        workers=args.workers,
        space_multiplier=args.space_multiplier,
    )
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
