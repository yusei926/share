from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "build_lossless_intra_video_cache.py"
    spec = importlib.util.spec_from_file_location("build_lossless_intra_video_cache", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_lossless_intra_cache_is_pixel_exact_and_reusable(tmp_path: Path) -> None:
    module = load_module()
    source_root = tmp_path / "source"
    training_view = tmp_path / "training_view"
    cache_root = tmp_path / "cache"
    camera_map = {
        "observation.images.head_left": "observation.images.cam_0",
        "observation.images.left_wrist": "observation.images.cam_2",
        "observation.images.right_wrist": "observation.images.cam_3",
    }

    for source_camera in camera_map.values():
        video = source_root / "videos" / source_camera / "chunk-000" / "file-000.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x48:rate=30",
                "-frames:v",
                "12",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-g",
                "6",
                str(video),
            ],
            check=True,
        )

    (training_view / "meta").mkdir(parents=True)
    (training_view / "videos").mkdir()
    marker = {
        "source_root": str(source_root),
        "source_repo_id": "example/private_dataset",
        "source_revision": "0123456789abcdef",
        "source_fingerprint_sha256": "a" * 64,
        "camera_map": camera_map,
    }
    (training_view / module.TRAINING_VIEW_MARKER).write_text(
        json.dumps(marker), encoding="utf-8"
    )
    for policy_camera, source_camera in camera_map.items():
        (training_view / "videos" / policy_camera).symlink_to(
            source_root / "videos" / source_camera,
            target_is_directory=True,
        )

    first = module.build_cache(
        training_view,
        cache_root,
        workers=2,
        space_multiplier=1.0,
    )
    assert first["summary"]["file_count"] == 3
    assert first["summary"]["encoded_count"] == 3
    assert first["pixel_exact"] is True
    for report in first["files"]:
        assert report["pixel_exact"] is True
        assert report["all_frames_keyframes"] is True
        assert report["cache_video"]["frame_count"] == 12
        assert report["decoded_frame_md5"] == module.decoded_md5(
            Path(report["source"]["source_path"])
        )
        assert Path(report["cache_path"]).is_file()

    cached_video = (
        cache_root
        / "videos"
        / "observation.images.head_left"
        / "chunk-000"
        / "file-000.mp4"
    )
    modification_time = cached_video.stat().st_mtime_ns
    second = module.build_cache(
        training_view,
        cache_root,
        workers=2,
        space_multiplier=1.0,
    )
    assert second["summary"]["encoded_count"] == 0
    assert second["summary"]["reused_count"] == 3
    assert cached_video.stat().st_mtime_ns == modification_time
    for policy_camera in camera_map:
        link = training_view / "videos" / policy_camera
        assert link.is_symlink()
        assert link.resolve() == (cache_root / "videos" / policy_camera).resolve()
