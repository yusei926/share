"""Index immutable LeRobot v3 source episodes and their shared videos."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _indexed_path(
    root: Path,
    prefix: str,
    chunk_index: int,
    file_index: int,
    suffix: str,
) -> Path:
    path = root / prefix / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.{suffix}"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class VideoSlice:
    """The time interval occupied by one episode in a shared video shard."""

    feature: str
    path: Path
    from_timestamp: float
    to_timestamp: float

    def timestamp_for_frame(self, frame_index: int, fps: int, frame_count: int) -> float:
        if not 0 <= frame_index < frame_count:
            raise ValueError(f"frame {frame_index} is outside [0, {frame_count})")
        timestamp = self.from_timestamp + frame_index / fps
        if timestamp > self.to_timestamp + 1.5 / fps:
            raise ValueError(
                f"{self.feature} frame {frame_index} timestamp {timestamp:.9f} exceeds "
                f"episode end {self.to_timestamp:.9f}"
            )
        return timestamp


@dataclass(frozen=True)
class SourceEpisode:
    """One immutable source episode and its LeRobot v3 index row."""

    source_root: Path
    metadata: dict[str, Any]

    @property
    def episode_index(self) -> int:
        return int(self.metadata["episode_index"])

    @property
    def frame_count(self) -> int:
        return int(self.metadata["length"])

    @property
    def data_path(self) -> Path:
        return _indexed_path(
            self.source_root,
            "data",
            int(self.metadata["data/chunk_index"]),
            int(self.metadata["data/file_index"]),
            "parquet",
        )

    def video_relative_path(self, feature: str) -> Path:
        prefix = f"videos/{feature}"
        required = (
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        )
        missing = [key for key in required if key not in self.metadata]
        if missing:
            raise ValueError(f"episode {self.episode_index} lacks video metadata: {missing}")
        return (
            Path(prefix)
            / f"chunk-{int(self.metadata[required[0]]):03d}"
            / f"file-{int(self.metadata[required[1]]):03d}.mp4"
        )

    def video_slice(self, feature: str) -> VideoSlice:
        prefix = f"videos/{feature}"
        relative = self.video_relative_path(feature)
        path = self.source_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return VideoSlice(
            feature=feature,
            path=path,
            from_timestamp=float(self.metadata[f"{prefix}/from_timestamp"]),
            to_timestamp=float(self.metadata[f"{prefix}/to_timestamp"]),
        )


class SourceDatasetIndex:
    """Resolve contiguous LeRobot v3 episode rows and shared media shards."""

    def __init__(self, source_root: str | Path):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to index source episodes") from exc

        self.source_root = Path(source_root).expanduser().resolve()
        episode_files = sorted((self.source_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        if not episode_files:
            raise FileNotFoundError(self.source_root / "meta" / "episodes")
        rows = pq.read_table([str(path) for path in episode_files]).to_pylist()
        rows.sort(key=lambda row: int(row["episode_index"]))
        indices = [int(row["episode_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError("source episode metadata must be contiguous from zero")
        self._rows = tuple(rows)

    def episode(self, episode_index: int) -> SourceEpisode:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError("episode_index must be an integer")
        if not 0 <= episode_index < len(self._rows):
            raise IndexError(f"episode {episode_index} is outside [0, {len(self._rows)})")
        return SourceEpisode(self.source_root, self._rows[episode_index])

    def __len__(self) -> int:
        return len(self._rows)


def _ffmpeg_executable() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    executable = configured or shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is unavailable; set FFMPEG_BINARY or install imageio-ffmpeg"
        ) from exc
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def select_review_frames(frame_count: int, count: int = 9) -> tuple[int, ...]:
    """Choose deterministic endpoint-inclusive frames for camera audits."""

    if frame_count <= 0 or count <= 0:
        raise ValueError("frame_count and count must be positive")
    if count == 1:
        return (0,)
    if frame_count <= count:
        return tuple(range(frame_count))
    return tuple(round(index * (frame_count - 1) / (count - 1)) for index in range(count))


def extract_video_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    """Extract one frame atomically from a shared LeRobot video shard."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}")
    temporary.unlink(missing_ok=True)
    result = subprocess.run(
        (
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.9f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vsync",
            "0",
            "-y",
            str(temporary),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg could not extract {video_path} at {timestamp:.9f}s: {result.stderr.strip()}"
        )
    os.replace(temporary, output_path)
