from __future__ import annotations

import sys
import types

from data.flip_table_data_augmentation import source_dataset


def test_ffmpeg_executable_prefers_explicit_environment(monkeypatch) -> None:
    monkeypatch.setenv("FFMPEG_BINARY", "/opt/tools/ffmpeg")
    monkeypatch.setattr(source_dataset.shutil, "which", lambda _: None)

    assert source_dataset._ffmpeg_executable() == "/opt/tools/ffmpeg"


def test_ffmpeg_executable_uses_imageio_fallback(monkeypatch) -> None:
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(source_dataset.shutil, "which", lambda _: None)
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "/cached/ffmpeg"),
    )

    assert source_dataset._ffmpeg_executable() == "/cached/ffmpeg"
