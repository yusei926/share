from __future__ import annotations

from typing import Any

from .config import CurationConfig
from .labels import write_selection
from .source import VIDEO_KEYS, download_source
from .util import atomic_write_json


def audit_labels(config: CurationConfig) -> dict[str, Any]:
    """Validate pinned source metadata and manual labels before any bulk download."""
    snapshot = download_source(config, include_data=False, include_videos=False)
    source = config.section("source")
    errors: list[str] = []
    if int(snapshot.info.get("fps", -1)) != int(source["fps"]):
        errors.append("source fps mismatch")
    if len(snapshot.episodes) != int(source["expected_episodes"]):
        errors.append("source episode count mismatch")
    if int(snapshot.info.get("total_frames", -1)) != int(source["expected_frames"]):
        errors.append("source frame count mismatch")
    features = snapshot.info.get("features", {})
    for key in VIDEO_KEYS:
        feature = features.get(key, {})
        if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
            errors.append(f"invalid source video feature: {key}")
    if errors:
        raise RuntimeError("; ".join(errors))
    source_lengths = {int(row["episode_index"]): int(row["length"]) for row in snapshot.episodes}
    try:
        selection = write_selection(config, source_lengths=source_lengths)
        expected_selected = int(config.section("target")["expected_episodes"])
        if int(selection["selected_segments"]) != expected_selected:
            raise RuntimeError(
                f"selected segment count {selection['selected_segments']} != expected {expected_selected}"
            )
        expected_frames = config.section("target").get("expected_frames")
        if expected_frames is not None and int(selection["selected_frames"]) != int(expected_frames):
            raise RuntimeError(
                f"selected frame count {selection['selected_frames']} != expected {expected_frames}"
            )
    except ValueError as error:
        report = {
            "schema_version": "team_ramen_manual_flip_table_audit/v1",
            "config_sha256": config.digest,
            "source_repo_id": config.source_repo_id,
            "source_revision": config.source_revision,
            "source_episodes": len(snapshot.episodes),
            "source_frames": int(snapshot.info["total_frames"]),
            "passed": False,
            "error": str(error),
        }
        atomic_write_json(config.workspace / "audit" / "label_audit.json", report)
        raise
    report = {
        "schema_version": "team_ramen_manual_flip_table_audit/v1",
        "config_sha256": config.digest,
        "source_repo_id": config.source_repo_id,
        "source_revision": config.source_revision,
        "source_episodes": len(snapshot.episodes),
        "source_frames": int(snapshot.info["total_frames"]),
        "selection": {
            "segments": selection["selected_segments"],
            "frames": selection["selected_frames"],
            "labels_sha256": selection["labels_sha256"],
        },
        "passed": True,
    }
    atomic_write_json(config.workspace / "audit" / "label_audit.json", report)
    return report
