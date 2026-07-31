from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import CurationConfig
from .source import (
    INDEX_KEYS,
    NUMERIC_WIDTHS,
    VIDEO_KEYS,
    download_source,
    fixed_list_numpy,
    read_numeric_table,
    source_inventory,
)
from .util import atomic_write_json, ensure_hash
from .walking import FootKinematics


def _video_decode_audit(config: CurationConfig) -> dict:
    snapshot = download_source(config, include_videos=True, rgb_only=False)
    files: dict[str, dict[str, int | float]] = {}
    errors: list[str] = []
    paths = [
        path
        for key in VIDEO_KEYS
        for path in sorted((snapshot.root / "videos" / key).glob("chunk-*/*.mp4"))
    ]

    def probe(path):
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
        try:
            stream = json.loads(subprocess.check_output(command))["streams"][0]
            numerator, denominator = (
                int(value) for value in stream["r_frame_rate"].split("/")
            )
            fps = numerator / denominator
            relative = path.relative_to(snapshot.root).as_posix()
            metadata = {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": fps,
                "decoded_frames": int(stream["nb_read_frames"]),
            }
            error = None
            if (
                metadata["width"] != 640
                or metadata["height"] != 480
                or abs(fps - snapshot.fps) > 1e-6
                or metadata["decoded_frames"] <= 0
            ):
                error = f"invalid decoded video stream: {relative}"
            return relative, metadata, error
        except Exception as error:
            return None, None, f"video decode failed: {path}: {error}"

    # ffprobe frame counting is CPU-bound but independent per packed file.
    # Eight workers keep the full 32-file audit practical on the workstation.
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
        for relative, metadata, error in executor.map(probe, paths):
            if relative is not None:
                files[relative] = metadata
            if error is not None:
                errors.append(error)
    return {"files": files, "errors": errors}


def audit_source(config: CurationConfig, *, include_video_decode: bool = False) -> dict:
    ensure_hash(
        config.asset_path("urdf_path"),
        str(config.section("assets")["urdf_sha256"]),
        "G1 URDF",
    )
    ensure_hash(
        config.asset_path("yolo_weight_path"),
        str(config.section("assets")["yolo_weight_sha256"]),
        "YOLO OBB weight",
    )
    snapshot = download_source(config, include_videos=False)
    inventory = source_inventory(snapshot)
    source_config = config.section("source")
    errors: list[str] = []
    warnings: list[str] = []
    if inventory["fps"] != int(source_config["fps"]):
        errors.append(f"fps mismatch: {inventory['fps']} != {source_config['fps']}")
    if inventory["episodes"] != int(source_config["expected_episodes"]):
        errors.append("episode count mismatch")
    if inventory["frames"] != int(source_config["expected_frames"]):
        errors.append("frame count mismatch")
    if inventory["missing_video_features"] or inventory["missing_numeric_features"]:
        errors.append("required source features are missing")
    if inventory["unique_source_episode_names"] != inventory["episodes"]:
        errors.append("source episode names are not unique")
    table = read_numeric_table(snapshot)
    if len(table) != inventory["frames"]:
        errors.append("numeric row count does not match metadata")
    nonfinite: dict[str, int] = {}
    for key, width in NUMERIC_WIDTHS.items():
        values = fixed_list_numpy(table[key], width)
        count = int(np.size(values) - np.sum(np.isfinite(values)))
        nonfinite[key] = count
        if count:
            errors.append(f"{key} contains {count} non-finite values")
    kinematics = FootKinematics(config.asset_path("urdf_path"))
    lower = kinematics.model.lowerPositionLimit[kinematics.joint_q]
    upper = kinematics.model.upperPositionLimit[kinematics.joint_q]
    joint_limit_audit = {}
    for key in ("observation.state.robot_q_current", "action.robot_q_desired"):
        joints = fixed_list_numpy(table[key], 36)[:, 7:]
        violation = np.maximum(
            np.maximum(lower[None, :] - joints, 0.0),
            np.maximum(joints - upper[None, :], 0.0),
        )
        count = int(np.count_nonzero(violation > 1e-4))
        joint_limit_audit[key] = {
            "violation_count": count,
            "maximum_violation_rad": float(np.max(violation)),
        }
        if count and key == "observation.state.robot_q_current":
            errors.append(f"{key} contains {count} URDF joint-limit violations")
        elif count:
            # The legacy v1 full-body WBC trace contains lower-body desired
            # values outside the kinematic URDF range. Those dimensions are
            # not part of the canonical 16D upper-body learning view, and this
            # curation is forbidden from modifying them. Preserve and disclose
            # them, while separately requiring all 14 arm targets to be valid.
            warnings.append(
                f"{key} contains {count} lower-body/full-trace joint-limit violations"
            )
    desired_joints = fixed_list_numpy(
        table["action.robot_q_desired"], 36
    )[:, 7:]
    arm_violation = np.maximum(
        np.maximum(lower[None, 15:] - desired_joints[:, 15:], 0.0),
        np.maximum(desired_joints[:, 15:] - upper[None, 15:], 0.0),
    )
    arm_violation_count = int(np.count_nonzero(arm_violation > 1e-4))
    joint_limit_audit["policy_arm_action_14d"] = {
        "violation_count": arm_violation_count,
        "maximum_violation_rad": float(np.max(arm_violation)),
    }
    if arm_violation_count:
        errors.append(
            "action.robot_q_desired arm14 contains "
            f"{arm_violation_count} URDF joint-limit violations"
        )
    indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(table), dtype=np.int64)):
        errors.append("global index is not contiguous")
    features = snapshot.info["features"]
    for key in VIDEO_KEYS:
        metadata = features[key]
        if metadata.get("dtype") != "video" or metadata.get("shape") != [480, 640, 3]:
            errors.append(f"invalid video metadata for {key}")
        if float(metadata.get("info", {}).get("video.fps", 0)) != snapshot.fps:
            errors.append(f"video fps mismatch for {key}")
    report_path = config.workspace / "audit" / "source_audit.json"
    cached_video_decode = None
    if not include_video_decode and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            previous.get("config_sha256") == config.digest
            and previous.get("source_revision") == config.source_revision
            and previous.get("video_decode", {}).get("files")
        ):
            cached_video_decode = previous["video_decode"]
    video_decode = (
        _video_decode_audit(config)
        if include_video_decode
        else (cached_video_decode or {"files": {}, "errors": []})
    )
    errors.extend(video_decode["errors"])
    report = {
        "schema_version": "team_ramen_flip_table_source_audit/v1",
        "config_sha256": config.digest,
        "code_sha256": config.code_digest,
        "source_repo_id": config.source_repo_id,
        "source_revision": config.source_revision,
        "inventory": inventory,
        "nonfinite": nonfinite,
        "joint_limits": joint_limit_audit,
        "video_decode": video_decode,
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    path = report_path
    atomic_write_json(path, report)
    if errors:
        raise RuntimeError(f"source audit failed; see {path}")
    return report
