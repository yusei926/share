#!/usr/bin/env python3
"""Convert an offline Isaac replay's rendered frames into one raw episode.

This is intentionally a host-side post-processing step.  AVP operation saves
only the 30 Hz real-compatible command trajectory, then Isaac replays that
trajectory without an operator waiting for all four RTX cameras.  This tool
refuses missing frames rather than padding or duplicating images.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .contracts import ArmHandTarget, ControlEvent, ControlMode, TeleopObservation
from .raw_episode import EpisodeIdentity, RawEpisodeWriter
from .shared.policy_contract import ACTION_DIM


REQUIRED_ROLES = ("head_left", "head_right", "left_wrist", "right_wrist")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--camera-frames", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--success", action="store_true")
    return parser.parse_args()


def _jpeg_444(path: Path) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (640, 480):
            raise ValueError(f"{path} must be 640x480, got {image.size}")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95, subsampling=0)
        return output.getvalue()


def _frame_directories(root: Path) -> list[Path]:
    result = sorted(
        (path for path in root.glob("frame_*") if path.is_dir()),
        key=lambda path: int(path.name.removeprefix("frame_")),
    )
    if not result:
        raise ValueError(f"no frame_XXXX directories under {root}")
    return result


def _frame_payload(frame_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    payload: dict[str, bytes] = {}
    for role in REQUIRED_ROLES:
        image_path = frame_dir / f"{role}.png"
        if not image_path.is_file():
            raise ValueError(f"missing {role} render: {image_path}")
        payload[role] = _jpeg_444(image_path)
    diagnostic: dict[str, bytes] = {}
    global_path = frame_dir / "global.png"
    if global_path.is_file():
        diagnostic["global"] = _jpeg_444(global_path)
    return payload, diagnostic


def _finite_vector(value: object, length: int, label: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite [{length}]")
    return tuple(float(item) for item in array)


def main() -> int:
    args = _arguments()
    trajectory = json.loads(args.trajectory.read_text(encoding="utf-8"))
    if trajectory.get("schema_version") != "team_ramen_flip_table_offline_replay_trajectory/v2":
        raise ValueError("unsupported replay trajectory schema")
    samples = trajectory.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("replay trajectory must contain at least two samples")
    frame_dirs = _frame_directories(args.camera_frames)
    if len(frame_dirs) != len(samples):
        raise ValueError(
            "offline replay frame count must exactly equal the 30 Hz command "
            f"count: frames={len(frame_dirs)}, samples={len(samples)}"
        )
    identity = EpisodeIdentity(
        backend="sim",
        dr_profile=str(trajectory["dr_profile"]),
        seed=int(trajectory["seed"]),
        config_sha256=str(trajectory["config_sha256"]),
        runtime_digest=str(trajectory["runtime_digest"]),
    )
    # A synthetic monotonic timeline is correct here: all images were rendered
    # from the same replay timeline at exactly n / 30 s.  It is not a claim
    # about the original interactive display transport clock.
    start_ns = 1_000_000_000
    dt_ns = round(1_000_000_000 / 30)
    with RawEpisodeWriter(args.output_root, identity) as writer:
        for index, (sample, frame_dir) in enumerate(zip(samples, frame_dirs, strict=True)):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {index} is not an object")
            camera_jpeg, diagnostic_jpeg = _frame_payload(frame_dir)
            timestamp = start_ns + index * dt_ns
            body = _finite_vector(sample["body_joint_position_rad_29d"], 29, "body state")
            velocity = _finite_vector(sample["body_joint_velocity_rad_s_29d"], 29, "body velocity")
            action = _finite_vector(sample["actions_16d"], ACTION_DIM, "action")
            root_pose = _finite_vector(sample["root_pose_xyzw"], 7, "root pose")
            hand = tuple(float(value / 4.5) for value in action[14:16])
            observation = TeleopObservation(
                sequence=index,
                capture_monotonic_ns=timestamp,
                backend="sim",
                body_joint_position_rad=body,
                body_joint_velocity_rad_s=velocity,
                dex1_opening_fraction=hand,
                applied_arm_target_rad=action[:14],
                applied_dex1_opening_target=hand,
                root_pose_xyzw=root_pose,
                camera_capture_monotonic_ns={role: timestamp for role in REQUIRED_ROLES},
                camera_jpeg=camera_jpeg,
                diagnostic_camera_capture_monotonic_ns={
                    role: timestamp for role in diagnostic_jpeg
                },
                diagnostic_camera_jpeg=diagnostic_jpeg,
                success=bool(args.success),
                diagnostics={
                    "offline_replay": True,
                    "source_trajectory": str(args.trajectory.resolve()),
                    "source_frame_directory": str(frame_dir.resolve()),
                    "privileged_policy_features": [],
                },
            )
            target = ArmHandTarget(
                sequence=index,
                monotonic_ns=timestamp,
                mode=ControlMode.TRACK,
                event=ControlEvent.NONE,
                arm_position_rad=action[:14],
                dex1_opening_fraction=hand,
            )
            writer.append(observation, target)
        result = writer.save(
            diagnostics={
                "offline_replay": True,
                "trajectory": str(args.trajectory.resolve()),
                "camera_frames": str(args.camera_frames.resolve()),
                "rendered_camera_hz": 30,
                "simulator_success": bool(args.success),
            },
            success=bool(args.success),
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
