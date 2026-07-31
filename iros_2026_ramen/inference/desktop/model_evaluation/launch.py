"""Launch a sealed registered model through its trusted local family plugin.

The command has no arbitrary runner-argument passthrough. Checkpoint, worker,
revision, task, interface, and camera server therefore cannot be replaced after
artifact validation. A small allowlist of bounded motion limits is exposed so
every policy family uses the same operator-facing command.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

from data.flip_table_data_augmentation.teleop.config import load_teleop_config

from .artifacts import load_prepared_spec, validate_prepared_artifacts
from .cli import runner_argv
from .recording import CAPTURE_ENV
from .wandb_export import (
    DEFAULT_MINIMUM_SECONDS,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    evaluate_run,
    upload_run,
    write_summary,
    write_upload_status,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="owner/repo or Hugging Face URL")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--device")
    parser.add_argument("--pre-motion-arm-velocity-rad-s", type=float)
    parser.add_argument("--pre-motion-arm-acceleration-rad-s2", type=float)
    parser.add_argument("--pre-motion-waypoint-tolerance-rad", type=float)
    parser.add_argument("--pre-motion-stage-timeout-s", type=float)
    parser.add_argument("--policy-arm-velocity-rad-s", type=float)
    parser.add_argument("--policy-arm-acceleration-rad-s2", type=float)
    parser.add_argument("--policy-hand-velocity-fraction-s", type=float)
    parser.add_argument("--policy-hand-acceleration-fraction-s2", type=float)
    parser.add_argument("--actuate", action="store_true")
    parser.add_argument(
        "--wandb-min-seconds",
        type=float,
        default=float(
            os.environ.get("IROS_WANDB_MIN_POLICY_SECONDS", DEFAULT_MINIMUM_SECONDS)
        ),
        help="Upload only normally completed policy runs at least this long.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY", DEFAULT_WANDB_ENTITY),
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT),
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args(argv)


def resolve_xr_runtime() -> tuple[Path, Path]:
    revision = load_teleop_config().runtime.xr_revision
    xr_root = Path.home() / ".cache/iros_2026_ramen" / f"xr_teleoperate-{revision}"
    candidates = (
        os.environ.get("XR_TELEOP_CONDA"),
        str(Path.home() / "miniforge3/condabin/conda"),
        str(Path.home() / "miniconda3/condabin/conda"),
    )
    conda = next(
        (Path(value) for value in candidates if value and Path(value).is_file()),
        None,
    )
    if conda is None:
        raise FileNotFoundError("xr_teleoperate conda executable is unavailable")
    env_name = os.environ.get("XR_TELEOP_ENV")
    if not env_name:
        envs = subprocess.run(
            [str(conda), "env", "list"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        env_name = "tv" if any(
            line.split() and line.split()[0] == "tv" for line in envs.splitlines()
        ) else "xr-teleop"
    python = Path(
        subprocess.run(
            [
                str(conda),
                "run",
                "-n",
                env_name,
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not python.is_file() or not (xr_root / ".git").is_dir():
        raise FileNotFoundError("pinned xr_teleoperate runtime is incomplete")
    actual = subprocess.run(
        ["git", "-C", str(xr_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual != revision:
        raise RuntimeError(
            f"xr_teleoperate revision mismatch: expected={revision}, actual={actual}"
        )
    return python, xr_root


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.wandb_min_seconds) or args.wandb_min_seconds <= 0:
        raise ValueError("--wandb-min-seconds must be finite and positive")
    if not args.wandb_entity.strip() or not args.wandb_project.strip():
        raise ValueError("W&B entity/project must be non-empty")
    spec = load_prepared_spec(
        args.local_dir,
        reference=args.model,
        revision=args.revision,
    )
    report = validate_prepared_artifacts(args.local_dir, spec)
    python, xr_root = resolve_xr_runtime()
    try:
        interface = os.environ["G1_DDS_INTERFACE"]
        image_server_ip = os.environ["G1_IMAGE_SERVER_IP"]
    except KeyError as exc:
        raise RuntimeError(
            "set G1_DDS_INTERFACE and G1_IMAGE_SERVER_IP before live evaluation"
        ) from exc
    started = datetime.now(ZoneInfo("Asia/Tokyo"))
    run_id = f"{started:%Y%m%d_%H%M%S}_{_slug(spec.model_id)}"
    run_dir = REPO_ROOT / "outputs/real_policy_evaluation/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    event_log = run_dir / "events.jsonl"
    argv = runner_argv(
        spec,
        args.local_dir,
        actuate=args.actuate,
        interface=interface,
        image_server_ip=image_server_ip,
        max_seconds=args.max_seconds,
        device=args.device,
        safety_limits={
            name: value
            for name, value in (
                ("pre_motion_arm_velocity_rad_s", args.pre_motion_arm_velocity_rad_s),
                ("pre_motion_arm_acceleration_rad_s2", args.pre_motion_arm_acceleration_rad_s2),
                ("pre_motion_waypoint_tolerance_rad", args.pre_motion_waypoint_tolerance_rad),
                ("pre_motion_stage_timeout_s", args.pre_motion_stage_timeout_s),
                ("policy_arm_velocity_rad_s", args.policy_arm_velocity_rad_s),
                ("policy_arm_acceleration_rad_s2", args.policy_arm_acceleration_rad_s2),
                ("policy_hand_velocity_fraction_s", args.policy_hand_velocity_fraction_s),
                ("policy_hand_acceleration_fraction_s2", args.policy_hand_acceleration_fraction_s2),
            )
            if value is not None
        },
        log_path=event_log,
    )
    argv[0] = str(python)
    env = os.environ.copy()
    paths = (
        REPO_ROOT,
        xr_root,
        xr_root / "teleop/televuer/src",
        xr_root / "teleop/teleimager/src",
    )
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = ":".join(
        [*(str(path) for path in paths), *([old] if old else [])]
    )
    env[CAPTURE_ENV] = str(run_dir / "capture")
    metadata = {
        "run_id": run_id,
        "started_at_jst": started.isoformat(timespec="seconds"),
        "model_id": spec.model_id,
        "model_repo_id": spec.repo_id,
        "model_revision": spec.revision,
        "family": spec.family,
        "task": spec.task,
        "camera_roles": list(spec.camera_roles),
        "canonical_output": spec.canonical_output,
        "lower_body_owner": "unitree_regular_mode",
        "lower_body_command_dimensions": 0,
        "requested_max_seconds": args.max_seconds,
        "actuated": args.actuate,
        "source_commit": _source_commit(),
        "safety_limit_overrides": {
            name: value
            for name, value in (
                ("pre_motion_arm_velocity_rad_s", args.pre_motion_arm_velocity_rad_s),
                ("pre_motion_arm_acceleration_rad_s2", args.pre_motion_arm_acceleration_rad_s2),
                ("pre_motion_waypoint_tolerance_rad", args.pre_motion_waypoint_tolerance_rad),
                ("pre_motion_stage_timeout_s", args.pre_motion_stage_timeout_s),
                ("policy_arm_velocity_rad_s", args.policy_arm_velocity_rad_s),
                ("policy_arm_acceleration_rad_s2", args.policy_arm_acceleration_rad_s2),
                ("policy_hand_velocity_fraction_s", args.policy_hand_velocity_fraction_s),
                ("policy_hand_acceleration_fraction_s2", args.policy_hand_acceleration_fraction_s2),
            )
            if value is not None
        },
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[registry] repo={spec.repo_id} revision={spec.revision} "
        f"family={spec.family} sealed={report['tamper_check']} "
        f"actuate={args.actuate}",
        flush=True,
    )
    if not args.actuate:
        print("[registry] live read-only preflight; no --actuate passed", flush=True)
    print(f"[recording] local run directory: {run_dir}", flush=True)
    process = subprocess.Popen(argv, env=env)
    try:
        runner_returncode = process.wait()
    except KeyboardInterrupt:
        # The foreground child receives the same SIGINT and performs its
        # controlled arm return/release. Do not start packaging while that
        # safety path is still running.
        try:
            runner_returncode = process.wait(timeout=45.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            runner_returncode = process.wait(timeout=10.0)
    package = subprocess.run(
        [
            str(python),
            "-m",
            "inference.desktop.model_evaluation.package_recording",
            str(run_dir / "capture"),
            "--delete-frames",
        ],
        env=env,
        check=False,
    )
    decision = evaluate_run(
        run_dir,
        runner_returncode=runner_returncode,
        actuated=args.actuate,
        requested_max_seconds=args.max_seconds,
        minimum_seconds=args.wandb_min_seconds,
    )
    metadata["finished_at_jst"] = datetime.now(
        ZoneInfo("Asia/Tokyo")
    ).isoformat(timespec="seconds")
    metadata["video_packager_returncode"] = package.returncode
    write_summary(run_dir, metadata=metadata, decision=decision)
    if not decision["eligible"]:
        status = {
            "uploaded": False,
            "status": "not_eligible",
            "reasons": decision["reasons"],
        }
        write_upload_status(run_dir, status)
        print(
            "[wandb] not uploaded: " + ", ".join(decision["reasons"]),
            flush=True,
        )
    elif args.no_wandb:
        status = {"uploaded": False, "status": "disabled_by_flag"}
        write_upload_status(run_dir, status)
        print("[wandb] eligible run retained locally; upload disabled", flush=True)
    else:
        try:
            status = upload_run(
                run_dir,
                entity=args.wandb_entity,
                project=args.wandb_project,
            )
        except Exception as exc:  # noqa: BLE001
            status = {
                "uploaded": False,
                "status": "upload_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        write_upload_status(run_dir, status)
        if status.get("uploaded"):
            print(f"[wandb] uploaded: {status['url']}", flush=True)
        else:
            print(
                f"[wandb] eligible run retained for retry: {status['status']}",
                flush=True,
            )
    return runner_returncode


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "policy"


def _source_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
