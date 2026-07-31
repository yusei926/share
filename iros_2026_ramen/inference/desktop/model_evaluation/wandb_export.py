"""Eligibility-gated W&B export for completed physical policy evaluations."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from datetime import datetime
import hashlib
import json
import netrc
import os
from pathlib import Path
import re
from typing import Any, Iterable


DEFAULT_WANDB_ENTITY = "ken05-matuo-llm-88_llm_2025_suzuki"
DEFAULT_WANDB_PROJECT = "iros-2026-ramen-real-policy-evaluation"
DEFAULT_MINIMUM_SECONDS = 10.0
SUMMARY_SCHEMA = "team_ramen_real_policy_evaluation/v1"

ARM_NAMES = (
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
BODY_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
    "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
    "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    *ARM_NAMES,
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_run(
    run_dir: Path,
    *,
    runner_returncode: int,
    actuated: bool,
    requested_max_seconds: float | None,
    minimum_seconds: float = DEFAULT_MINIMUM_SECONDS,
) -> dict[str, Any]:
    """Return a deterministic upload decision; task success is not required."""

    if minimum_seconds <= 0:
        raise ValueError("minimum W&B duration must be positive")
    events = _read_jsonl(run_dir / "events.jsonl")
    commands = [event for event in events if event.get("event") == "command"]
    command_times = [int(event["monotonic_ns"]) for event in commands]
    action_span_s = (
        0.0
        if len(command_times) < 2
        else (command_times[-1] - command_times[0]) / 1e9
    )
    return_complete = any(
        event.get("event") == "return_motion_complete" for event in events
    )
    capture = _read_json(run_dir / "capture" / "capture_report.json", {})
    videos = _read_json(run_dir / "capture" / "video_manifest.json", {})
    reasons = []
    if not actuated:
        reasons.append("read_only_preflight")
    if runner_returncode != 0:
        reasons.append(f"runner_returncode_{runner_returncode}")
    if requested_max_seconds is not None and requested_max_seconds < minimum_seconds:
        reasons.append("requested_duration_below_threshold")
    # A completed N-second deadline normally has a final command one control
    # period before N. The 250 ms tolerance accepts that scheduling edge while
    # still excluding 1 s and 5 s tests by a wide margin.
    if action_span_s < maximum_acceptable_span(minimum_seconds):
        reasons.append("observed_policy_duration_below_threshold")
    if not commands:
        reasons.append("no_policy_commands")
    if not return_complete:
        reasons.append("safe_return_not_confirmed")
    if not capture.get("complete", False):
        reasons.append("capture_incomplete")
    if not videos.get("complete", False):
        reasons.append("camera_video_incomplete")
    error_events = [
        event.get("event")
        for event in events
        if "error" in str(event.get("event", "")).lower()
    ]
    if error_events:
        reasons.append("error_event_present")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "minimum_policy_seconds": minimum_seconds,
        "requested_max_seconds": requested_max_seconds,
        "observed_policy_action_span_s": action_span_s,
        "policy_command_count": len(commands),
        "safe_return_complete": return_complete,
        "runner_returncode": runner_returncode,
        "capture_complete": bool(capture.get("complete", False)),
        "camera_video_complete": bool(videos.get("complete", False)),
    }


def maximum_acceptable_span(minimum_seconds: float) -> float:
    return max(0.0, minimum_seconds - 0.25)


def run_name(metadata: dict[str, Any]) -> str:
    task = _slug(str(metadata.get("task", "manipulation")))
    model = _slug(str(metadata.get("model_id", "policy")))
    timestamp = str(metadata.get("started_at_jst", ""))
    try:
        stamp = datetime.fromisoformat(timestamp).strftime("%Y%m%d-%H%M%S")
    except ValueError:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{task}--{model}--{stamp}"[:128]


def _slug(value: str) -> str:
    value = value.lower().replace("_", "-")
    value = re.sub(r"[^a-z0-9-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "run"


def _api_key_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(auth and auth[2])


def _nearest_state(
    states: list[dict[str, Any]], times: list[int], target_ns: int
) -> dict[str, Any] | None:
    if not states:
        return None
    index = bisect_left(times, target_ns)
    candidates = [max(0, min(len(states) - 1, index))]
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(times[item] - target_ns))
    return states[best]


def _downsample(rows: list[dict[str, Any]], maximum: int = 1000) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return rows
    indices = {round(i * (len(rows) - 1) / (maximum - 1)) for i in range(maximum)}
    return [rows[index] for index in sorted(indices)]


def upload_run(
    run_dir: Path,
    *,
    entity: str = DEFAULT_WANDB_ENTITY,
    project: str = DEFAULT_WANDB_PROJECT,
) -> dict[str, Any]:
    """Upload an already-eligible run without changing its eligibility."""

    run_dir = run_dir.expanduser().resolve()
    summary = _read_json(run_dir / "run_summary.json", {})
    if not summary.get("upload_decision", {}).get("eligible", False):
        raise ValueError("run_summary does not mark this run eligible")
    if not _api_key_available():
        return {
            "uploaded": False,
            "status": "pending_wandb_login",
            "message": "run `wandb login` once, then retry this run",
        }

    import wandb

    metadata = summary["metadata"]
    capture = _read_json(run_dir / "capture" / "capture_report.json", {})
    video_manifest = _read_json(
        run_dir / "capture" / "video_manifest.json", {}
    )
    events = _read_jsonl(run_dir / "events.jsonl")
    commands = [event for event in events if event.get("event") == "command"]
    states = _read_jsonl(run_dir / "capture" / "states.jsonl")
    states.sort(key=lambda row: int(row["capture_monotonic_ns"]))
    state_times = [int(row["capture_monotonic_ns"]) for row in states]

    wb = wandb.init(
        entity=entity,
        project=project,
        id=_wandb_run_id(metadata),
        resume="allow",
        name=run_name(metadata),
        group=_slug(str(metadata.get("task", "manipulation"))),
        job_type="real-robot-policy-evaluation",
        tags=[
            "real-robot",
            "arm-only",
            _slug(str(metadata.get("task", "manipulation"))),
            _slug(str(metadata.get("family", "policy"))),
        ],
        config={
            **metadata,
            "lower_body_command_dimensions": 0,
            "regular_mode_owns_lower_body": True,
            "upload_minimum_policy_seconds": summary["upload_decision"][
                "minimum_policy_seconds"
            ],
        },
        settings=wandb.Settings(init_timeout=30),
    )
    try:
        wb.summary.update(
            {
                "policy/action_span_s": summary["upload_decision"][
                    "observed_policy_action_span_s"
                ],
                "policy/command_count": len(commands),
                "safety/safe_return_complete": True,
                "safety/lower_body_command_dimensions": 0,
                "capture/observation_count": capture.get("observation_count", 0),
                "capture/queue_drop_count": capture.get("queue_drop_count", 0),
                **{
                    f"camera/{role}/frame_count": count
                    for role, count in capture.get("camera_frame_count", {}).items()
                },
                **{
                    f"camera/{role}/effective_hz": hz
                    for role, hz in capture.get("camera_effective_hz", {}).items()
                },
            }
        )
        media = {}
        for role, video in video_manifest.get("videos", {}).items():
            path = run_dir / "capture" / video["relative_path"]
            media[f"camera/{role}"] = wandb.Video(
                str(path), format="mp4"
            )
        if media:
            wb.log(media)

        trace_rows = []
        for event in _downsample(commands):
            timestamp = int(event["monotonic_ns"])
            state = _nearest_state(states, state_times, timestamp)
            row = {
                "policy_time_s": (timestamp - int(commands[0]["monotonic_ns"])) / 1e9,
                "sequence": int(event.get("sequence", 0)),
            }
            arm = event.get("arm_target_rad", ())
            hand = event.get("dex1_target_fraction", ())
            row.update({f"action/{name}": value for name, value in zip(ARM_NAMES, arm)})
            row.update(
                {
                    "action/dex1_left": hand[0] if len(hand) == 2 else None,
                    "action/dex1_right": hand[1] if len(hand) == 2 else None,
                }
            )
            if state is not None:
                row.update(
                    {
                        f"state/{name}": value
                        for name, value in zip(
                            BODY_NAMES, state["body_joint_position_rad"]
                        )
                    }
                )
                row["state/dex1_left"] = state["dex1_opening_fraction"][0]
                row["state/dex1_right"] = state["dex1_opening_fraction"][1]
                row["camera/skew_ms"] = state["camera_skew_ms"]
            trace_rows.append(row)
        if trace_rows:
            columns = sorted({key for row in trace_rows for key in row})
            table = wandb.Table(
                columns=columns,
                data=[[row.get(column) for column in columns] for row in trace_rows],
            )
            wb.log({"telemetry/state_action_trace": table})

        artifact = wandb.Artifact(
            name=f"{_slug(str(metadata['model_id']))}-{metadata['run_id']}",
            type="real-policy-evaluation",
            metadata={
                "model_repo_id": metadata["model_repo_id"],
                "model_revision": metadata["model_revision"],
                "task": metadata["task"],
                "policy_action_span_s": summary["upload_decision"][
                    "observed_policy_action_span_s"
                ],
                "task_success_required_for_upload": False,
            },
        )
        for path in run_dir.rglob("*"):
            if path.is_file() and "frames" not in path.relative_to(run_dir).parts:
                artifact.add_file(
                    str(path), name=path.relative_to(run_dir).as_posix()
                )
        wb.log_artifact(artifact)
        url = wb.url
        offline = str(getattr(wb.settings, "mode", "online")) == "offline"
    finally:
        wb.finish()
    if offline:
        return {
            "uploaded": False,
            "status": "offline_staged",
            "url": None,
        }
    return {"uploaded": True, "status": "uploaded", "url": url}


def _wandb_run_id(metadata: dict[str, Any]) -> str:
    identity = ":".join(
        str(metadata.get(key, ""))
        for key in ("model_repo_id", "model_revision", "run_id")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def write_summary(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    decision: dict[str, Any],
) -> Path:
    path = run_dir / "run_summary.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SUMMARY_SCHEMA,
                "metadata": metadata,
                "upload_decision": decision,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def write_upload_status(run_dir: Path, value: dict[str, Any]) -> Path:
    path = run_dir / "wandb_upload.json"
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    args = parser.parse_args()
    result = upload_run(args.run_dir, entity=args.entity, project=args.project)
    write_upload_status(args.run_dir, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("uploaded") else 2


if __name__ == "__main__":
    raise SystemExit(main())
