from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from inference.desktop.model_evaluation import recording as recording_module
from inference.desktop.model_evaluation.recording import (
    CAMERA_ROLES,
    RealEvaluationRecorder,
)
from inference.desktop.model_evaluation.wandb_export import (
    _wandb_run_id,
    evaluate_run,
    run_name,
)


def _observation(sequence: int, generation: int) -> SimpleNamespace:
    timestamp = 1_000_000_000 + generation * 33_333_333
    return SimpleNamespace(
        sequence=sequence,
        capture_monotonic_ns=timestamp,
        body_joint_position_rad=tuple(float(i) / 100 for i in range(29)),
        body_joint_velocity_rad_s=(0.0,) * 29,
        dex1_opening_fraction=(0.25, 0.75),
        applied_arm_target_rad=(0.0,) * 14,
        applied_dex1_opening_target=(0.25, 0.75),
        camera_capture_monotonic_ns={role: timestamp for role in CAMERA_ROLES},
        camera_jpeg={role: b"jpeg-" + role.encode() for role in CAMERA_ROLES},
        camera_bundle_valid=True,
        camera_skew_ms=0.0,
        stale_roles=(),
        camera_stream_metadata={
            role: {"jpeg_generation": generation, "source_fps": 30.0}
            for role in CAMERA_ROLES
        },
    )


def test_recorder_deduplicates_camera_generations_and_keeps_numeric_logs(
    tmp_path: Path,
) -> None:
    recorder = RealEvaluationRecorder(tmp_path)
    recorder.record_observation(_observation(1, 1))
    recorder.record_observation(_observation(2, 1))
    recorder.record_observation(_observation(3, 2))
    recorder.record_action(
        SimpleNamespace(
            to_message=lambda: {
                "type": "command",
                "sequence": 1,
                "monotonic_ns": 2_000_000_000,
                "mode": "track",
                "event": "none",
                "arm_position_rad": [0.0] * 14,
                "dex1_opening_fraction": [0.25, 0.75],
                "arm_feedforward_torque_nm": [0.0] * 14,
            }
        )
    )
    report = recorder.close()
    assert report["complete"] is True
    assert report["observation_count"] == 3
    assert report["backend_action_count"] == 1
    assert report["camera_frame_count"] == {role: 2 for role in CAMERA_ROLES}
    assert len((tmp_path / "states.jsonl").read_text().splitlines()) == 3
    assert len((tmp_path / "backend_actions.jsonl").read_text().splitlines()) == 1
    frames = [json.loads(line) for line in (tmp_path / "camera_frames.jsonl").read_text().splitlines()]
    assert len(frames) == 8
    assert all((tmp_path / row["relative_path"]).is_file() for row in frames)


class _FakePreview:
    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, bytes], list[float], str]] = []
        self.closed = False

    def submit(
        self,
        camera_jpeg: dict[str, bytes],
        arm_joint_position_rad: list[float],
        status: str,
    ) -> None:
        self.updates.append((camera_jpeg, arm_joint_position_rad, status))

    def close(self) -> None:
        self.closed = True


def test_recorder_feeds_four_camera_preview_without_changing_capture(
    tmp_path: Path,
) -> None:
    preview = _FakePreview()
    recorder = RealEvaluationRecorder(tmp_path, preview=preview)
    observation = _observation(1, 1)
    recorder.record_observation(observation)
    report = recorder.close()

    assert len(preview.updates) == 1
    camera_jpeg, arm_joint_position_rad, status = preview.updates[0]
    assert tuple(camera_jpeg) == CAMERA_ROLES
    assert arm_joint_position_rad == list(observation.body_joint_position_rad[15:29])
    assert status == "MODEL EVALUATION"
    assert preview.closed is True
    assert report["complete"] is True
    assert report["desktop_preview_errors"] == []


def test_broken_desktop_gui_does_not_prevent_capture(
    tmp_path: Path, monkeypatch
) -> None:
    def fail_preview(**_kwargs):
        raise RuntimeError("GUI unavailable")

    monkeypatch.setenv(recording_module.CAPTURE_ENV, str(tmp_path))
    monkeypatch.setenv(recording_module.PREVIEW_ENV, "true")
    monkeypatch.setattr(recording_module, "DesktopPreviewProcess", fail_preview)
    recorder = RealEvaluationRecorder.from_environment()
    assert recorder is not None
    recorder.record_observation(_observation(1, 1))
    report = recorder.close()
    assert report["complete"] is True
    assert report["desktop_preview_errors"] == ["RuntimeError: GUI unavailable"]


def _candidate_run(tmp_path: Path, *, span_s: float = 9.8) -> Path:
    run = tmp_path / "run"
    capture = run / "capture"
    capture.mkdir(parents=True)
    events = [
        {"event": "command", "monotonic_ns": 1_000_000_000, "sequence": 1},
        {
            "event": "command",
            "monotonic_ns": 1_000_000_000 + int(span_s * 1e9),
            "sequence": 2,
        },
        {"event": "return_motion_complete", "monotonic_ns": 20_000_000_000},
    ]
    (run / "events.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in events)
    )
    (capture / "capture_report.json").write_text(json.dumps({"complete": True}))
    (capture / "video_manifest.json").write_text(json.dumps({"complete": True}))
    return run


def test_upload_gate_accepts_completed_long_task_failure_run(tmp_path: Path) -> None:
    run = _candidate_run(tmp_path)
    decision = evaluate_run(
        run,
        runner_returncode=0,
        actuated=True,
        requested_max_seconds=10.0,
        minimum_seconds=10.0,
    )
    assert decision["eligible"] is True
    # Deliberately no task-success field: task failure is still uploadable.
    assert decision["reasons"] == []


def test_upload_gate_rejects_short_error_or_incomplete_return(tmp_path: Path) -> None:
    run = _candidate_run(tmp_path, span_s=0.97)
    assert not evaluate_run(
        run,
        runner_returncode=0,
        actuated=True,
        requested_max_seconds=1.0,
    )["eligible"]
    assert not evaluate_run(
        run,
        runner_returncode=1,
        actuated=True,
        requested_max_seconds=30.0,
    )["eligible"]
    lines = (run / "events.jsonl").read_text().splitlines()
    (run / "events.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    decision = evaluate_run(
        run,
        runner_returncode=0,
        actuated=True,
        requested_max_seconds=30.0,
    )
    assert "safe_return_not_confirmed" in decision["reasons"]


def test_wandb_run_name_is_readable_and_stable() -> None:
    metadata = {
        "task": "pick table leg",
        "model_id": "pick_legs_groot_v2_lora",
        "model_repo_id": "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora",
        "model_revision": "1" * 40,
        "run_id": "20260731_204958_pick_legs_groot_v2_lora",
        "started_at_jst": "2026-07-31T20:49:58+09:00",
    }
    assert run_name(metadata) == (
        "pick-table-leg--pick-legs-groot-v2-lora--20260731-204958"
    )
    assert _wandb_run_id(metadata) == _wandb_run_id(dict(metadata))
    assert len(_wandb_run_id(metadata)) == 20
