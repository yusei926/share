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
    # Repeated high-rate polling is reduced to the 30 Hz state contract while
    # every unique physical camera generation remains losslessly captured.
    assert report["observation_count"] == 2
    assert report["backend_action_count"] == 1
    assert report["camera_frame_count"] == {role: 2 for role in CAMERA_ROLES}
    assert len((tmp_path / "states.jsonl").read_text().splitlines()) == 2
    assert len((tmp_path / "backend_actions.jsonl").read_text().splitlines()) == 1
    frames = [json.loads(line) for line in (tmp_path / "camera_frames.jsonl").read_text().splitlines()]
    assert len(frames) == 8
    assert all((tmp_path / row["relative_path"]).is_file() for row in frames)


def test_recorder_bounds_busy_poll_state_rate_without_dropping_camera_frames(
    tmp_path: Path,
) -> None:
    recorder = RealEvaluationRecorder(tmp_path)
    for sequence in range(1000):
        # Ten observe() calls per 30 Hz camera period reproduce policy-loader
        # busy polling without manufacturing thousands of duplicate states.
        generation = sequence // 10
        observation = _observation(sequence, generation)
        observation.capture_monotonic_ns = (
            1_000_000_000 + sequence * 3_333_333
        )
        recorder.record_observation(observation)
    report = recorder.close()

    assert 90 <= report["observation_count"] <= 100
    assert report["camera_frame_count"] == {
        role: 100 for role in CAMERA_ROLES
    }


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
    assert status == "LIVE CAMERA"
    assert preview.closed is True
    assert report["complete"] is True
    assert report["desktop_preview_errors"] == []


def test_environment_recorder_previews_entire_run_but_captures_only_policy_interval(
    tmp_path: Path, monkeypatch
) -> None:
    preview = _FakePreview()
    monkeypatch.setenv(recording_module.CAPTURE_ENV, str(tmp_path))
    monkeypatch.setenv(recording_module.PREVIEW_ENV, "true")
    monkeypatch.setattr(
        recording_module,
        "DesktopPreviewProcess",
        lambda **_kwargs: preview,
    )
    recorder = RealEvaluationRecorder.from_environment()
    assert recorder is not None

    # Model loading/pre-motion is visible, but absent from the dataset.
    recorder.record_observation(
        _observation(1, 1), preview_status="PRE-MOTION"
    )
    assert recorder.start_capture() is True
    recorder.record_observation(
        _observation(2, 2), preview_status="POLICY INFERENCE"
    )
    recorder.record_action(
        SimpleNamespace(
            to_message=lambda: {
                "type": "command",
                "sequence": 9,
                "monotonic_ns": 2_000_000_000,
                "mode": "track",
                "event": "none",
                "arm_position_rad": [0.1] * 14,
                "dex1_opening_fraction": [0.2, 0.8],
                "arm_feedforward_torque_nm": [0.0] * 14,
            }
        )
    )
    assert recorder.stop_capture() is True
    # Reverse return remains visible and is excluded from files.
    recorder.record_observation(
        _observation(3, 3), preview_status="SAFE ARM RETURN"
    )
    report = recorder.close()

    assert [update[2] for update in preview.updates] == [
        "PRE-MOTION",
        "POLICY INFERENCE",
        "SAFE ARM RETURN",
    ]
    assert report["recording_started"] is True
    assert report["recording_active"] is False
    assert report["recording_duration_s"] is not None
    assert report["observation_count"] == 1
    assert report["backend_action_count"] == 1
    assert report["camera_frame_count"] == {
        role: 1 for role in CAMERA_ROLES
    }
    states = [
        json.loads(line)
        for line in (tmp_path / "states.jsonl").read_text().splitlines()
    ]
    actions = [
        json.loads(line)
        for line in (tmp_path / "backend_actions.jsonl").read_text().splitlines()
    ]
    assert [state["sequence"] for state in states] == [2]
    assert len(states[0]["body_joint_position_rad"]) == 29
    assert len(states[0]["applied_arm_target_rad"]) == 14
    assert actions[0]["sequence"] == 9
    assert len(actions[0]["arm_position_rad"]) == 14
    assert len(actions[0]["dex1_opening_fraction"]) == 2


def test_policy_capture_interval_cannot_be_appended_twice(tmp_path: Path) -> None:
    recorder = RealEvaluationRecorder(
        tmp_path,
        capture_initially_active=False,
    )
    assert recorder.start_capture() is True
    assert recorder.stop_capture() is True
    try:
        recorder.start_capture()
    except RuntimeError as exc:
        assert "already been completed" in str(exc)
    else:  # pragma: no cover - explicit fail message is clearer than pytest.raises
        raise AssertionError("a second policy interval must be rejected")
    recorder.close()


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


def test_preview_runs_without_evaluation_capture_directory(
    monkeypatch,
) -> None:
    preview = _FakePreview()
    monkeypatch.delenv(recording_module.CAPTURE_ENV, raising=False)
    monkeypatch.setenv(recording_module.PREVIEW_ENV, "true")
    monkeypatch.setattr(
        recording_module,
        "DesktopPreviewProcess",
        lambda **_kwargs: preview,
    )

    recorder = RealEvaluationRecorder.from_environment()
    assert recorder is not None
    recorder.record_observation(_observation(1, 1), preview_status="LOADING POLICY")
    report = recorder.close()

    assert preview.updates[0][2] == "LOADING POLICY"
    assert preview.closed is True
    assert report["capture_enabled"] is False
    assert report["observation_count"] == 0
    assert report["complete"] is True


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
