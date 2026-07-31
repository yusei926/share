from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from data.flip_table_data_augmentation.teleop.raw_episode import EpisodeIdentity
from data.flip_table_data_augmentation.teleop.real.lossless_camera import (
    CameraFrameEnvelope,
    ClockSyncSample,
    LosslessCameraRecorder,
    RecorderControlClient,
    RecorderControlServer,
    audit_camera_mcap,
    estimate_clock_mapping,
    read_camera_mcap,
)
import data.flip_table_data_augmentation.teleop.real.lossless_camera as lossless_camera
from data.flip_table_data_augmentation.teleop.real.lossless_episode import (
    _nearest_unique_matches,
    materialize_lossless_real_episode,
    start_lossless_real_episode_async,
)
from data.flip_table_data_augmentation.teleop.real.backend import (
    camera_bundle_status_from_histories,
)
from data.flip_table_data_augmentation.teleop.real.teleimager import (
    LatestCameraSample,
    LatestCameraTracker,
)


def _jpeg(width: int, height: int, value: int) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(
        stream, format="JPEG", quality=80
    )
    return stream.getvalue()


def _frame(
    role: str,
    sequence: int,
    timestamp_ns: int,
    jpeg: bytes,
) -> CameraFrameEnvelope:
    return CameraFrameEnvelope(
        role=role,
        usb_serial=f"serial-{role}",
        source_sequence=sequence,
        orin_capture_monotonic_ns=timestamp_ns,
        device_frame_counter=sequence if role != "head_stereo" else None,
        device_timestamp=sequence / 30.0,
        timestamp_domain="hardware_clock",
        jpeg=jpeg,
    )


def test_camera_envelope_round_trip_and_digest_rejection() -> None:
    original = _frame("head_stereo", 7, 123_000, b"\xff\xd8jpeg\xff\xd9")
    encoded = original.encode()
    assert CameraFrameEnvelope.decode(encoded) == original
    corrupt = bytearray(encoded)
    corrupt[-3] ^= 1
    with pytest.raises(ValueError, match="digest"):
        CameraFrameEnvelope.decode(bytes(corrupt))


def test_clock_sync_selects_minimum_rtt_samples() -> None:
    samples = [
        ClockSyncSample(1_000, 11_100, 11_200, 2_200),
        ClockSyncSample(2_000, 12_050, 12_100, 2_150),
        ClockSyncSample(3_000, 13_040, 13_090, 3_140),
        ClockSyncSample(4_000, 14_030, 14_080, 4_130),
        ClockSyncSample(5_000, 25_000, 25_100, 15_100),
    ]
    mapping = estimate_clock_mapping(samples, best_sample_count=4)
    assert len(mapping.samples) == 4
    assert mapping.uncertainty_ns <= 550
    assert 9_900 <= mapping.desktop_to_orin_offset_ns <= 10_100


def test_latest_camera_tracker_retains_unique_transition_ring() -> None:
    class Value:
        bgr = None
        fps = 30.0

        def __init__(self, jpg: bytes) -> None:
            self.jpg = jpg

    values = iter(
        [
            Value(b"a"),
            Value(b"a"),
            Value(b"b"),
            Value(b"c"),
            Value(b"d"),
            Value(b"e"),
            Value(b"f"),
            Value(b"g"),
            Value(b"h"),
            Value(b"i"),
        ]
    )
    tracker = LatestCameraTracker("head", lambda: next(values), history_size=8)
    for _ in range(10):
        tracker.poll()
    assert [sample.jpg for sample in tracker.history] == [
        bytes([value]) for value in b"bcdefghi"
    ]
    assert [sample.jpeg_generation for sample in tracker.samples_after(7)] == [8, 9]


def test_120hz_ring_matching_keeps_ten_minutes_of_phase_shifted_30hz() -> None:
    count = 18_000
    period_ns = 1_000_000_000 / 30.0
    head = np.rint(np.arange(count) * period_ns).astype(np.int64)
    left = head + 8_000_000
    right = head - 7_000_000
    left_match, left_error = _nearest_unique_matches(head, left)
    right_match, right_error = _nearest_unique_matches(head, right)
    assert np.array_equal(left_match, np.arange(count))
    assert np.array_equal(right_match, np.arange(count))
    assert np.max(left_error) == 8_000_000
    assert np.max(right_error) == 7_000_000


def test_preview_history_matcher_consumes_each_generation_once() -> None:
    def sample(role: str, generation: int, timestamp_ns: int) -> LatestCameraSample:
        return LatestCameraSample(
            role=role,
            jpg=f"{role}-{generation}".encode(),
            source_fps=30.0,
            jpeg_generation=generation,
            first_observed_monotonic_ns=timestamp_ns,
            transition_hz=30.0,
        )

    base = 1_000_000_000
    histories = {
        "head": tuple(sample("head", i, base + i * 33_333_333) for i in range(1, 4)),
        "left_wrist": tuple(
            sample("left_wrist", i, base + i * 33_333_333 + 5_000_000)
            for i in range(1, 4)
        ),
        "right_wrist": tuple(
            sample("right_wrist", i, base + i * 33_333_333 - 6_000_000)
            for i in range(1, 4)
        ),
    }
    baseline = {role: 0 for role in histories}
    consumed = []
    for _ in range(3):
        valid, skew_ms, baseline = camera_bundle_status_from_histories(
            histories, baseline, camera_hz=30.0
        )
        assert valid
        assert skew_ms == 11.0
        consumed.append(dict(baseline))
    assert consumed == [
        {"head": 1, "left_wrist": 1, "right_wrist": 1},
        {"head": 2, "left_wrist": 2, "right_wrist": 2},
        {"head": 3, "left_wrist": 3, "right_wrist": 3},
    ]
    valid, _skew, unchanged = camera_bundle_status_from_histories(
        histories, baseline, camera_hz=30.0
    )
    assert not valid
    assert unchanged == baseline


def test_recorder_control_and_mcap_sequence_failure_are_diagnostic(
    tmp_path: Path,
) -> None:
    recorder = LosslessCameraRecorder(tmp_path, queue_capacity=90)
    server = RecorderControlServer(recorder, host="127.0.0.1", port=0)
    server.start()
    client = RecorderControlClient("127.0.0.1", port=server.port)
    try:
        assert client.status()["active"] is False
        assert client.synchronize().uncertainty_ns < 10_000_000
        client.start("sequence_gap")
        payloads = {}
        for role in ("head_stereo", "left_wrist", "right_wrist"):
            payloads[role] = (
                _jpeg(1280, 480, 10)
                if role == "head_stereo"
                else _jpeg(640, 480, 20)
            )
            assert recorder.capture(
                _frame(role, 1, 1_000_000_000, payloads[role])
            )
        for role in ("head_stereo", "left_wrist", "right_wrist"):
            assert recorder.capture(
                _frame(role, 2, 1_033_333_333, payloads[role])
            )
        recorder.capture(
            _frame("head_stereo", 4, 1_066_666_667, payloads["head_stereo"])
        )
        result = client.stop()
        assert "source_sequence_gap" in str(result["failed"])
        retried = client.stop()
        assert retried["path"] == result["path"]
        assert client.status()["last_result"]["sha256"] == result["sha256"]
        streams = read_camera_mcap(
            result["path"], require_contiguous_sequences=False
        )
        assert all(len(frames) >= 2 for frames in streams.values())
        audit = audit_camera_mcap(result["path"])
        assert audit["counts"] == {
            role: len(frames) for role, frames in streams.items()
        }
        assert audit["source_sequence_gaps"]["head_stereo"] == 1
        assert audit["bundle_valid_fraction"] == pytest.approx(2 / 3)
        assert audit["camera_match"]["left_wrist"]["maximum_error_ms"] == 0.0
        assert audit["camera_match"]["right_wrist"]["maximum_error_ms"] == 0.0
        with pytest.raises(ValueError, match="not contiguous"):
            read_camera_mcap(result["path"])
    finally:
        server.close()
        recorder.close()


def test_recorder_refuses_low_disk_without_creating_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = LosslessCameraRecorder(tmp_path, queue_capacity=90)
    monkeypatch.setattr(
        lossless_camera.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=20 * 1024**3,
            used=19 * 1024**3,
            free=1 * 1024**3,
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="insufficient"):
            recorder.start("no_space")
        assert not list(tmp_path.glob("*.partial"))
        assert not list(tmp_path.glob("*.mcap"))
    finally:
        recorder.close()


def test_real_recording_start_and_discard_do_not_block_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = LosslessCameraRecorder(tmp_path / "orin", queue_capacity=90)
    server = RecorderControlServer(recorder, host="127.0.0.1", port=0)
    server.start()
    gate = threading.Event()
    original = RecorderControlClient.synchronize

    def delayed_synchronize(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert gate.wait(timeout=2.0)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        RecorderControlClient, "synchronize", delayed_synchronize
    )
    identity = EpisodeIdentity(
        backend="real",
        dr_profile="real",
        seed=42,
        config_sha256="a" * 64,
        runtime_digest="b" * 40,
    )
    try:
        future = start_lossless_real_episode_async(
            tmp_path / "raw",
            identity,
            recorder_host="127.0.0.1",
            ssh_target="unused",
            recorder_port=server.port,
        )
        assert not future.done()
        gate.set()
        writer = future.result(timeout=3.0)
        discard = writer.discard_async()
        assert discard.result(timeout=3.0) is None
        assert recorder.status()["active"] is False
    finally:
        server.close()
        recorder.close()


def test_materializer_emits_exact_30hz_unique_valid_frames(
    tmp_path: Path,
) -> None:
    pending = tmp_path / "raw" / ".episode.recording"
    pending.mkdir(parents=True)
    remote_root = tmp_path / "orin"
    recorder = LosslessCameraRecorder(remote_root, queue_capacity=900)
    head_jpeg = _jpeg(1280, 480, 60)
    left_jpeg = _jpeg(640, 480, 90)
    right_jpeg = _jpeg(640, 480, 120)
    count = 90
    base = 10_000_000_000
    period = 1_000_000_000 / 30.0
    recorder.start("episode")
    for index in range(count):
        sequence = index + 1
        head_ns = base + round(index * period)
        assert recorder.capture(
            _frame("head_stereo", sequence, head_ns, head_jpeg)
        )
        assert recorder.capture(
            _frame("left_wrist", sequence, head_ns + 5_000_000, left_jpeg)
        )
        assert recorder.capture(
            _frame("right_wrist", sequence, head_ns - 6_000_000, right_jpeg)
        )
    remote = recorder.stop()
    recorder.close()
    shutil.copy2(remote["path"], pending / "cameras.mcap")

    rows = []
    for index in range(count):
        timestamp = base - 1_000_000 + round(index * period)
        rows.append(
            {
                "sequence": index + 1,
                "desktop_monotonic_ns": timestamp,
                "orin_estimated_ns": timestamp,
                "clock_mapping_index": 0,
                "body_joint_position_rad": [0.0] * 29,
                "body_joint_velocity_rad_s": [0.0] * 29,
                "dex1_opening_state": [0.5, 0.5],
                "applied_arm_target_rad": [0.0] * 14,
                "applied_dex1_opening_target": [0.5, 0.5],
                "root_pose_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "commanded_arm_target_rad": [0.0] * 14,
                "commanded_arm_feedforward_torque_nm": [0.0] * 14,
                "commanded_dex1_opening_target": [0.5, 0.5],
                "control_mode": "hold" if index == 10 else "track",
                "control_event": "none",
            }
        )
    with (pending / "numeric.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    identity = EpisodeIdentity(
        backend="real",
        dr_profile="real",
        seed=42,
        config_sha256="a" * 64,
        runtime_digest="b" * 40,
    )
    destination = materialize_lossless_real_episode(
        pending,
        identity=identity,
        episode_id="episode",
        diagnostics={
            "clock_uncertainty_ms_p95": 0.1,
            "clock_sync_error": None,
            "orin_recorder_status": remote,
        },
        operator_success=True,
    )
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["success"] is True
    assert manifest["fps"] == 30
    assert manifest["diagnostics"]["camera_valid_fraction"] == 1.0
    lines = [
        json.loads(line)
        for line in (destination / "frames.jsonl").read_text().splitlines()
    ]
    assert all(
        row["canonical_timestamp_s"] == index / 30.0
        for index, row in enumerate(lines)
    )
    assert any(row["control_mode"] == "hold" for row in lines)
    for key in (
        "observation.images.cam_0",
        "observation.images.cam_2",
        "observation.images.cam_3",
    ):
        sequences = [row["policy_cameras"][key]["source_sequence"] for row in lines]
        assert len(sequences) == len(set(sequences))
        assert all(row["policy_cameras"][key]["valid"] for row in lines)
