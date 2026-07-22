from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data.flip_table_data_augmentation.annotations import (
    ANNOTATION_SCHEMA_VERSION,
    SourceEpisodeAnnotation,
)
from data.flip_table_data_augmentation.config import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_SUBTASKS,
    load_pipeline_config,
)
from data.flip_table_data_augmentation.fk_audit import (
    select_frame_indices,
    synthetic_action_fk_report,
)
from data.flip_table_data_augmentation.io_utils import (
    atomic_write_json,
    read_json_object,
    sha256_file,
)
from data.flip_table_data_augmentation.provenance import (
    AppearanceRecord,
    CandidateLedger,
    CandidateRecord,
)
from data.flip_table_data_augmentation.object_pose.artifacts import artifact_specs, verify_artifact
from data.flip_table_data_augmentation.raw_source_contract import (
    audit_raw_source_bindings,
)
from data.flip_table_data_augmentation.runtime_contract import stable_tree_sha256
from data.flip_table_data_augmentation.source_contract import (
    NUMERIC_FEATURES,
    download_pinned_source_files,
    validate_source_info,
)
from data.flip_table_data_augmentation.source_dataset import SourceEpisode, select_review_frames


class DurableArtifactIoTest(unittest.TestCase):
    def test_atomic_json_round_trip_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "artifact.json"
            atomic_write_json(path, {"finite": 1.25, "values": [1, 2]})

            self.assertEqual(
                read_json_object(path, label="artifact"),
                {"finite": 1.25, "values": [1, 2]},
            )
            self.assertEqual(sha256_file(path), hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(list(path.parent.glob(".*.tmp")), [])

    def test_atomic_json_rejects_non_finite_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"invalid": float("nan")})
            self.assertFalse(path.exists())

    def test_json_object_reader_rejects_non_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact must contain a JSON object"):
                read_json_object(path, label="artifact")


class SourceMediaResolutionTest(unittest.TestCase):
    def test_review_sampling_includes_episode_endpoints(self) -> None:
        frames = select_review_frames(100, 9)
        self.assertEqual((frames[0], frames[-1], len(frames)), (0, 99, 9))
        self.assertEqual(tuple(sorted(set(frames))), frames)
        self.assertEqual(select_review_frames(100, 1), (0,))

    def test_resolves_episode_video_shard_and_reuses_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = Path("videos/observation.images.cam_0/chunk-002/file-003.mp4")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"video")
            episode = SourceEpisode(
                root,
                {
                    "episode_index": 7,
                    "length": 30,
                    "videos/observation.images.cam_0/chunk_index": 2,
                    "videos/observation.images.cam_0/file_index": 3,
                    "videos/observation.images.cam_0/from_timestamp": 1.0,
                    "videos/observation.images.cam_0/to_timestamp": 2.0,
                },
            )
            self.assertEqual(episode.video_relative_path("observation.images.cam_0"), relative)
            self.assertEqual(episode.video_slice("observation.images.cam_0").path, path)
            self.assertEqual(
                download_pinned_source_files(load_pipeline_config(), root, (relative,)),
                (path,),
            )
            with self.assertRaises(ValueError):
                download_pinned_source_files(load_pipeline_config(), root, ("../escape",))


class PipelineConfigTest(unittest.TestCase):
    def test_default_contract(self) -> None:
        config = load_pipeline_config()
        self.assertEqual(config.source.episodes, 531)
        self.assertEqual(config.raw_source.episodes, 533)
        self.assertEqual(config.raw_source.repo_id, "BitRobot/2026-humanoid-ikea-assembly-challenge")
        self.assertEqual(config.target.data_shard_size_mb, 100)
        self.assertEqual(config.target.video_shard_size_mb, 500)
        self.assertEqual(config.dataset_runtime.lerobot_version, "0.6.0")
        self.assertEqual(
            config.runtime.dex1_gripper_python_sha256,
            "02edd7928913f76b7da74b97ce0e9ff3d9e5cd178e91cbb6599902d01782f269",
        )
        self.assertEqual(config.object_pose_runtime.source_frame_stride, 3)
        self.assertEqual(config.object_pose_runtime.segmentation_iou_threshold, 0.05)
        self.assertEqual(config.object_pose_runtime.dense_mask_min_bidirectional_iou, 0.5)
        self.assertEqual(config.object_pose_runtime.dense_mask_min_area_fraction, 0.005)
        self.assertEqual(config.object_pose_runtime.dense_mask_max_area_fraction, 0.75)
        self.assertEqual(config.object_pose_runtime.dense_temporal_propagation_beam_size, 128)
        self.assertEqual(config.object_pose_runtime.robot_silhouette_dilation_px, 48)
        self.assertEqual(
            config.object_pose_runtime.robot_visual_urdf_sha256,
            "b2ce7a8b620dc5511e9189ab19577788a65615eb508f54013b03f8156363ef3f",
        )
        self.assertEqual(config.source_annotation.automatic_phase.minimum_flip_angle_rad, 2.6)
        self.assertEqual(
            config.object_pose_runtime.foundationpose_revision,
            "a1b694b83e633c2cb6115b9063d940a687759392",
        )
        self.assertEqual(
            config.object_pose_runtime.detector_revision,
            "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        )
        self.assertEqual(
            config.object_pose_runtime.segmentation_checkpoint_sha256,
            "dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230",
        )
        self.assertEqual(len(artifact_specs(config.object_pose_runtime)), 8)
        self.assertEqual(config.object_pose_runtime.timm_version, "1.0.27")
        self.assertEqual(
            config.object_pose_runtime.fast_stereo_revision,
            "a290ba04c1b3ad1ec41a33974a157b2917b624d4",
        )
        self.assertEqual(len(config.dataset_runtime.lerobot_revision), 40)
        self.assertEqual([camera.source_key for camera in config.cameras], [
            "observation.images.cam_0",
            "observation.images.cam_2",
            "observation.images.cam_3",
        ])
        self.assertEqual(len(config.digest), 64)
        hand = config.raw["source_contract"]["dex1_hand_command"]
        self.assertEqual((hand["closed_motor_position"], hand["open_motor_position"]), (0.0, 4.5))
        self.assertEqual(
            (hand["organizer_normalized_open"], hand["organizer_normalized_closed"]),
            (-1.0, 1.0),
        )

    def test_pose_artifact_verifier_rejects_wrong_content(self) -> None:
        config = load_pipeline_config()
        spec = artifact_specs(config.object_pose_runtime)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"x" * spec.size_bytes)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                verify_artifact(path, spec)

    def test_cosmos_key_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["cosmos"] = {"enabled": False}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cosmos"):
                load_pipeline_config(path)

    def test_unknown_nested_config_key_is_rejected(self) -> None:
        payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["source_annotation"]["removed_threshold"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_annotation keys differ"):
                load_pipeline_config(path)

    def test_camera_calibration_hashes_must_be_an_array(self) -> None:
        payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        payload["cameras"][0]["intrinsic_calibration_sha256s"] = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an array"):
                load_pipeline_config(path)

    def test_source_info_contract(self) -> None:
        config = load_pipeline_config()
        features = {
            key: {"dtype": dtype, "shape": [width]}
            for key, (dtype, width) in NUMERIC_FEATURES.items()
        }
        for camera in config.cameras:
            features[camera.source_key] = {
                "dtype": "video",
                "shape": [480, 640, 3],
                "info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.fps": 30,
                    "video.channels": 3,
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        validate_source_info(
            {
                "codebase_version": "v3.0",
                "robot_type": "unitree_g1",
                "fps": 30,
                "total_episodes": 531,
                "total_frames": 290941,
                "data_files_size_in_mb": 100,
                "video_files_size_in_mb": 500,
                "features": features,
            },
            config,
        )


class AnnotationTest(unittest.TestCase):
    @staticmethod
    def _value() -> dict[str, object]:
        boundaries = {
            name: [index * 10, (index + 1) * 10]
            for index, name in enumerate(EXPECTED_SUBTASKS)
        }
        return {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "episode_index": 7,
            "frame_count": 70,
            "table_pose_trajectory_robot_root_xyzw": [
                [0.3, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0] for _ in range(70)
            ],
            "pose_evidence": {
                "method": "reviewed_pnp_cad_alignment",
                "reviewer": "pose-reviewer",
                "calibration_artifact_sha256": "a" * 64,
                "quality_metrics": [
                    {"name": "reprojection_rms", "value": 1.5, "unit": "px", "maximum": 2.0}
                ],
            },
            "subtask_reviewer": "phase-reviewer",
            "subtask_evidence_sha256": "b" * 64,
            "subtasks": {"left": dict(boundaries), "right": dict(boundaries)},
        }

    def test_valid_annotation(self) -> None:
        annotation = SourceEpisodeAnnotation.from_json(self._value())
        self.assertEqual(annotation.episode_index, 7)
        self.assertEqual(annotation.subtasks["left"]["rotate_180"].start, 30)

    def test_unsynchronized_bimanual_phase_is_rejected(self) -> None:
        value = self._value()
        value["subtasks"]["right"]["rotate_180"] = [31, 40]  # type: ignore[index]
        with self.assertRaises(ValueError):
            SourceEpisodeAnnotation.from_json(value)

    def test_guessed_pose_is_rejected(self) -> None:
        value = self._value()
        value["pose_evidence"]["method"] = "guess"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "measured"):
            SourceEpisodeAnnotation.from_json(value)

    def test_failed_pose_metric_is_rejected(self) -> None:
        value = self._value()
        value["pose_evidence"]["quality_metrics"][0]["value"] = 2.1  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "failed its acceptance gate"):
            SourceEpisodeAnnotation.from_json(value)


class CandidateLedgerTest(unittest.TestCase):
    def test_state_machine_is_resumable_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = CandidateLedger(directory)
            record = CandidateRecord(
                candidate_id="trajectory-000001",
                status="claimed",
                source_episode_indices=(),
                trajectory_seed=42,
                config_sha256="a" * 64,
                runtime_digest="b" * 64,
                payload={"claim_worker": "worker-0"},
            )
            ledger.claim(record)
            generated = ledger.transition(
                record.candidate_id,
                "generated",
                {"generator_success": True},
                source_episode_indices=(1, 5),
            )
            self.assertEqual(generated.status, "generated")
            self.assertEqual(generated.source_episode_indices, (1, 5))
            self.assertEqual(ledger.load(record.candidate_id), generated)
            with self.assertRaises(FileExistsError):
                ledger.claim(record)
            with self.assertRaisesRegex(ValueError, "immutable"):
                ledger.transition(record.candidate_id, "rejected", {"generator_success": False})

    def test_invalid_transition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = CandidateLedger(directory)
            record = CandidateRecord(
                candidate_id="trajectory-2",
                status="claimed",
                source_episode_indices=(),
                trajectory_seed=2,
                config_sha256="a" * 64,
                runtime_digest="b" * 64,
                payload={},
            )
            ledger.claim(record)
            with self.assertRaisesRegex(ValueError, "invalid transition"):
                ledger.transition(record.candidate_id, "rendered", {})

    def test_render_variants_are_independent_and_gate_candidate_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = CandidateLedger(directory)
            candidate = CandidateRecord(
                candidate_id="trajectory-3",
                status="claimed",
                source_episode_indices=(),
                trajectory_seed=3,
                config_sha256="a" * 64,
                runtime_digest="b" * 64,
                payload={},
            )
            ledger.claim(candidate)
            ledger.transition(
                candidate.candidate_id,
                "generated",
                {"generator_success": True},
                source_episode_indices=(9,),
            )
            ledger.transition(
                candidate.candidate_id,
                "validated",
                {"trajectory_sha256": "c" * 64},
            )
            for variant_index in range(2):
                variant = AppearanceRecord(
                    candidate_id=candidate.candidate_id,
                    variant_index=variant_index,
                    status="claimed",
                    appearance_seed=100 + variant_index,
                    trajectory_sha256="c" * 64,
                    config_sha256="a" * 64,
                    runtime_digest="b" * 64,
                    payload={},
                )
                ledger.claim_variant(variant)
                ledger.transition_variant(
                    candidate.candidate_id,
                    variant_index,
                    "rendered",
                    {"manifest_sha256": str(variant_index) * 64},
                )
            rendered = ledger.complete_rendering(candidate.candidate_id, minimum_variants=2)
            self.assertEqual(rendered.status, "rendered")
            self.assertEqual(rendered.payload["rendered_variant_indices"], [0, 1])


class StableTreeHashTest(unittest.TestCase):
    def test_ignores_bytecode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            first, count = stable_tree_sha256(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "module.pyc").write_bytes(b"unstable")
            second, second_count = stable_tree_sha256(root)
            self.assertEqual((first, count), (second, second_count))

    def test_explicit_output_directory_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            output = root / "outputs"
            output.mkdir()
            (output / "runtime.json").write_text("{}\n", encoding="utf-8")
            first = stable_tree_sha256(
                root,
                excluded_directory_names=frozenset({"outputs"}),
            )
            (output / "runtime.json").write_text('{"changed": true}\n', encoding="utf-8")
            second = stable_tree_sha256(
                root,
                excluded_directory_names=frozenset({"outputs"}),
            )
            self.assertEqual(first, second)


class RawSourceBindingTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path, *, include_unlabelled: bool = False):
        import pyarrow as pa
        import pyarrow.parquet as pq

        config = load_pipeline_config()
        raw = root / "raw"
        source = root / "source"
        raw_episode = raw / "episode_2026-05-05_09-50-59" / "episode_0001"
        calibration = raw_episode / "calibration" / "params"
        calibration.mkdir(parents=True)
        info = {
            "episode_name": "episode_2026-05-05_09-50-59",
            "start_timestamp_ns": 1_000_000_000,
            "end_timestamp_ns": 6_000_000_000,
            "subtasks": [{"task": "flip table", "timestamp_ns": 2_000_000_000}],
        }
        (raw_episode / "info.json").write_text(json.dumps(info), encoding="utf-8")
        head = {
            "success": True,
            "image_size": [640, 480],
            "baseline": 1000.0 * config.raw_source.head_stereo_baseline_m,
            "rms_error": config.raw_source.head_stereo_rms_error_px,
        }
        head_path = calibration / "head_camera_params.yaml"
        import yaml

        head_path.write_text(yaml.safe_dump(head), encoding="utf-8")
        wrist_hashes = {}
        for serial in ("111", "222"):
            wrist = {
                "serial_number": serial,
                "color": {
                    "intrinsics": {
                        "width": 640,
                        "height": 480,
                        "fx": 430.0,
                        "fy": 431.0,
                        "ppx": 320.0,
                        "ppy": 240.0,
                        "model": "distortion.inverse_brown_conrady",
                        "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                    }
                },
            }
            path = calibration / f"camera_{serial}.json"
            path.write_text(json.dumps(wrist), encoding="utf-8")
            wrist_hashes[serial] = hashlib.sha256(path.read_bytes()).hexdigest()
        if include_unlabelled:
            omitted = raw / "episode_2026-05-05_10-00-00" / "episode_0001"
            omitted_calibration = omitted / "calibration" / "params"
            omitted_calibration.mkdir(parents=True)
            omitted_info = dict(info)
            omitted_info["episode_name"] = "episode_2026-05-05_10-00-00"
            omitted_info["subtasks"] = []
            (omitted / "info.json").write_text(json.dumps(omitted_info), encoding="utf-8")
            (omitted_calibration / "head_camera_params.yaml").write_bytes(head_path.read_bytes())
            for serial in ("111", "222"):
                source_path = calibration / f"camera_{serial}.json"
                (omitted_calibration / source_path.name).write_bytes(source_path.read_bytes())

        episode_dir = source / "meta" / "episodes" / "chunk-000"
        episode_dir.mkdir(parents=True)
        source_row = {
            "episode_index": 0,
            "source_episode_index": 0,
            "source_episode_name": info["episode_name"],
            "source_start_sec": 1.0,
            "source_end_sec": 5.0,
        }
        pq.write_table(pa.Table.from_pylist([source_row]), episode_dir / "file-000.parquet")
        return raw, source, source_row, hashlib.sha256(head_path.read_bytes()).hexdigest(), wrist_hashes

    def _config(self, head_hash: str, wrist_hashes: dict[str, str], *, raw_episodes: int = 1):
        config = load_pipeline_config()
        return replace(
            config,
            source=replace(config.source, episodes=1),
            raw_source=replace(
                config.raw_source,
                episodes=raw_episodes,
                head_stereo_calibration_sha256=head_hash,
                wrist_calibration_sha256_by_serial=wrist_hashes,
            ),
        )

    def test_exact_raw_binding_and_calibration_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw, source, _source_row, head_hash, wrist_hashes = self._write_fixture(
                Path(directory)
            )
            report = audit_raw_source_bindings(
                raw,
                source,
                self._config(head_hash, wrist_hashes),
            )
            provenance = report["bindings"][0]
            self.assertEqual(provenance["flip_interval_sec"], [1.0, 5.0])
            self.assertEqual(
                {item["serial_number"] for item in provenance["wrist_d405_calibrations"]},
                {"111", "222"},
            )

    def test_full_audit_accounts_for_unlabelled_raw_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw, source, _source_row, head_hash, wrist_hashes = self._write_fixture(
                Path(directory), include_unlabelled=True
            )
            report = audit_raw_source_bindings(
                raw,
                source,
                self._config(head_hash, wrist_hashes, raw_episodes=2),
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["counts"]["source_flip_episodes"], 1)
            self.assertEqual(report["omitted_raw_episodes"][0]["raw_episode_index"], 1)

    def test_time_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw, source, _source_row, head_hash, wrist_hashes = self._write_fixture(
                Path(directory)
            )
            import pyarrow as pa
            import pyarrow.parquet as pq

            path = source / "meta/episodes/chunk-000/file-000.parquet"
            row = pq.read_table(path).to_pylist()[0]
            row["source_start_sec"] = 1.1
            pq.write_table(pa.Table.from_pylist([row]), path)
            with self.assertRaisesRegex(ValueError, "start time"):
                audit_raw_source_bindings(
                    raw,
                    source,
                    self._config(head_hash, wrist_hashes),
                )

class FkSamplingTest(unittest.TestCase):
    def test_samples_include_episode_endpoints(self) -> None:
        self.assertEqual(select_frame_indices(10, 4), (0, 3, 6, 9))
        self.assertEqual(select_frame_indices(2, 8), (0, 1))

    def test_single_sample_uses_middle_frame(self) -> None:
        self.assertEqual(select_frame_indices(9, 1), (4,))

    def test_synthetic_action_fk_gate_rejects_label_drift(self) -> None:
        class Placement:
            def __init__(self, translation):
                self.translation = np.asarray(translation, dtype=np.float64)
                self.rotation = np.eye(3, dtype=np.float64)

        class Data:
            oMf = {1: Placement((0.3, 0.2, 0.7)), 2: Placement((0.3, -0.2, 0.7))}

        class Model:
            nq = 29

        class Pin:
            @staticmethod
            def framesForwardKinematics(_model, _data, _q):
                return None

        joints = np.zeros((5, 36), dtype=np.float64)
        targets = np.tile(
            np.asarray(
                [0.35, 0.2, 0.7, 0.0, 0.0, 0.0, 0.35, -0.2, 0.7, 0.0, 0.0, 0.0]
            ),
            (5, 1),
        )
        contract = {
            "robot_q_desired": joints,
            "ee_action": targets,
            "frame_names": {"left": "left_wrist", "right": "right_wrist"},
            "tool_transforms": {
                side: {
                    "translation_m": [0.05, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
                for side in ("left", "right")
            },
            "position_p95_max": 0.08,
            "rotation_p95_max": 0.2,
        }
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "robot.urdf"
            urdf.write_text("fixture", encoding="utf-8")
            with patch(
                "data.flip_table_data_augmentation.fk_audit._build_fk",
                return_value=(Pin(), Model(), Data(), np.arange(29), {"left": 1, "right": 2}),
            ):
                accepted = synthetic_action_fk_report(urdf_path=urdf, **contract)
                shifted = targets.copy()
                shifted[:, 0] += 0.2
                rejected = synthetic_action_fk_report(
                    urdf_path=urdf,
                    **{**contract, "ee_action": shifted},
                )
        self.assertTrue(accepted["pass"])
        self.assertFalse(rejected["pass"])
        self.assertAlmostEqual(rejected["sides"]["left"]["position_error_m"]["p95"], 0.2)

    def test_fk_audit_builds_kinematics_without_optional_mesh_assets(self) -> None:
        source = (Path(__file__).parents[1] / "fk_audit.py").read_text(encoding="utf-8")
        self.assertIn("pin.buildModelFromUrdf", source)
        self.assertIn("model.createData()", source)
        self.assertNotIn("RobotWrapper.BuildFromURDF", source)


if __name__ == "__main__":
    unittest.main()
