from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from data.flip_table_data_augmentation.annotations import ANNOTATION_SCHEMA_VERSION
from data.flip_table_data_augmentation.config import EXPECTED_SUBTASKS, load_pipeline_config
from data.flip_table_data_augmentation.fk_audit import FK_AUDIT_SCHEMA_VERSION
from data.flip_table_data_augmentation.mimic.source_hdf5 import export_mimic_source_hdf5


HAS_EXPORT_STACK = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "scipy", "pyarrow", "h5py")
)


@unittest.skipUnless(HAS_EXPORT_STACK, "requires NumPy, SciPy, PyArrow, and h5py")
class MimicSourceHdf5Test(unittest.TestCase):
    def test_exports_pink_tool_and_dex1_contract(self) -> None:
        import h5py
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame_count = len(EXPECTED_SUBTASKS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
            (source / "data" / "chunk-000").mkdir(parents=True)
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0],
                        "length": [frame_count],
                        "data/chunk_index": [0],
                        "data/file_index": [0],
                    }
                ),
                source / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
            )
            eef = [[0.30, 0.10, 0.80, 0.0, 0.0, 0.0, 0.30, -0.10, 0.80, 0.0, 0.0, 0.0]] * frame_count
            hand = [[4.5 * index / (frame_count - 1)] * 2 for index in range(frame_count)]
            pq.write_table(
                pa.table(
                    {
                        "episode_index": [0] * frame_count,
                        "frame_index": list(range(frame_count)),
                        "observation.state.ee_state": eef,
                        "action.ee_action": eef,
                        "action.hand_cmd": hand,
                    }
                ),
                source / "data" / "chunk-000" / "file-000.parquet",
            )
            boundaries = {
                name: [index, index + 1] for index, name in enumerate(EXPECTED_SUBTASKS)
            }
            annotations = root / "annotations.json"
            annotations.write_text(
                json.dumps(
                    {
                        "schema_version": ANNOTATION_SCHEMA_VERSION,
                        "episodes": [
                            {
                                "episode_index": 0,
                                "frame_count": frame_count,
                                "table_pose_trajectory_robot_root_xyzw": [
                                    [0.3, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]
                                    for _ in range(frame_count)
                                ],
                                "pose_evidence": {
                                    "method": "measured_test_calibration",
                                    "reviewer": "test-reviewer",
                                    "calibration_artifact_sha256": "a" * 64,
                                    "quality_metrics": [
                                        {"name": "test", "value": 0.0, "unit": "m", "maximum": 0.01}
                                    ],
                                },
                                "subtask_reviewer": "test-reviewer",
                                "subtask_evidence_sha256": "b" * 64,
                                "subtasks": {"left": dict(boundaries), "right": dict(boundaries)},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "mimic_source.hdf5"
            config = load_pipeline_config()
            fk_audit = root / "fk_audit.json"
            fk_audit.write_text(
                json.dumps(
                    {
                        "schema_version": FK_AUDIT_SCHEMA_VERSION,
                        "source_repo_id": config.source.repo_id,
                        "source_revision": config.source.revision,
                        "config_sha256": config.digest,
                        "pass": True,
                        "frame_assignment_pass": True,
                        "action_fk_residual_pass": True,
                        "mimic_source_episode_gate": {
                            "eligible_episode_indices": [0],
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = export_mimic_source_hdf5(
                source_root=source,
                annotations_path=annotations,
                fk_audit_path=fk_audit,
                output_path=output,
                config=config,
            )
            self.assertEqual((report["episodes"], report["source_frames"]), (1, frame_count))
            self.assertEqual(report["control_steps"], 12)
            with h5py.File(output, "r") as stream:
                demo = stream["data/demo_0"]
                actions = demo["actions"][:]
                self.assertEqual(actions.shape, (12, 16))
                self.assertAlmostEqual(float(actions[0, 14]), 1.0)
                self.assertAlmostEqual(float(actions[-1, 15]), -1.0)
                self.assertAlmostEqual(float(actions[0, 0]), 0.25, places=6)
                self.assertLess(float(demo.attrs["pink_tool_inverse_max_abs_error"]), 1.0e-5)
                signals = demo["obs/datagen_info/subtask_term_signals"]
                self.assertEqual(len(signals), 12)


if __name__ == "__main__":
    unittest.main()
