from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from data.flip_table_data_augmentation.scripts.extract_oci_rootfs import extract
from data.flip_table_data_augmentation.scripts.patch_v1_recorder_api import (
    CALL_AFTER,
    MARKER,
    patch_recorder_api,
)


FEATURE_ROOT = Path(__file__).resolve().parents[1]


def _layer(path: Path, entries: dict[str, bytes]) -> str:
    with tarfile.open(path, "w") as archive:
        for name, value in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    renamed = path.with_name(digest)
    path.replace(renamed)
    return f"sha256:{digest}"


class OciRootfsExtractionTest(unittest.TestCase):
    def test_applies_deletion_and_opaque_whiteouts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = root / "layout"
            layout.mkdir()
            first = _layer(
                layout / "first.tar",
                {
                    "workspace/robofinals/old.txt": b"old\n",
                    "workspace/remove.txt": b"remove\n",
                },
            )
            second = _layer(
                layout / "second.tar",
                {
                    "workspace/robofinals/.wh..wh..opq": b"",
                    "workspace/robofinals/new.txt": b"new\n",
                    "workspace/.wh.remove.txt": b"",
                    "workspace/robofinalsbak/robofinals/marker.txt": b"backup\n",
                },
            )
            manifest = {
                "schemaVersion": 2,
                "config": {"digest": "sha256:" + "0" * 64},
                "layers": [{"digest": first}, {"digest": second}],
            }
            manifest_path = layout / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_digest = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"
            output = root / "rootfs"

            marker = extract(
                layout=layout,
                output=output,
                expected_digest=manifest_digest,
            )

            self.assertFalse((output / "workspace/robofinals/old.txt").exists())
            self.assertEqual(
                (output / "workspace/robofinals/new.txt").read_text(encoding="utf-8"),
                "new\n",
            )
            self.assertFalse((output / "workspace/remove.txt").exists())
            self.assertEqual(marker["manifest_digest"], manifest_digest)
            self.assertEqual(
                extract(layout=layout, output=output, expected_digest=manifest_digest),
                marker,
            )

    def test_refuses_unverified_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout = root / "layout"
            layout.mkdir()
            digest = _layer(layout / "layer.tar", {"workspace/robofinals/a": b"a"})
            manifest = {
                "schemaVersion": 2,
                "config": {"digest": "sha256:" + "0" * 64},
                "layers": [{"digest": digest}],
            }
            manifest_path = layout / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_digest = f"sha256:{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}"
            output = root / "rootfs"
            output.mkdir()

            with self.assertRaises(FileExistsError):
                extract(
                    layout=layout,
                    output=output,
                    expected_digest=manifest_digest,
                )


class ObjectPoseRuntimeScriptTest(unittest.TestCase):
    def test_handeye_consensus_uses_the_pinned_runtime(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_object_pose_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("handeye-consensus", script)
        self.assertIn("wrist_handeye_consensus.py", script)

    def test_head_cad_alignment_uses_the_pinned_runtime(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_object_pose_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("head-cad", script)
        self.assertIn("source_cad_alignment.py", script)

    def test_state_timing_audit_uses_the_pinned_runtime(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_object_pose_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("state-timing", script)
        self.assertIn("state_timing_audit.py", script)

    def test_prepare_keeps_hub_online_while_inference_is_offline(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_object_pose_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('if [[ "$COMMAND" != "prepare" ]]; then', script)
        self.assertIn(
            "offline_arguments+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)",
            script,
        )
        self.assertIn(
            'if [[ "$COMMAND" == "prepare" ]]; then\n'
            "  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE",
            script,
        )

    def test_diagnostic_trace_variables_are_forwarded_to_docker(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_object_pose_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("diagnostic_arguments=()", script)
        self.assertIn("FLIP_TABLE_TRACK_TRACE", script)
        self.assertIn("FLIP_TABLE_TEMPORAL_SELECTION_TRACE", script)
        self.assertIn('"${diagnostic_arguments[@]}"', script)

    def test_v1_entrypoints_share_mesh_and_recorder_contracts(self) -> None:
        direct = (FEATURE_ROOT / "scripts/run_v1_direct.sh").read_text(encoding="utf-8")
        container = (FEATURE_ROOT / "scripts/run_v1_container.sh").read_text(
            encoding="utf-8"
        )
        for script in (direct, container):
            self.assertIn("export-table-mesh", script)
            self.assertIn("patch_v1_recorder_api.py", script)


class RecorderApiPatchTest(unittest.TestCase):
    def test_forwards_demo_ids_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monkey_patch.py"
            path.write_text(
                "    def export_episodes(self, env_ids=None) -> None:\n"
                "        orig_export_episodes(self, env_ids)\n",
                encoding="utf-8",
            )

            self.assertTrue(patch_recorder_api(path))
            self.assertFalse(patch_recorder_api(path))
            patched = path.read_text(encoding="utf-8")
            self.assertIn(MARKER, patched)
            self.assertIn(CALL_AFTER, patched)


class MimicQuaternionContractTest(unittest.TestCase):
    def test_adapter_converts_only_at_the_controller_boundary(self) -> None:
        adapter = (FEATURE_ROOT / "mimic/env.py").read_text(encoding="utf-8")

        self.assertIn("wrist_quaternion_xyzw[[3, 0, 1, 2]]", adapter)
        self.assertIn("wrist_quaternion_wxyz[:, (1, 2, 3, 0)]", adapter)
        self.assertIn("new_tensor((0.0, 0.0, 0.0, 1.0))", adapter)
        self.assertNotIn("table_quaternion_wxyz", adapter)
        self.assertIn("table_quaternion_xyzw,", adapter)


class GenerationAndReplayRuntimeContractTest(unittest.TestCase):
    def test_generation_uses_official_v1_g1_root_height(self) -> None:
        script = (FEATURE_ROOT / "scripts/run_mimic_generation.py").read_text(
            encoding="utf-8"
        )
        env_cfg = (FEATURE_ROOT / "mimic/env_cfg.py").read_text(encoding="utf-8")
        recorders = (FEATURE_ROOT / "mimic/recorders.py").read_text(encoding="utf-8")

        self.assertIn("OFFICIAL_V1_G1_ROOT_HEIGHT_M = 0.78", script)
        self.assertIn(
            '"FLIP_TABLE_ROBOT_BASE_HEIGHT_M": OFFICIAL_V1_G1_ROOT_HEIGHT_M',
            script,
        )
        self.assertIn("rl_name=None", script)
        self.assertIn("G1PinkActionsCfg", env_cfg)
        self.assertIn("actions.base_action = None", env_cfg)
        self.assertIn("env_cfg.actions = _pink_dex1_actions()", env_cfg)
        self.assertNotIn("_wbc_dex1_actions", env_cfg)
        self.assertIn('get_term("arms_action")', recorders)
        self.assertNotIn('get_term("base_action")', recorders)

    def test_replay_uses_eval_cameras_without_residual_rl(self) -> None:
        script = (FEATURE_ROOT / "scripts/render_accepted_trajectories.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("rl_name=None", script)
        self.assertIn("execute_mode=ExecuteMode.EVAL", script)
        self.assertIn("active_observation_camera_names", script)
        self.assertNotIn("execute_mode=ExecuteMode.REPLAY_STATE", script)


if __name__ == "__main__":
    unittest.main()
