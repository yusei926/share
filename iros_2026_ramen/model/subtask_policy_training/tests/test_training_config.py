from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_default_configs_are_independent_of_working_directory() -> None:
    expected = (ROOT / "configs" / "subtask_training.json").resolve()
    script_names = (
        "download_source_mcaps.py",
        "materialize_lerobot_training_view.py",
        "resolve_training_config.py",
        "upload_lerobot_dataset.py",
        "write_groot_modality_json.py",
    )
    for index, script_name in enumerate(script_names):
        module = load_module(
            f"default_config_path_{index}",
            ROOT / "scripts" / script_name,
        )
        assert module.DEFAULT_CONFIG == expected
        assert module.DEFAULT_CONFIG.is_file()


def test_vast_flow_matching_config_pins_a_dataset_revision_not_a_cache_path() -> None:
    config = (ROOT / "deployment" / "vast" / "flow_matching_bc.conf").read_text(
        encoding="utf-8"
    )
    assert 'DATASET_REVISION="10a6ec05f9993b8d59faad2957e47153b0f15f37"' in config
    assert "SOURCE_DATASET_ROOT=" not in config
    assert ".cache/huggingface" not in config


def test_act_diagnostic_uses_strict_processor_contract(tmp_path: Path) -> None:
    module = load_module(
        "diagnose_act_processor_contract",
        ROOT / "scripts" / "diagnose_act_policy_outputs.py",
    )
    state_path = tmp_path / "stats.safetensors"
    state_path.write_bytes(b"state")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": state_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved, eps = module.resolve_processor_state(
        tmp_path,
        "policy_preprocessor.json",
        "normalizer_processor",
    )
    assert resolved == state_path.resolve()
    assert eps == pytest.approx(1e-8)
    with pytest.raises(KeyError, match="statistics are missing"):
        module.normalize_feature({}, "observation.state", module.torch.zeros(19), eps)

    source = (ROOT / "scripts" / "diagnose_act_policy_outputs.py").read_text(encoding="utf-8")
    assert 'load_kwargs: dict[str, Any] = {"strict": True}' in source
    assert "/home/suzuki" not in source


def test_policy_uploader_validates_processor_state_files(tmp_path: Path) -> None:
    module = load_module(
        "upload_policy_processor_contract",
        ROOT / "scripts" / "upload_policy.py",
    )
    for name in (
        "config.json",
        "evaluation_report.json",
        "train_config.json",
        "training_manifest.json",
        "training_run_record.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"eps": 1e-8},
                        "state_file": "normalizer.safetensors",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "policy_postprocessor.json").write_text(
        json.dumps({"steps": [{"registry_name": "device_processor", "config": {}}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="normalizer.safetensors"):
        module.validate_model_dir(tmp_path)
    (tmp_path / "normalizer.safetensors").write_bytes(b"state")
    module.validate_model_dir(tmp_path)

    processor = json.loads((tmp_path / "policy_preprocessor.json").read_text(encoding="utf-8"))
    processor["steps"][0]["state_file"] = "../outside.safetensors"
    (tmp_path / "policy_preprocessor.json").write_text(json.dumps(processor), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes model directory"):
        module.validate_model_dir(tmp_path)


def test_policy_uploader_requires_finalized_furniture_groot_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module(
        "upload_policy_furniture_groot_contract",
        ROOT / "scripts" / "upload_policy.py",
    )
    for name in module.REQUIRED_POLICY_FILES:
        if name.endswith(".json"):
            payload = {"type": "furniture_groot"} if name == "config.json" else {}
            (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
        else:
            (tmp_path / name).write_bytes(b"model")
    for processor_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (tmp_path / processor_name).write_text(
            json.dumps({"steps": []}),
            encoding="utf-8",
        )
    for name in (
        "candidate_selection.json",
        "dex1_g1_synergy.json",
        "groot_contract.json",
        "orientation_contact_sheet.approved",
        "orientation_contact_sheet.jpg",
        "progress_manifest.json",
        "progress.jsonl",
        "sim_candidate_selection.json",
        "sim_evaluation_manifest.json",
        "sim_release_evaluation.json",
        "source_snapshot_manifest.json",
        "visual_rotation_manifest.json",
        "visual_rotation.jsonl",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="eef_fk_audit.json"):
        module.validate_model_dir(tmp_path)

    (tmp_path / "eef_fk_audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="simulator evaluation bundle"):
        module.validate_model_dir(tmp_path)
    (tmp_path / "sim_evaluation").mkdir()
    validated: list[Path] = []
    monkeypatch.setattr(
        module,
        "validate_finalized_furniture_checkpoint",
        lambda path: validated.append(path),
    )
    module.validate_model_dir(tmp_path)
    assert validated == [tmp_path]


def test_resolve_default_training_config(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    for name in [
        "SUBTASK",
        "POLICY_TYPE",
        "CONTROL_SCOPE",
        "DATASET_REPO_ID",
        "DATASET_REVISION",
        "GROOT_DATASET_REPO_ID",
        "POLICY_REPO_ID",
        "OUTPUT_DIR",
        "JOB_NAME",
        "DEVICE",
        "WANDB_ENABLE",
        "WANDB_PROJECT",
        "PUSH_TO_HUB",
        "UPLOAD_AFTER_TRAIN",
        "PRIVATE",
        "MATERIALIZE_TRAINING_VIEW",
        "TRAINING_VIEW_ROOT",
        "TRAINING_VIEW_FORCE",
        "POLICY_INPUT_FEATURES",
        "POLICY_OUTPUT_FEATURES",
        "GROOT_BASE_MODEL_PATH",
        "GROOT_BASE_MODEL_REVISION",
        "GROOT_EMBODIMENT_TAG",
        "GROOT_CHUNK_SIZE",
        "GROOT_N_ACTION_STEPS",
        "GROOT_USE_RELATIVE_ACTIONS",
        "GROOT_RELATIVE_EXCLUDE_JOINTS",
        "GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR",
        "GROOT_PROCESSOR_OVERLAY_ROOT",
        "GROOT_USE_BF16",
    ]:
        monkeypatch.delenv(name, raising=False)

    values = module.resolve(config)

    assert values["SUBTASK"] == "pick_leg"
    assert values["POLICY_TYPE"] == "act"
    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
    assert values["DATASET_REVISION"] == ""
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1"
    assert values["POLICY_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_act_1"
    assert values["OUTPUT_DIR"] == "outputs/train/act_pick_leg"
    assert values["WANDB_PROJECT"] == "iros2026-ramen-pick-leg"
    assert values["PUSH_TO_HUB"] == "false"
    assert values["UPLOAD_AFTER_TRAIN"] == "true"
    assert values["CONTROL_SCOPE"] == "upper_body"
    assert values["STATE_DIM"] == "19"
    assert values["ACTION_DIM"] == "16"
    assert values["CAMERAS"] == "head_left,left_wrist,right_wrist"
    assert values["MATERIALIZE_TRAINING_VIEW"] == "true"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/act_pick_leg"
    assert values["TRAINING_VIEW_FORCE"] == "false"
    assert values["TRAIN_IMAGE_TRANSFORMS_ENABLE"] == "true"
    assert values["TRAIN_BATCH_SIZE"] == "8"
    assert values["TRAIN_STEPS"] == "300000"
    assert values["TRAIN_EVAL_STEPS"] == "10000"
    assert values["TRAIN_MAX_EVAL_SAMPLES"] == "512"
    assert values["ACT_CHUNK_SIZE"] == "100"
    assert values["ACT_N_ACTION_STEPS"] == "10"
    assert json.loads(values["SOURCE_CAMERA_MAP"]) == {
        "observation.images.head_left": "observation.images.cam_0",
        "observation.images.left_wrist": "observation.images.cam_2",
        "observation.images.right_wrist": "observation.images.cam_3",
    }
    assert values["POLICY_VIEW_LAYOUT"] == "robot_q_state19_action16"
    input_features = json.loads(values["POLICY_INPUT_FEATURES"])
    output_features = json.loads(values["POLICY_OUTPUT_FEATURES"])
    assert list(input_features) == [
        "observation.state",
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert "observation.images.head_right" not in input_features
    assert input_features["observation.images.head_left"] == {
        "type": "VISUAL",
        "shape": [3, 480, 640],
    }
    assert input_features["observation.state"] == {"type": "STATE", "shape": [19]}
    assert output_features == {"action": {"type": "ACTION", "shape": [16]}}
    assert "GROOT_BASE_MODEL_PATH" not in values


def test_resolve_allows_subtask_override(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "tighten")
    monkeypatch.setenv("POLICY_TYPE", "diffusion")

    values = module.resolve(config)

    assert values["TASK"] == "rotate leg to tighten"
    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_1"
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_1"
    assert values["POLICY_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_tighten_diffusion_1"
    assert values["WANDB_PROJECT"] == "iros2026-ramen-tighten"


def test_resolve_flow_matching_defaults(monkeypatch) -> None:
    module = load_module("resolve_flow_matching", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("POLICY_TYPE", "flow_matching")

    values = module.resolve(config)

    assert values["CONTROL_SCOPE"] == "upper_body"
    assert values["STATE_DIM"] == "19"
    assert values["ACTION_DIM"] == "16"
    assert values["FLOW_ACTION_HORIZON"] == "24"
    assert values["FLOW_N_ACTION_STEPS"] == "6"
    assert values["FLOW_INFERENCE_STEPS"] == "10"
    assert values["FLOW_MODEL_DIM"] == "384"
    assert values["FLOW_TRANSFORMER_LAYERS"] == "6"
    assert values["FLOW_TRANSFORMER_HEADS"] == "8"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/flow_matching_flip_table"
    assert values["POLICY_REPO_ID"].endswith("_flip_table_flow_matching_1")

    monkeypatch.setenv("FLOW_ACTION_HORIZON", "4")
    monkeypatch.setenv("FLOW_N_ACTION_STEPS", "5")
    with pytest.raises(ValueError, match="cannot exceed"):
        module.resolve(config)


def test_resolve_allows_pinned_dataset_revision(monkeypatch) -> None:
    module = load_module("resolve_training_config_revision", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("DATASET_REVISION", "10a6ec05f9993b8d59faad2957e47153b0f15f37")

    values = module.resolve(config)

    assert values["DATASET_REVISION"] == "10a6ec05f9993b8d59faad2957e47153b0f15f37"


def test_resolve_rejects_incomplete_camera_and_temporal_contracts(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    config["cameras"] = ["head_left", "left_wrist"]
    with pytest.raises(ValueError, match="exactly"):
        module.resolve(config)

    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("ACT_CHUNK_SIZE", "4")
    monkeypatch.setenv("ACT_N_ACTION_STEPS", "5")
    with pytest.raises(ValueError, match="cannot exceed"):
        module.resolve(config)


def test_resolve_groot_n17_defaults(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("POLICY_TYPE", "groot")

    values = module.resolve(config)

    assert values["DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
    assert values["DATASET_REVISION"] == "0dc47877dfb2efbea796a059c81290c649bc773c"
    assert values["GROOT_DATASET_REPO_ID"] == "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
    assert values["POLICY_REPO_ID"] == (
        "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2"
    )
    assert values["WANDB_PROJECT"] == "iros2026-ramen-flip-table"
    assert values["CONTROL_SCOPE"] == "upper_body_relative_eef"
    assert values["STATE_DIM"] == "49"
    assert values["ACTION_DIM"] == "53"
    assert values["ACTION_SEMANTICS"] == "real_g1_relative_eef_relative_arm_absolute_hand_waist"
    assert values["POLICY_VIEW_LAYOUT"] == "real_g1_relative_eef_relative_joints"
    assert values["TRAINING_VIEW_ROOT"] == "outputs/training_views/groot_flip_table"
    assert json.loads(values["POLICY_INPUT_FEATURES"])["observation.state"] == {
        "type": "STATE",
        "shape": [49],
    }
    assert json.loads(values["POLICY_OUTPUT_FEATURES"]) == {
        "action": {"type": "ACTION", "shape": [53]}
    }
    assert values["GROOT_BASE_MODEL_PATH"] == "nvidia/GR00T-N1.7-3B"
    assert values["GROOT_BASE_MODEL_REVISION"] == "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    assert values["GROOT_EMBODIMENT_TAG"] == "real_g1_relative_eef_relative_joints"
    assert values["GROOT_POLICY_IMPL"] == "furniture_groot"
    assert values["GROOT_CHUNK_SIZE"] == "40"
    assert values["GROOT_N_ACTION_STEPS"] == "10"
    assert values["GROOT_VALID_ACTION_DIM"] == "46"
    assert values["GROOT_TUNE_TOP_LLM_LAYERS"] == "0"
    assert values["GROOT_PROGRESS_ENABLED"] == "true"
    assert values["GROOT_PROGRESS_LOSS_WEIGHT"] == "0.050000000000000003"
    assert values["GROOT_PROGRESS_MONOTONICITY_WEIGHT"] == "0.01"
    assert values["GROOT_PROGRESS_HIDDEN_DIM"] == "512"
    assert values["GROOT_PROGRESS_SIDECAR"] == (
        "outputs/progress_sidecars/flip_table_v2/progress.jsonl"
    )
    assert values["GROOT_USE_RELATIVE_ACTIONS"] == "true"
    assert values["GROOT_RELATIVE_EXCLUDE_JOINTS"] == '["hand","waist","base_height","navigate"]'
    assert values["GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR"] == "true"
    assert values["GROOT_PROCESSOR_OVERLAY_ROOT"] == (
        "outputs/groot_base_overlays/real_g1_relative_eef_3cam"
    )
    assert values["GROOT_USE_BF16"] == "true"
    assert values["GROOT_IMAGE_TRANSFORMS_ENABLE"] == "false"
    assert values["GROOT_CONSISTENT_GPU_AUGMENTATION"] == "true"
    assert values["GROOT_BATCH_SIZE"] == "64"
    assert values["GROOT_STEPS"] == "20000"
    assert values["GROOT_WARMUP_STEPS"] == "1000"


def test_resolve_groot_warmup_steps_follow_training_length(monkeypatch) -> None:
    module = load_module(
        "resolve_training_config_warmup",
        ROOT / "scripts" / "resolve_training_config.py",
    )
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_STEPS", "1000")
    monkeypatch.setenv("GROOT_WARMUP_RATIO", "0.05")

    values = module.resolve(config)

    assert values["GROOT_WARMUP_STEPS"] == "50"
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text(encoding="utf-8")
    assert '--policy.max_steps="$GROOT_STEPS"' in script
    assert "--scheduler.num_warmup_steps" not in script


def test_resolve_groot_n17_allows_temporal_overrides_but_not_slot_semantics(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_CHUNK_SIZE", "40")
    monkeypatch.setenv("GROOT_N_ACTION_STEPS", "25")

    values = module.resolve(config)

    assert values["GROOT_CHUNK_SIZE"] == "40"
    assert values["GROOT_N_ACTION_STEPS"] == "25"
    assert values["GROOT_USE_RELATIVE_ACTIONS"] == "true"
    assert values["GROOT_RELATIVE_EXCLUDE_JOINTS"] == '["hand","waist","base_height","navigate"]'

    monkeypatch.setenv("GROOT_RELATIVE_EXCLUDE_JOINTS", '["left_gripper","right_gripper"]')
    with pytest.raises(ValueError, match="must be exactly"):
        module.resolve(config)


def test_resolve_groot_n17_forbids_unfreezing_top_llm_layers(monkeypatch) -> None:
    module = load_module(
        "resolve_training_config_tuning_scope",
        ROOT / "scripts" / "resolve_training_config.py",
    )
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_TUNE_TOP_LLM_LAYERS", "1")

    with pytest.raises(ValueError, match="tuning scope"):
        module.resolve(config)


def test_resolve_rejects_unknown_policy_type(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")
    monkeypatch.setenv("POLICY_TYPE", "unknown")

    with pytest.raises(ValueError, match="POLICY_TYPE must be one of"):
        module.resolve(config)


def test_resolve_groot_rejects_disabling_relative_eef(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_USE_RELATIVE_ACTIONS", "false")

    try:
        module.resolve(config)
    except ValueError as exc:
        assert "chunk's current observation" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("disabling native relative EEF must fail")


def test_resolve_groot_rejects_disabling_native_eef_processor(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("POLICY_TYPE", "groot")
    monkeypatch.setenv("GROOT_REQUIRE_NATIVE_RELATIVE_EEF_PROCESSOR", "false")

    with pytest.raises(ValueError, match="must stay true"):
        module.resolve(config)


def test_resolve_allows_wandb_project_override(monkeypatch) -> None:
    module = load_module("resolve_training_config", ROOT / "scripts" / "resolve_training_config.py")
    config = module.load_config(ROOT / "configs" / "subtask_training.json")

    monkeypatch.setenv("SUBTASK", "flip_table")
    monkeypatch.setenv("WANDB_PROJECT", "custom-project")

    values = module.resolve(config)

    assert values["WANDB_PROJECT"] == "custom-project"


def test_train_wrapper_passes_policy_feature_overrides() -> None:
    script = (ROOT / "scripts" / "train_lerobot.sh").read_text()

    assert '--policy.input_features="$POLICY_INPUT_FEATURES"' in script
    assert '--policy.output_features="$POLICY_OUTPUT_FEATURES"' in script
    assert "materialize_lerobot_training_view.py" in script
    assert "prepare_groot_n17_real_g1_overlay.py" in script
    assert "patch_lerobot_groot_relative_eef.py --check" in script
    assert "restore_groot_base_model_path.py" in script
    assert '--policy.tune_top_llm_layers="$GROOT_TUNE_TOP_LLM_LAYERS"' in script
    assert '--dataset.root="$TRAINING_VIEW_ROOT"' in script


def test_lerobot_training_view_materializer_maps_official_schema() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    table = pa.table(
        {
            "observation.state.ee_state": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0] * 2],
            "observation.state.robot_q_current": [[float(i) for i in range(36)]],
            "observation.state.hand_state": [[36.0, 37.0]],
            "action.ee_action": [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0] * 2],
            "action.robot_q_desired": [[float(i) for i in range(100, 136)]],
            "action.hand_cmd": [[136.0, 137.0]],
            "timestamp": [0.0],
            "frame_index": [0],
            "episode_index": [0],
            "index": [0],
            "task_index": [0],
        }
    )

    state_rows, action_rows = module.build_policy_vectors_from_table(
        table,
        config=config,
        policy_type="act",
    )
    assert state_rows == [[float(i) for i in range(19, 38)]]
    assert action_rows == [[float(i) for i in range(122, 138)]]

    state_rows, action_rows = module.build_policy_vectors_from_table(
        table,
        config=config,
        policy_type="groot",
    )
    assert len(state_rows[0]) == 49
    assert len(action_rows[0]) == 53
    assert state_rows[0][0:9] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    synergy = load_module("dex1_hand_synergy", ROOT / "gr00t" / "dex1_hand_synergy.py")
    assert state_rows[0][18:25] == pytest.approx(
        synergy.dex1_to_hand(36.0, side="left", kind="state")
    )
    assert state_rows[0][32:39] == [22, 23, 24, 25, 26, 27, 28]
    assert state_rows[0][39:46] == [29, 30, 31, 32, 33, 34, 35]
    assert state_rows[0][46:49] == [19, 20, 21]
    assert action_rows[0][0:9] == [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert action_rows[0][18:25] == pytest.approx(
        synergy.dex1_to_hand(136.0, side="left", kind="action")
    )
    assert action_rows[0][32:39] == [122, 123, 124, 125, 126, 127, 128]
    assert action_rows[0][39:46] == [129, 130, 131, 132, 133, 134, 135]
    assert action_rows[0][46:49] == [119, 120, 121]
    assert action_rows[0][49:53] == [0.0, 0.0, 0.0, 0.0]


def test_delta_action_contract_encodes_and_decodes_chunk_boundaries() -> None:
    from model.subtask_policy_training import delta_action

    state = [0.0, 0.0, 0.0] + [float(index) for index in range(14)] + [0.2, 0.3]
    first_target = [float(index + 10) for index in range(14)] + [1.1, 1.2]
    second_target = [float(index + 12) for index in range(14)] + [1.3, 1.4]

    first = delta_action.encode_action(
        state=state, absolute_action=first_target, previous_arm_target=None
    )
    second = delta_action.encode_action(
        state=state,
        absolute_action=second_target,
        previous_arm_target=first_target[:14],
    )

    assert first[:14] == [10.0] * 14
    assert second[:14] == [2.0] * 14
    assert first[14:] == [1.1, 1.2]
    assert second[14:] == [1.3, 1.4]
    assert delta_action.decode_action_chunk(initial_state=state, delta_actions=[first, second]) == [
        first_target,
        second_target,
    ]


def test_materializer_delta_resets_at_episode_boundaries_and_keeps_gripper_absolute() -> None:
    module = load_module(
        "materialize_lerobot_training_view_delta",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    table = pa.table(
        {
            "observation.state.robot_q_current": [
                [float(index) for index in range(36)],
                [float(index) for index in range(36)],
                [float(index + 20) for index in range(36)],
            ],
            "observation.state.hand_state": [[0.0, 0.0]] * 3,
            "action.robot_q_desired": [
                [float(index + 100) for index in range(36)],
                [float(index + 102) for index in range(36)],
                [float(index + 300) for index in range(36)],
            ],
            "action.hand_cmd": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            "timestamp": [0.0, 1.0 / 30.0, 0.0],
            "frame_index": [0, 1, 0],
            "episode_index": [10, 10, 11],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
        }
    )

    state_rows, action_rows = module.build_policy_vectors_from_table(
        table,
        config=config,
        policy_type="act",
        action_representation="arm_delta_gripper_absolute",
    )

    assert state_rows[0][3:17] == [float(index) for index in range(22, 36)]
    assert action_rows[0][:14] == [100.0] * 14
    assert action_rows[1][:14] == [2.0] * 14
    assert action_rows[2][:14] == [300.0 - 20.0] * 14
    assert action_rows[0][14:] == [1.0, 2.0]
    assert action_rows[1][14:] == [3.0, 4.0]
    assert action_rows[2][14:] == [5.0, 6.0]


def test_fixed_heldout_split_has_no_validation_or_training_leakage(tmp_path) -> None:
    materializer = load_module(
        "materialize_lerobot_training_view_fixed_split",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    resolver = load_module(
        "resolve_training_split_fixed_split",
        ROOT / "scripts" / "resolve_training_split.py",
    )
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    episode_path = tmp_path / "source" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    (tmp_path / "source" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 10}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(10)),
                "tasks": [["flip table"] for _ in range(10)],
                "source_episode_name": [f"recording_{index:02d}" for index in range(10)],
            }
        ),
        episode_path,
    )

    split = materializer.build_grouped_episode_split(
        tmp_path / "source",
        config=config,
        pq=pq,
        validation_fraction=0.0,
        heldout_episode_indices=[2, 7],
    )
    assert split["splits"]["test"]["episode_indices"] == [2, 7]
    assert split["splits"]["validation"]["episode_indices"] == []
    assert 2 not in split["splits"]["train"]["episode_indices"]
    assert 7 not in split["splits"]["train"]["episode_indices"]
    assert split["fractions"] == {"train": 0.8, "validation": 0.0, "test": 0.2}

    dataset_root = tmp_path / "view"
    split_path = dataset_root / resolver.SPLIT_PATH
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    resolved = resolver.resolve_split(dataset_root, allow_empty_validation=True)
    assert resolved["DATASET_EVAL_SPLIT"] == "0"
    assert resolved["VALIDATION_EPISODE_COUNT"] == 0


def test_fixed_training_and_heldout_splits_leave_other_episodes_unused(tmp_path) -> None:
    materializer = load_module(
        "materialize_lerobot_training_view_fixed_train_split",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    episode_path = tmp_path / "source" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    (tmp_path / "source" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 6}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(6)),
                "tasks": [["flip table"] for _ in range(6)],
                "source_episode_name": [f"recording_{index:02d}" for index in range(6)],
            }
        ),
        episode_path,
    )

    split = materializer.build_grouped_episode_split(
        tmp_path / "source",
        config=config,
        pq=pq,
        validation_fraction=0.0,
        training_episode_indices=[0, 1, 2],
        heldout_episode_indices=[4, 5],
    )

    assert split["splits"]["train"]["episode_indices"] == [0, 1, 2]
    assert split["splits"]["validation"]["episode_indices"] == []
    assert split["splits"]["test"]["episode_indices"] == [4, 5]


def test_chunk_reset_metrics_are_zero_for_exact_decoded_targets() -> None:
    evaluator = load_module(
        "evaluate_delta_chunk_reset", ROOT / "scripts" / "evaluate_delta_chunk_reset.py"
    )
    target = __import__("numpy").asarray(
        [[float(index) for index in range(16)], [float(index + 1) for index in range(16)]],
        dtype="float32",
    )
    metrics = evaluator.compute_metrics(target, target.copy())
    assert metrics["arm_rmse_rad"] == 0.0
    assert metrics["arm_p95_abs_error_rad"] == 0.0
    assert metrics["gripper_mae"] == 0.0
    assert metrics["gripper_transition_f1"] == 1.0


def test_materializer_rejects_nonfinite_source_values() -> None:
    module = load_module(
        "materialize_lerobot_training_view_finite",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        module.as_float_list([0.0, float("nan")])


def test_grouped_split_is_disjoint_and_resolves_for_lerobot(tmp_path) -> None:
    materializer = load_module(
        "materialize_lerobot_training_view_split",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    resolver = load_module(
        "resolve_training_split",
        ROOT / "scripts" / "resolve_training_split.py",
    )
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    pa = __import__("pyarrow")
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    episode_path = tmp_path / "source" / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episode_path.parent.mkdir(parents=True)
    (tmp_path / "source" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 10}), encoding="utf-8"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": list(range(10)),
                "tasks": [["flip table"] for _ in range(10)],
                "source_episode_name": [f"recording_{index:02d}" for index in range(10)],
            }
        ),
        episode_path,
    )

    split = materializer.build_grouped_episode_split(tmp_path / "source", config=config, pq=pq)
    dataset_root = tmp_path / "view"
    split_path = dataset_root / resolver.SPLIT_PATH
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(split), encoding="utf-8")
    resolved = resolver.resolve_split(dataset_root)

    episode_sets = {
        name: set(value["episode_indices"]) for name, value in split["splits"].items()
    }
    assert episode_sets["train"].isdisjoint(episode_sets["validation"])
    assert episode_sets["train"].isdisjoint(episode_sets["test"])
    assert episode_sets["validation"].isdisjoint(episode_sets["test"])
    selected = json.loads(resolved["DATASET_EPISODES_JSON"])
    assert selected[-resolved["VALIDATION_EPISODE_COUNT"] :] == sorted(episode_sets["validation"])
    assert math.ceil(len(selected) * float(resolved["DATASET_EVAL_SPLIT"])) == len(
        episode_sets["validation"]
    )


def test_overfit_split_is_a_checked_subset_and_keeps_test_unused(tmp_path) -> None:
    resolver = load_module(
        "resolve_training_split_overfit",
        ROOT / "scripts" / "resolve_training_split.py",
    )
    payload = {
        "schema_version": "team_ramen_grouped_episode_split_v1",
        "splits": {
            "train": {
                "episode_indices": [0, 1, 8, 20, 21],
                "source_episode_names": ["e0", "e1", "e8", "e20", "e21"],
            },
            "validation": {
                "episode_indices": [139, 140],
                "source_episode_names": ["e139", "e140"],
            },
            "test": {
                "episode_indices": [156, 157],
                "source_episode_names": ["e156", "e157"],
            },
        },
    }
    unsigned = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = __import__("hashlib").sha256(unsigned).hexdigest()
    split_path = tmp_path / resolver.SPLIT_PATH
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    resolved = resolver.resolve_split(
        tmp_path,
        overfit_train_episodes=[0, 1, 8, 20],
        overfit_validation_count=1,
    )

    assert json.loads(resolved["DATASET_EPISODES_JSON"]) == [0, 1, 8, 20, 139]
    assert resolved["TRAIN_EPISODE_COUNT"] == 4
    assert resolved["VALIDATION_EPISODE_COUNT"] == 1
    assert resolved["TEST_EPISODE_COUNT"] == 2
    assert resolved["OVERFIT_MODE"] is True
    assert math.ceil(5 * float(resolved["DATASET_EVAL_SPLIT"])) == 1

    with pytest.raises(ValueError, match="declared train split"):
        resolver.resolve_split(tmp_path, overfit_train_episodes=[0, 139])


def test_training_view_fingerprint_changes_with_source_data(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view_fingerprint",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    source = tmp_path / "source"
    for relative, content in (
        ("meta/info.json", b"{}"),
        ("meta/stats.json", b"{}"),
        ("meta/tasks.parquet", b"tasks"),
        ("meta/episodes/chunk-000/file-000.parquet", b"episodes"),
        ("data/chunk-000/file-000.parquet", b"data-v1"),
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    before = module.source_dataset_fingerprint(source)
    (source / "data/chunk-000/file-000.parquet").write_bytes(b"data-v2")
    after = module.source_dataset_fingerprint(source)

    assert before != after


def test_groot_materializer_rejects_ambiguous_source_eef_format() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    config["source_dataset"]["eef_pose_format"]["angle_unit"] = "degree"

    with pytest.raises(ValueError, match="Euler-XYZ radians"):
        module.validate_source_eef_pose_format(config["source_dataset"])


def test_groot_data_rewrite_emits_49d_53d_parquet_and_stats(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_file = tmp_path / "source" / "data" / "chunk-000" / "file-000.parquet"
    source_file.parent.mkdir(parents=True)
    rows = 2
    table = pa.table(
        {
            "observation.state.ee_state": [[0.0] * 12 for _ in range(rows)],
            "observation.state.robot_q_current": [
                [float(i + row) for i in range(36)] for row in range(rows)
            ],
            "observation.state.hand_state": [[4.0, 4.5], [3.5, 4.0]],
            "action.ee_action": [
                [0.01 * (i + row) for i in range(12)] for row in range(rows)
            ],
            "action.robot_q_desired": [
                [float(i + row) + 0.1 for i in range(36)] for row in range(rows)
            ],
            "action.hand_cmd": [[3.9, 4.4], [3.4, 3.9]],
            "timestamp": [0.0, 1.0 / 30.0],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "index": [0, 1],
            "task_index": [0, 0],
        }
    )
    pq.write_table(table, source_file)

    stats = module.rewrite_data_parquets(
        tmp_path / "source",
        tmp_path / "output",
        config=config,
        policy_type="groot",
        pa=pa,
        pq=pq,
    )

    rewritten = pq.read_table(tmp_path / "output" / "data" / "chunk-000" / "file-000.parquet")
    assert rewritten.schema.field("observation.state").type.list_size == 49
    assert rewritten.schema.field("action").type.list_size == 53
    assert len(stats["observation.state"]["mean"]) == 49
    assert len(stats["action"]["q99"]) == 53


def test_training_view_stats_exclude_validation_and_test_episodes(tmp_path) -> None:
    module = load_module(
        "materialize_lerobot_training_view_train_stats",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    pa = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_file = tmp_path / "source" / "data" / "chunk-000" / "file-000.parquet"
    source_file.parent.mkdir(parents=True)
    table = pa.table(
        {
            "observation.state.robot_q_current": [[0.0] * 36, [0.0] * 36, [100.0] * 36],
            "observation.state.hand_state": [[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]],
            "action.robot_q_desired": [[1.0] * 36, [1.0] * 36, [101.0] * 36],
            "action.hand_cmd": [[1.0, 1.0], [1.0, 1.0], [101.0, 101.0]],
            "timestamp": [0.0, 1.0 / 30.0, 0.0],
            "frame_index": [0, 1, 0],
            "episode_index": [0, 0, 1],
            "index": [0, 1, 2],
            "task_index": [0, 0, 0],
        }
    )
    pq.write_table(table, source_file)

    stats = module.rewrite_data_parquets(
        tmp_path / "source",
        tmp_path / "output",
        config=config,
        policy_type="act",
        stats_episode_indices={0},
        pa=pa,
        pq=pq,
    )

    assert stats["observation.state"]["mean"] == pytest.approx([0.0] * 19)
    assert stats["action"]["mean"] == pytest.approx([1.0] * 16)


def test_lerobot_training_view_features_and_stats_use_cam0_as_head_left() -> None:
    module = load_module(
        "materialize_lerobot_training_view",
        ROOT / "scripts" / "materialize_lerobot_training_view.py",
    )
    config = json.loads((ROOT / "configs" / "subtask_training.json").read_text())
    source_features = {
        "observation.images.cam_0": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_1": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_2": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.images.cam_3": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state.ee_state": {"dtype": "float32", "shape": [12]},
        "observation.state.robot_q_current": {"dtype": "float32", "shape": [36]},
        "observation.state.hand_state": {"dtype": "float32", "shape": [2]},
        "action.ee_action": {"dtype": "float32", "shape": [12]},
        "action.robot_q_desired": {"dtype": "float32", "shape": [36]},
        "action.hand_cmd": {"dtype": "float32", "shape": [2]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }

    features = module.build_training_features(
        config=config,
        policy_type="act",
        source_features=source_features,
    )
    assert list(key for key in features if key.startswith("observation.images.")) == [
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert features["observation.images.head_left"] == {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
    }
    assert features["observation.state"]["shape"] == [19]
    assert features["action"]["shape"] == [16]

    source_stats = {
        "observation.state.robot_q_current": {"min": list(range(36)), "max": list(range(36)), "count": [10]},
        "observation.state.hand_state": {"min": [36, 37], "max": [36, 37], "count": [10]},
        "action.robot_q_desired": {"min": list(range(100, 136)), "max": list(range(100, 136)), "count": [10]},
        "action.hand_cmd": {"min": [136, 137], "max": [136, 137], "count": [10]},
        "observation.images.cam_0": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
        "observation.images.cam_2": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
        "observation.images.cam_3": {"min": [[[0.0]]], "max": [[[1.0]]], "count": [10]},
    }
    mapped_stats = {
        "observation.state": {"min": list(range(49)), "max": list(range(49)), "count": [10]},
        "action": {"min": list(range(100, 153)), "max": list(range(100, 153)), "count": [10]},
    }
    stats = module.build_training_stats(
        config=config,
        policy_type="groot",
        source_stats=source_stats,
        source_features=source_features,
        mapped_vector_stats=mapped_stats,
    )
    assert stats["observation.images.head_left"] == source_stats["observation.images.cam_0"]
    assert stats["observation.state"] == mapped_stats["observation.state"]
    assert stats["action"] == mapped_stats["action"]


def test_groot_real_g1_relative_eef_slot_mapping() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    synergy = load_module("dex1_hand_synergy_mapping", ROOT / "gr00t" / "dex1_hand_synergy.py")
    state, action = module.map_source_row_to_real_g1_relative_eef(
        ee_state=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.4, -0.2, 0.1, 0.0, 0.0, math.pi / 2],
        ee_action=[0.2, 0.4, 0.6, 0.0, 0.0, math.pi / 2, 0.3, -0.1, 0.2, 0.0, 0.0, 0.0],
        robot_q_current=[float(i) for i in range(36)],
        robot_q_desired=[float(i + 100) for i in range(36)],
        hand_state=[4.0, 3.0],
        hand_cmd=[2.0, 1.0],
    )

    assert len(state) == 49
    assert len(action) == 53
    assert state[:9] == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert state[9:18] == pytest.approx([0.4, -0.2, 0.1, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert state[18:25] == pytest.approx(
        synergy.dex1_to_hand(4.0, side="left", kind="state")
    )
    assert state[25:32] == pytest.approx(
        synergy.dex1_to_hand(3.0, side="right", kind="state")
    )
    assert state[32:39] == [22, 23, 24, 25, 26, 27, 28]
    assert state[39:46] == [29, 30, 31, 32, 33, 34, 35]
    assert state[46:49] == [19, 20, 21]
    assert action[:9] == pytest.approx([0.2, 0.4, 0.6, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0])
    assert action[18:25] == pytest.approx(
        synergy.dex1_to_hand(2.0, side="left", kind="action")
    )
    assert action[25:32] == pytest.approx(
        synergy.dex1_to_hand(1.0, side="right", kind="action")
    )
    assert action[32:39] == [122, 123, 124, 125, 126, 127, 128]
    assert action[39:46] == [129, 130, 131, 132, 133, 134, 135]
    assert action[46:49] == [119, 120, 121]
    assert action[49:53] == [0.0, 0.0, 0.0, 0.0]


def test_groot_relative_eef_uses_se3_not_component_subtraction() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    current = module.source_euler_xyz_pose_to_xyz_rot6d([1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2])
    target = module.source_euler_xyz_pose_to_xyz_rot6d([1.0, 3.0, 3.0, 0.0, 0.0, math.pi])

    relative = module.absolute_eef_xyz_rot6d_to_relative(current, target)

    assert relative == pytest.approx([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0], abs=1e-7)


def test_groot_real_g1_relative_eef_modality_contract() -> None:
    module = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    modality = module.build_real_g1_relative_eef_modality_json(
        [
            "observation.images.head_left",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        ]
    )

    assert list(modality["state"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
    ]
    assert list(modality["action"]) == [
        "left_wrist_eef_9d",
        "right_wrist_eef_9d",
        "left_hand",
        "right_hand",
        "left_arm",
        "right_arm",
        "waist",
        "base_height_command",
        "navigate_command",
    ]
    assert modality["meta"]["action_configs"]["left_wrist_eef_9d"] == {
        "rep": "RELATIVE",
        "type": "EEF",
        "format": "XYZ_ROT6D",
        "state_key": "left_wrist_eef_9d",
    }
    assert list(modality["video"]) == ["head_left", "left_wrist", "right_wrist"]
    assert modality["meta"]["dex1_hand_mapping"] == (
        "fixed official G1 seven-joint synergy with least-squares inverse"
    )


def test_lerobot_groot_processor_converts_complete_eef_chunk_from_current_state() -> None:
    patcher = load_module(
        "patch_lerobot_groot_relative_eef",
        ROOT / "scripts" / "patch_lerobot_groot_relative_eef.py",
    )
    assert patcher.patch_processor(patcher.processor_path(), check_only=True) is False

    import torch
    from lerobot.policies.groot.processor_groot import GrootN17PackInputsStep

    mapping = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    state = torch.zeros(1, mapping.REAL_G1_RELATIVE_EEF_STATE_DIM)
    action = torch.zeros(1, 2, mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    for span in (
        mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["left_wrist_eef_9d"],
        mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["right_wrist_eef_9d"],
    ):
        state[:, span[0] + 3 : span[1]] = identity
    for span in (
        mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_wrist_eef_9d"],
        mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_wrist_eef_9d"],
    ):
        action[:, :, span[0] + 3 : span[1]] = identity

    left_state_eef = mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["left_wrist_eef_9d"]
    left_eef = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_wrist_eef_9d"]
    current_eef = mapping.source_euler_xyz_pose_to_xyz_rot6d(
        [1.0, 2.0, 3.0, 0.0, 0.0, math.pi / 2]
    )
    target_eef = [
        mapping.source_euler_xyz_pose_to_xyz_rot6d(
            [1.0, 3.0, 3.0, 0.0, 0.0, math.pi]
        ),
        mapping.source_euler_xyz_pose_to_xyz_rot6d(
            [0.0, 2.0, 3.0, 0.0, 0.0, 0.0]
        ),
    ]
    state[0, slice(*left_state_eef)] = torch.tensor(current_eef)
    action[0, :, slice(*left_eef)] = torch.tensor(target_eef)
    left_arm_state = mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES["left_arm"]
    left_arm_action = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_arm"]
    state[:, slice(*left_arm_state)] = 1.0
    action[:, :, slice(*left_arm_action)] = torch.tensor([[[1.5] * 7, [2.0] * 7]])
    left_hand = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_hand"]
    waist = mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES["waist"]
    action[:, :, slice(*left_hand)] = 0.75
    action[:, :, slice(*waist)] = 0.25

    state_keys = list(mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES)
    action_keys = list(mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)
    raw_stats = {
        "state": {
            key: {"mean": [0.0] * (end - start)}
            for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES.items()
        },
        "action": {
            key: {"mean": [0.0] * (end - start)}
            for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES.items()
        },
    }
    modality = {
        "state": {"modality_keys": state_keys},
        "action": {
            "modality_keys": action_keys,
            "action_configs": [mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in action_keys],
        },
    }
    processor = GrootN17PackInputsStep(raw_stats=raw_stats, modality_config=modality)

    converted = processor._convert_relative_action_groups_for_training(action, state)

    expected_eef = torch.tensor(
        [
            mapping.absolute_eef_xyz_rot6d_to_relative(current_eef, target)
            for target in target_eef
        ]
    )
    assert converted[0, :, slice(*left_eef)] == pytest.approx(
        expected_eef,
        abs=1e-6,
    )
    assert converted[0, :, slice(*left_arm_action)] == pytest.approx(
        torch.tensor([[0.5] * 7, [1.0] * 7])
    )
    assert converted[0, :, slice(*left_hand)] == pytest.approx(torch.full((2, 7), 0.75))
    assert converted[0, :, slice(*waist)] == pytest.approx(torch.full((2, 3), 0.25))


def test_prepare_groot_overlay_preserves_action_contract_and_enables_three_cameras(tmp_path) -> None:
    overlay_module = load_module(
        "prepare_groot_n17_real_g1_overlay",
        ROOT / "scripts" / "prepare_groot_n17_real_g1_overlay.py",
    )
    mapping = load_module("g1_full_body_mapping", ROOT / "gr00t" / "g1_full_body_mapping.py")
    source = tmp_path / "source"
    source.mkdir()
    tag = mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_TAG
    state_keys = list(mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES)
    action_keys = list(mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES)
    processor_config = {
        "processor_kwargs": {
            "use_relative_action": True,
            "max_action_horizon": mapping.GROOT_N17_NATIVE_ACTION_HORIZON,
            "modality_configs": {
                tag: {
                    "video": {"delta_indices": [-20, 0], "modality_keys": ["ego_view"]},
                    "state": {"delta_indices": [0], "modality_keys": state_keys},
                    "action": {
                        "delta_indices": list(range(mapping.GROOT_N17_NATIVE_ACTION_HORIZON)),
                        "modality_keys": action_keys,
                        "action_configs": [
                            mapping.REAL_G1_RELATIVE_EEF_ACTION_CONFIGS[key] for key in action_keys
                        ],
                    },
                }
            },
        }
    }
    statistics = {
        tag: {
            "state": {
                key: {"mean": [0.0] * (end - start)}
                for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_STATE_SLICES.items()
            },
            "action": {
                key: {"mean": [0.0] * (end - start)}
                for key, (start, end) in mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES.items()
            },
            "relative_action": {
                key: {
                    "mean": [
                        [0.0] * (mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES[key][1]
                                 - mapping.REAL_G1_RELATIVE_EEF_ACTION_SLICES[key][0])
                        for _ in range(mapping.GROOT_N17_NATIVE_ACTION_HORIZON)
                    ]
                }
                for key in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_arm", "right_arm")
            },
        }
    }
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "action_horizon": mapping.GROOT_N17_NATIVE_ACTION_HORIZON,
            }
        )
    )
    (source / "embodiment_id.json").write_text(
        json.dumps({tag: mapping.REAL_G1_RELATIVE_EEF_EMBODIMENT_ID})
    )
    (source / "processor_config.json").write_text(json.dumps(processor_config))
    (source / "statistics.json").write_text(json.dumps(statistics))
    (source / "model.safetensors").write_bytes(b"fixture")

    output = overlay_module.prepare_overlay(
        source_root=source,
        output_root=tmp_path / "overlay",
        model_path="fixture/gr00t-n1.7",
        revision="fixture-revision",
        force=False,
    )

    overlay_config = json.loads((output / "processor_config.json").read_text())
    overlay_modality = overlay_config["processor_kwargs"]["modality_configs"][tag]
    assert overlay_modality["video"] == {
        "delta_indices": [-20, 0],
        "modality_keys": ["head_left", "left_wrist", "right_wrist"],
    }
    assert overlay_modality["action"] == processor_config["processor_kwargs"]["modality_configs"][tag][
        "action"
    ]
    assert (output / "model.safetensors").is_symlink()
    assert overlay_module.prepare_overlay(
        source_root=source,
        output_root=output,
        model_path="fixture/gr00t-n1.7",
        revision="fixture-revision",
        force=False,
    ) == output


def test_groot_contract_is_49d_53d_packed_132d_and_never_54d() -> None:
    contract = load_module("n17_contract", ROOT / "gr00t" / "n17_contract.py")
    mapping = load_module("g1_full_body_mapping_contract", ROOT / "gr00t" / "g1_full_body_mapping.py")

    assert contract.LOGICAL_STATE_DIM == mapping.REAL_G1_RELATIVE_EEF_STATE_DIM == 49
    assert contract.LOGICAL_ACTION_DIM == mapping.REAL_G1_RELATIVE_EEF_ACTION_DIM == 53
    assert contract.PACKED_STATE_DIM == mapping.GROOT_N17_PACKED_STATE_DIM == 132
    assert contract.PACKED_ACTION_DIM == mapping.GROOT_N17_PACKED_ACTION_DIM == 132
    assert contract.ACTION_HORIZON == mapping.GROOT_N17_NATIVE_ACTION_HORIZON == 40
    assert contract.VALID_ACTION_DIM == mapping.GROOT_N17_VALID_ACTION_DIM == 46
    with pytest.raises(ValueError, match="53-D"):
        mapping._require_dim("logical action", [0.0] * 54, 53)


def test_dex1_official_synergy_roundtrip_and_direction() -> None:
    synergy = load_module("dex1_hand_synergy_roundtrip", ROOT / "gr00t" / "dex1_hand_synergy.py")

    for side in ("left", "right"):
        for kind in ("state", "action"):
            opened = synergy.dex1_to_hand(4.5, side=side, kind=kind)
            closed = synergy.dex1_to_hand(0.0, side=side, kind=kind)
            assert opened != closed
            for dex1 in (0.0, 0.5, 2.25, 4.0, 4.5):
                hand = synergy.dex1_to_hand(dex1, side=side, kind=kind)
                assert synergy.hand_to_dex1(hand, side=side, kind=kind) == pytest.approx(
                    dex1, abs=1e-12
                )


def test_restore_groot_base_model_path_makes_checkpoints_portable(tmp_path) -> None:
    module = load_module(
        "restore_groot_base_model_path",
        ROOT / "scripts" / "restore_groot_base_model_path.py",
    )
    checkpoint = tmp_path / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    config_path = checkpoint / "config.json"
    config_path.write_text(
        json.dumps({"type": "groot", "base_model_path": "/tmp/local-overlay"}) + "\n"
    )

    changed = module.restore_base_model_paths(
        tmp_path,
        runtime_path="/tmp/local-overlay",
        canonical_path="nvidia/GR00T-N1.7-3B",
    )

    assert changed == 1
    assert json.loads(config_path.read_text())["base_model_path"] == "nvidia/GR00T-N1.7-3B"
