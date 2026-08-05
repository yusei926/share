from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from inference.desktop.model_evaluation import cli as cli_module
from inference.desktop.model_evaluation import artifacts
from inference.desktop.model_evaluation.adapters import (
    CanonicalObservation,
    adapter_for,
)
from inference.desktop.model_evaluation.artifacts import (
    LOCK_FILENAME,
    download_plan,
    load_prepared_spec,
    seal_local_artifacts,
    validate_prepared_artifacts,
)
from inference.desktop.model_evaluation.cli import (
    adapter_dry_run,
    real_command,
    runner_argv,
)
from inference.desktop.model_evaluation.launch import parse_args as parse_launch_args
from inference.desktop.model_evaluation.offline_model import (
    _request,
    load_bundle,
)
from inference.desktop.model_evaluation.registry import (
    CANONICAL_OUTPUT,
    DEPLOYMENT_SCHEMA,
    LOWER_BODY_OWNER,
    ModelSpec,
    get_model_spec,
    load_registry,
    model_spec_from_manifest,
)
from inference.desktop.model_evaluation.resolver import (
    UnsupportedModelError,
    _contract_metadata_summary,
    _is_inference_artifact,
    _lfs_sha256,
    onboarding_manifest_draft,
    parse_hf_reference,
    resolve_model,
)
from inference.desktop.upper_policy.motion_limits import (
    FLIP_TABLE_ARM_ACCELERATION_RAD_S2,
    FLIP_TABLE_ARM_VELOCITY_RAD_S,
    FLIP_TABLE_HAND_ACCELERATION_FRACTION_S2,
    FLIP_TABLE_HAND_VELOCITY_FRACTION_S,
)


def _observation(spec_name: str) -> CanonicalObservation:
    spec = get_model_spec(spec_name)
    return CanonicalObservation(
        body_joint_position_rad=np.arange(29, dtype=np.float64) / 100.0,
        dex1_opening_fraction=np.asarray([0.25, 0.75]),
        camera_jpeg={role: role.encode() for role in spec.camera_roles},
        eef_xyz_euler=np.zeros(12),
    )


def _remote_manifest(model: str) -> dict[str, object]:
    result = get_model_spec(model).to_lock_mapping()
    for key in ("repo_id", "revision", "manifest_source"):
        result.pop(key, None)
    return result


def test_registry_covers_requested_models_and_has_one_safe_output() -> None:
    registry = load_registry()
    assert {
        "pick_legs_act_joint16_augxx_s40k",
        "pick_legs_groot_v1",
        "pick_legs_groot_v2_lora",
        "coarse_insert_groot_n17_v2",
        "flip_table_diffusion_chunk_relative_v2",
        "flip_table_groot_n17_v2_baseline_20k_candidate",
    } <= set(registry)
    assert {spec.canonical_output for spec in registry.values()} == {CANONICAL_OUTPUT}
    assert {spec.lower_body_command_dimensions for spec in registry.values()} == {0}
    assert all(spec.artifact.file_sha256 for spec in registry.values())
    assert {
        spec.execution_steps
        for spec in registry.values()
        if spec.family == "act_absolute_joint16_v1"
    } == {30}


def test_full_hf_path_and_url_resolve_for_registered_model_without_network() -> None:
    path = "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora"
    assert get_model_spec(path).model_id == "pick_legs_groot_v2_lora"
    assert (
        get_model_spec(f"https://huggingface.co/{path}").model_id
        == "pick_legs_groot_v2_lora"
    )
    resolved = resolve_model(path, revision=get_model_spec(path).revision)
    assert resolved.resolution_source == "legacy_catalog"


@pytest.mark.parametrize("model", list(load_registry()))
def test_adapter_dry_run_never_initializes_physical_transport(model: str) -> None:
    before = set(sys.modules)
    report = adapter_dry_run(get_model_spec(model))
    assert report["canonical_action_shape"][1] == 16
    assert report["lower_body_command_dimensions"] == 0
    assert report["model_weights_loaded"] is False
    assert report["robot_command_sent"] is False
    assert report["dds_initialized"] is False
    imported = set(sys.modules) - before
    assert not any(name.startswith(("unitree_sdk2py", "cyclonedds")) for name in imported)


def test_pick_adapter_drops_root_legs_waist_and_scales_dex1() -> None:
    spec = get_model_spec("pick_legs_groot_v1")
    native = np.zeros((16, 38))
    native[:, :22] = 999.0
    native[:, 22:36] = np.arange(14)
    native[:, 36:] = [1.125, 3.375]
    result = adapter_for(spec).canonical_action(native, _observation(spec.model_id))
    np.testing.assert_allclose(result[:, :14], np.tile(np.arange(14), (16, 1)))
    np.testing.assert_allclose(result[:, 14:], [[0.25, 0.75]] * 16)


def test_act_joint16_adapter_maps_only_arms_and_hands_and_clamps() -> None:
    spec = get_model_spec("pick_legs_act_joint16_augxx_s40k")
    observation = _observation(spec.model_id)
    state = adapter_for(spec).model_state(observation)
    np.testing.assert_allclose(state[:14], observation.body_joint_position_rad[15:29])
    np.testing.assert_allclose(state[14:], [1.125, 3.375])

    native = np.zeros((30, 16), dtype=np.float64)
    native[:, 0] = -99.0
    native[:, 1] = 99.0
    native[:, 14:] = [1.125, 3.375]
    result = adapter_for(spec).canonical_action(native, observation)
    assert result.shape == (30, 16)
    np.testing.assert_allclose(result[:, 0], -1.396479)
    np.testing.assert_allclose(result[:, 1], 1.222892)
    np.testing.assert_allclose(result[:, 14:], [[0.25, 0.75]] * 30)

    request = adapter_for(spec).offline_request(observation, state)
    assert list(request["cameras"]) == [
        "observation.images.cam_0",
        "observation.images.cam_1",
        "observation.images.cam_2",
        "observation.images.cam_3",
    ]
    assert request["cameras"]["observation.images.cam_2"] == b"left_wrist"
    assert request["cameras"]["observation.images.cam_3"] == b"right_wrist"


def test_relative_eef_adapter_never_emits_waist() -> None:
    spec = get_model_spec("coarse_insert_groot_n17_v2")
    native = np.zeros((16, 53))
    native[:, 32:46] = np.arange(14)
    native[:, 46:53] = 999.0
    native[:, 18] = -0.375
    native[:, 25] = 1.125
    result = adapter_for(spec).canonical_action(native, _observation(spec.model_id))
    np.testing.assert_allclose(result[:, :14], np.tile(np.arange(14), (16, 1)))
    np.testing.assert_allclose(result[:, 14:], [[0.25, 0.75]] * 16)
    assert not np.any(result == 999.0)


def test_diffusion_relative_arm_is_anchored_exactly_once() -> None:
    spec = get_model_spec("flip_table_diffusion_chunk_relative_v2")
    observation = _observation(spec.model_id)
    native = np.zeros((16, 16))
    native[:, :14] = 0.1
    native[:, 14:] = [1.125, 3.375]
    result = adapter_for(spec).canonical_action(native, observation)
    expected = observation.body_joint_position_rad[15:29] + 0.1
    np.testing.assert_allclose(result[:, :14], np.tile(expected, (16, 1)))
    np.testing.assert_allclose(result[:, 14:], [[0.25, 0.75]] * 16)


def test_furniture_candidate_adapter_exposes_only_arms_and_dex1() -> None:
    spec = get_model_spec("flip_table_groot_n17_v2_baseline_20k_candidate")
    observation = _observation(spec.model_id)
    state = adapter_for(spec).model_state(observation)
    assert state.shape == (49,)
    native = np.zeros((40, 16), dtype=np.float64)
    native[:, :14] = np.arange(14, dtype=np.float64)
    native[:, 14:] = [1.125, 3.375]
    result = adapter_for(spec).canonical_action(native, observation)
    np.testing.assert_allclose(result[:, :14], np.tile(np.arange(14), (40, 1)))
    np.testing.assert_allclose(result[:, 14:], [[0.25, 0.75]] * 40)
    request = adapter_for(spec).offline_request(observation, state)
    assert set(request["camera_history"]) == {
        "observation.images.head_left",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    }
    assert all(len(history) == 2 for history in request["camera_history"].values())
    assert request["task"] == "flip table"


def test_download_plan_is_commit_pinned_and_excludes_training_state() -> None:
    spec = get_model_spec("pick_legs_groot_v1")
    plan = download_plan(spec, Path("/tmp/model"))
    assert plan["revision"] == spec.revision
    assert "**/optimizer.pt" in plan["ignore_patterns"]
    assert "**/optimizer_state.safetensors" in plan["ignore_patterns"]


def test_manifest_cannot_select_remote_executable_or_unhashed_weight() -> None:
    base = _remote_manifest("flip_table_diffusion_chunk_relative_v2")
    base["model_id"] = "remote"
    base["runner"] = "malicious.module"
    with pytest.raises(ValueError, match="unsupported fields"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="1" * 40,
            source="test",
        )
    base.pop("runner")
    base["artifact"]["file_sha256"] = {}
    with pytest.raises(ValueError, match="SHA-256 pinned"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="1" * 40,
            source="test",
        )


def test_manifest_rejects_path_traversal_and_lower_body_control() -> None:
    base = _remote_manifest("flip_table_diffusion_chunk_relative_v2")
    base["model_id"] = "remote"
    base["artifact"]["checkpoint_subdir"] = "../escape"
    with pytest.raises(ValueError, match="safe relative"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="1" * 40,
            source="test",
        )
    base["artifact"]["checkpoint_subdir"] = ""
    base["lower_body_owner"] = "policy"
    with pytest.raises(ValueError, match="Regular Mode"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="1" * 40,
            source="test",
        )


def test_runner_command_has_sealed_identity_and_no_passthrough() -> None:
    spec = get_model_spec("flip_table_diffusion_chunk_relative_v2")
    argv = runner_argv(
        spec,
        Path("/tmp/diffusion"),
        actuate=True,
        interface="test-nic",
        image_server_ip="192.0.2.10",
        max_seconds=1.0,
        log_path=Path("/tmp/run/events.jsonl"),
    )
    assert spec.repo_id in argv
    assert spec.revision in argv
    assert spec.expected_model_sha256 in argv
    assert argv[argv.index("--log") + 1] == "/tmp/run/events.jsonl"
    assert argv[-1] == "--actuate"
    rendered = real_command(spec, Path("/tmp/diffusion"))
    assert "-e model-eval" in rendered
    assert spec.repo_id in rendered
    assert spec.revision in rendered
    assert "--actuate" not in rendered
    with pytest.raises(SystemExit):
        parse_launch_args(
            [
                spec.repo_id,
                "--revision",
                spec.revision,
                "--local-dir",
                "/tmp/diffusion",
                "--",
                "--checkpoint",
                "/tmp/unsealed",
            ]
        )


def test_act_runner_receives_checkpoint_n_action_steps() -> None:
    spec = get_model_spec("pre_straddle_act_augxx_s40k")
    argv = runner_argv(
        spec,
        Path("/tmp/act"),
        interface="test-nic",
        image_server_ip="192.0.2.10",
    )
    assert argv[argv.index("--action-execution-steps") + 1] == "30"
    assert argv[argv.index("--replan-after-steps") + 1] == "26"


@pytest.mark.parametrize(
    ("model", "expected", "expected_replan"),
    [
        ("pick_legs_groot_v1", "8", "4"),
        ("coarse_insert_groot_n17_v2", "8", "4"),
        ("flip_table_diffusion_chunk_relative_v2", "8", "6"),
    ],
)
def test_async_runner_receives_sealed_execution_prefix(
    model: str, expected: str, expected_replan: str
) -> None:
    spec = get_model_spec(model)
    argv = runner_argv(
        spec,
        Path("/tmp/model"),
        interface="test-nic",
        image_server_ip="192.0.2.10",
    )
    assert argv[argv.index("--action-execution-steps") + 1] == expected
    assert argv[argv.index("--replan-after-steps") + 1] == expected_replan


def test_common_launcher_forwards_only_bounded_motion_limits() -> None:
    spec = get_model_spec("flip_table_diffusion_chunk_relative_v2")
    limits = {
        "pre_motion_arm_velocity_rad_s": 0.5,
        "pre_motion_arm_acceleration_rad_s2": 1.0,
        "pre_motion_waypoint_tolerance_rad": 0.10,
        "pre_motion_stage_timeout_s": 15.0,
        "policy_arm_velocity_rad_s": FLIP_TABLE_ARM_VELOCITY_RAD_S,
        "policy_arm_acceleration_rad_s2": FLIP_TABLE_ARM_ACCELERATION_RAD_S2,
        "policy_hand_velocity_fraction_s": FLIP_TABLE_HAND_VELOCITY_FRACTION_S,
        "policy_hand_acceleration_fraction_s2": (
            FLIP_TABLE_HAND_ACCELERATION_FRACTION_S2
        ),
    }
    argv = runner_argv(
        spec,
        Path("/tmp/diffusion"),
        actuate=True,
        interface="test-nic",
        image_server_ip="192.0.2.10",
        safety_limits=limits,
    )
    for name, value in limits.items():
        flag = "--" + name.replace("_", "-")
        assert argv[argv.index(flag) + 1] == str(value)
    assert argv[-1] == "--actuate"

    with pytest.raises(ValueError, match="unsupported safety"):
        runner_argv(
            spec,
            Path("/tmp/diffusion"),
            interface="test-nic",
            image_server_ip="192.0.2.10",
            safety_limits={"checkpoint": 1.0},
        )
    with pytest.raises(ValueError, match="reviewed range"):
        runner_argv(
            spec,
            Path("/tmp/diffusion"),
            interface="test-nic",
            image_server_ip="192.0.2.10",
            safety_limits={
                "policy_arm_velocity_rad_s": FLIP_TABLE_ARM_VELOCITY_RAD_S + 0.01
            },
        )
    with pytest.raises(ValueError, match="reviewed range"):
        runner_argv(
            get_model_spec("pick_legs_groot_v1"),
            Path("/tmp/groot"),
            interface="test-nic",
            image_server_ip="192.0.2.10",
            safety_limits={"policy_arm_velocity_rad_s": 1.01},
        )

    parsed = parse_launch_args(
        [
            spec.repo_id,
            "--revision",
            spec.revision,
            "--local-dir",
            "/tmp/diffusion",
            "--policy-arm-velocity-rad-s",
            "0.5",
        ]
    )
    assert parsed.policy_arm_velocity_rad_s == 0.5


def test_local_seal_detects_post_validation_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"weight"
    digest = hashlib.sha256(payload).hexdigest()
    raw = {
        "repo_id": "Team-RAMEN/test",
        "revision": "1" * 40,
        "family": "diffusion_chunk_relative_v1",
        "task": "test",
        "camera_roles": ["head_left", "left_wrist", "right_wrist"],
        "observation_horizon": 2,
        "model_state_dim": 19,
        "model_action_dim": 16,
        "model_action_horizon": 16,
        "execution_steps": 8,
        "state_semantics": "waist3+arms14+dex1_physical2",
        "action_semantics": (
            "arm14_relative_to_measured_chunk_start+"
            "dex1_physical2_absolute;zscore;clip_sample_false"
        ),
        "artifact": {
            "checkpoint_subdir": "",
            "required_files": ["model.safetensors"],
            "allow_patterns": ["model.safetensors"],
            "file_sha256": {"model.safetensors": digest},
        },
        "expected_model_sha256": digest,
    }
    spec = ModelSpec.from_mapping("test", raw)
    (tmp_path / "model.safetensors").write_bytes(payload)
    monkeypatch.setattr(artifacts, "_validate_static_contract", lambda *_: None)
    seal_local_artifacts(tmp_path, spec)
    assert (tmp_path / LOCK_FILENAME).is_file()
    assert validate_prepared_artifacts(tmp_path, spec)["tamper_check"] == "passed"
    loaded = load_prepared_spec(
        tmp_path, reference=spec.repo_id, revision=spec.revision
    )
    assert loaded.repo_id == spec.repo_id
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after validation|hash mismatch"):
        validate_prepared_artifacts(tmp_path, spec)


def test_parse_hf_url_captures_revision() -> None:
    repo, revision = parse_hf_reference(
        "https://huggingface.co/Team-RAMEN/model/tree/abc123"
    )
    assert repo == "Team-RAMEN/model"
    assert revision == "abc123"


def test_unknown_repo_without_manifest_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Info:
        sha = "2" * 40

    class Api:
        def model_info(self, *args: object, **kwargs: object) -> Info:
            return Info()

    def missing(**kwargs: object) -> str:
        raise FileNotFoundError("no manifest")

    monkeypatch.setattr(
        "inference.desktop.model_evaluation.resolver._hub",
        lambda: (Api(), missing),
    )
    with pytest.raises(UnsupportedModelError, match="actuation is refused"):
        resolve_model("Team-RAMEN/unknown")


def test_unknown_repo_with_valid_manifest_resolves_to_trusted_local_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "6" * 40
    manifest = _remote_manifest("flip_table_diffusion_chunk_relative_v2")
    manifest["model_id"] = "remote_diffusion"
    path = tmp_path / "iros_ramen_deployment.json"
    path.write_text(json.dumps(manifest))

    class Info:
        sha = revision

    class Api:
        def model_info(self, *args: object, **kwargs: object) -> Info:
            return Info()

    def download(**kwargs: object) -> str:
        assert kwargs["filename"] == "iros_ramen_deployment.json"
        assert kwargs["revision"] == revision
        return str(path)

    monkeypatch.setattr(
        "inference.desktop.model_evaluation.resolver._hub",
        lambda: (Api(), download),
    )
    resolved = resolve_model("Team-RAMEN/remote-diffusion")
    assert resolved.resolution_source == "hf_manifest"
    assert resolved.spec.repo_id == "Team-RAMEN/remote-diffusion"
    assert resolved.spec.revision == revision
    assert resolved.spec.runner == (
        "inference.desktop.upper_policy.run_flip_table_diffusion"
    )


def test_cli_reports_fail_closed_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> int:
        raise UnsupportedModelError("missing deployment contract")

    monkeypatch.setattr(cli_module, "main", fail)
    assert cli_module.cli_entry() == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "ERROR: missing deployment contract\n"


def test_contract_metadata_summary_does_not_expose_training_paths_or_episode_lists() -> None:
    summary = _contract_metadata_summary(
        {
            "type": "act",
            "input_features": {"observation.state": {"shape": [16]}},
            "output_features": {"action": {"shape": [16]}},
            "output_dir": "/home/private/workstation",
            "dataset": {"episodes": list(range(20_000)), "root": "/mnt/private"},
        }
    )
    assert summary == {
        "type": "act",
        "input_features": {"observation.state": {"shape": [16]}},
        "output_features": {"action": {"shape": [16]}},
    }
    assert "/home/private" not in json.dumps(summary)


def test_lfs_object_and_training_only_artifacts_are_handled() -> None:
    class Lfs:
        sha256 = "a" * 64

    assert _lfs_sha256(Lfs()) == "a" * 64
    assert _is_inference_artifact("pretrained_model/model.safetensors")
    assert not _is_inference_artifact("optimizer/state.pt")
    assert not _is_inference_artifact("optimizer_state.safetensors")
    assert not _is_inference_artifact("rng/random.pth")
    assert not _is_inference_artifact("rng_state.pth")


def test_manifest_semantics_cannot_disagree_with_trusted_family() -> None:
    base = _remote_manifest("flip_table_diffusion_chunk_relative_v2")
    base["model_id"] = "remote"
    base["action_semantics"] = "absolute full body"
    with pytest.raises(ValueError, match="semantics must exactly match"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="3" * 40,
            source="test",
        )


def test_manifest_rejects_unknown_top_level_and_artifact_fields() -> None:
    base = _remote_manifest("flip_table_diffusion_chunk_relative_v2")
    base["post_hook"] = "arbitrary.py"
    with pytest.raises(ValueError, match="unsupported fields"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="5" * 40,
            source="test",
        )
    base.pop("post_hook")
    assert isinstance(base["artifact"], dict)
    base["artifact"]["download_script"] = "arbitrary.py"
    with pytest.raises(ValueError, match="artifact contains unsupported"):
        model_spec_from_manifest(
            base,
            repo_id="Team-RAMEN/remote",
            revision="5" * 40,
            source="test",
        )


def test_onboarding_draft_stays_non_executable_and_filters_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "inference.desktop.model_evaluation.resolver.inspect_hf_model",
        lambda *args, **kwargs: {
            "repo_id": "Team-RAMEN/unknown",
            "revision": "4" * 40,
            "candidate_families": [],
            "weight_files": [
                {"path": "model.safetensors", "sha256": "a" * 64},
            ],
            "unresolved_contract_fields": ["action ordering"],
        },
    )
    draft = onboarding_manifest_draft("Team-RAMEN/unknown")
    assert draft["family"] == "REQUIRED"
    assert draft["artifact"]["required_files"] == ["model.safetensors"]
    assert draft["_onboarding"]["resolved_revision"] == "4" * 40
    with pytest.raises(ValueError, match="unsupported fields"):
        model_spec_from_manifest(
            draft,
            repo_id="Team-RAMEN/unknown",
            revision="4" * 40,
            source="test",
        )


@pytest.mark.parametrize("model", list(load_registry()))
def test_offline_bundle_builds_family_request_without_transport(
    model: str, tmp_path: Path
) -> None:
    spec = get_model_spec(model)
    camera_jpeg: dict[str, str] = {}
    for index, role in enumerate(spec.camera_roles):
        name = f"{index}.jpg"
        (tmp_path / name).write_bytes(b"jpeg")
        camera_jpeg[role] = name
    document = {
        "body_joint_position_rad": [0.0] * 29,
        "dex1_opening_fraction": [0.25, 0.75],
        "eef_xyz_euler": [0.0] * 12,
        "camera_jpeg": camera_jpeg,
    }
    (tmp_path / "observation.json").write_text(json.dumps(document))
    observation = load_bundle(tmp_path, spec)
    state = adapter_for(spec).model_state(observation)
    request = _request(spec, observation, state)
    assert request["type"] == "predict"
    assert request["request_id"] == 1
    assert "cameras" in request or "camera_history" in request


def test_offline_bundle_rejects_camera_path_escape(tmp_path: Path) -> None:
    spec = get_model_spec("flip_table_diffusion_chunk_relative_v2")
    document = {
        "body_joint_position_rad": [0.0] * 29,
        "dex1_opening_fraction": [0.25, 0.75],
        "eef_xyz_euler": [0.0] * 12,
        "camera_jpeg": {
            "head_left": "../outside.jpg",
            "left_wrist": "left.jpg",
            "right_wrist": "right.jpg",
        },
    }
    (tmp_path / "observation.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="escapes bundle"):
        load_bundle(tmp_path, spec)
