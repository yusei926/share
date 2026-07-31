"""Pinned GR00T N1.7 full-body G1 checkpoint contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

BASE_MODEL_REPO_ID = "nvidia/GR00T-N1.7-3B"
BASE_MODEL_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
DATASET_REPO_ID = "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2"
DATASET_REVISION = "0dc47877dfb2efbea796a059c81290c649bc773c"
EMBODIMENT_TAG = "real_g1_relative_eef_relative_joints"
EMBODIMENT_ID = 25
LOGICAL_STATE_DIM = 49
LOGICAL_ACTION_DIM = 53
PACKED_STATE_DIM = 132
PACKED_ACTION_DIM = 132
ACTION_HORIZON = 40
VALID_ACTION_DIM = 46
VIDEO_DELTA_INDICES = [-20, 0]
STATE_DELTA_INDICES = [0]
ACTION_DELTA_INDICES = list(range(ACTION_HORIZON))
POLICY_VIDEO_KEYS = ["head_left", "left_wrist", "right_wrist"]
SIM_VALIDATION_DR_PROFILE = "validation_v1"
SIM_FIXED_DR_PROFILE = "nominal_v1"
SIM_UNSEEN_DR_PROFILE = "held_out_v1"
DEX1_SYNERGY_SHA256 = (
    "16c5c500607441255d447d7c15e948b84fada96d41034f01fff2be1d951d465d"
)
EXPECTED_TUNING_SCOPE = {
    "tune_llm": False,
    "tune_visual": False,
    "tune_projector": True,
    "tune_diffusion_model": True,
    "tune_vlln": True,
    "tune_top_llm_layers": 0,
}
EEF_FK_POSITION_P95_M_MAX = 0.08
EEF_FK_ROTATION_P95_RAD_MAX = 0.2
EEF_FK_SWAPPED_SCORE_RATIO_MIN = 4.0
EEF_FK_TOOL_TRANSFORMS = {
    side: {
        "parent_frame": f"{side}_wrist_yaw_link",
        "translation_m": [0.05, 0.0, 0.0],
        "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    for side in ("left", "right")
}

EXPECTED_SHA256 = {
    "config.json": "54c0367060cd310d0b3343fe72a589860a8b6e8173810164a4ffd6253f52e689",
    "processor_config.json": "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be",
    "statistics.json": "c97b1b07a82a8a8858771d56d278732d97dd8eae506032c146c77eb53828afc9",
    "embodiment_id.json": "abc4a749389837416a102d9455a8c089336f2e417afb18565703b9944974bc35",
}

STATE_MODALITY_KEYS = [
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
]
ACTION_MODALITY_KEYS = [
    *STATE_MODALITY_KEYS,
    "base_height_command",
    "navigate_command",
]
RELATIVE_ACTION_EXCLUSIONS = {"hand", "waist", "base_height", "navigate"}
TEMPORAL_SETTINGS = {
    (temporal_lambda, execution_steps)
    for temporal_lambda in ("none", "-0.25", "-0.1", "0")
    for execution_steps in (5, 10, 20)
}
TEMPORAL_PREFERENCE = [
    ("-0.1", 10),
    ("-0.1", 5),
    ("-0.1", 20),
    ("-0.25", 10),
    ("-0.25", 5),
    ("-0.25", 20),
    ("0", 10),
    ("0", 5),
    ("0", 20),
    ("none", 10),
    ("none", 5),
    ("none", 20),
]


def expected_sim_candidate_selection(comparison: dict[str, Any]) -> str:
    """Recompute the deterministic baseline/auxiliary simulator ranking."""

    candidates = comparison.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != {
        "baseline",
        "auxiliary_progress",
    }:
        raise ValueError("simulator comparison has an invalid candidate set")

    def selection_key(name: str) -> tuple[float, ...]:
        candidate = candidates[name]
        test_count = int(candidate.get("test_count", -1))
        success_count = int(candidate.get("success_count", -1))
        if test_count <= 0 or not 0 <= success_count <= test_count:
            raise ValueError(f"invalid simulator result for {name}")
        success_rate = success_count / test_count
        recorded_success_rate = float(candidate.get("success_rate", math.nan))
        trace = candidate.get("trace")
        if not isinstance(trace, dict):
            raise ValueError(f"simulator trace metrics are missing for {name}")
        jerk = float(trace.get("target_jerk_rms", math.nan))
        acceleration = float(trace.get("target_acceleration_rms", math.nan))
        tracking_value = trace.get("tracking_rmse")
        tracking = math.inf if tracking_value is None else float(tracking_value)
        offline = float(candidate.get("offline_validation_score", math.nan))
        if (
            not math.isfinite(recorded_success_rate)
            or abs(recorded_success_rate - success_rate) > 1.0e-12
            or not math.isfinite(jerk)
            or jerk < 0.0
            or not math.isfinite(acceleration)
            or acceleration < 0.0
            or tracking < 0.0
            or (
                tracking_value is not None
                and not math.isfinite(tracking)
            )
            or not math.isfinite(offline)
            or offline < 0.0
        ):
            raise ValueError(f"invalid simulator ranking metrics for {name}")
        return (
            -success_rate,
            jerk,
            acceleration,
            tracking,
            offline,
            0.0 if name == "baseline" else 1.0,
        )

    return min(candidates, key=selection_key)


def validate_temporal_selection_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate and recompute the 12-way temporal-ensemble selection."""

    if report.get("schema_version") != "team_ramen_groot_n17_temporal_sweep/v1":
        raise ValueError("unexpected temporal-selection schema")
    scripted = report.get("scripted_controller_tracking")
    if not isinstance(scripted, dict):
        raise ValueError("scripted-controller tracking evidence is missing")
    arm_rmse = float(scripted.get("arm_rmse_rad", math.nan))
    arm_p95 = float(scripted.get("arm_p95_abs_error_rad", math.nan))
    arm_range = float(scripted.get("actual_arm_range_rad", math.nan))
    if (
        scripted.get("passed") is not True
        or not all(math.isfinite(value) for value in (arm_rmse, arm_p95, arm_range))
        or arm_rmse < 0.0
        or arm_rmse > 0.08
        or arm_p95 < 0.0
        or arm_p95 > 0.16
        or arm_range < 0.05
    ):
        raise ValueError("scripted-controller tracking gate did not pass")

    candidates = report.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 12:
        raise ValueError("temporal selection must contain exactly 12 candidates")
    candidate_by_setting: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("temporal candidate must be a JSON object")
        setting = (
            str(candidate.get("temporal_lambda")),
            int(candidate.get("execution_steps", -1)),
        )
        if setting not in TEMPORAL_SETTINGS or setting in candidate_by_setting:
            raise ValueError(f"invalid or duplicate temporal setting: {setting}")
        test_count = int(candidate.get("test_count", -1))
        success_count = int(candidate.get("success_count", -1))
        success_rate = success_count / test_count if test_count > 0 else math.nan
        recorded_success_rate = float(candidate.get("success_rate", math.nan))
        trace = candidate.get("trace")
        if not isinstance(trace, dict):
            raise ValueError(f"temporal trace is missing for {setting}")
        jerk = float(trace.get("target_jerk_rms", math.nan))
        acceleration = float(trace.get("target_acceleration_rms", math.nan))
        tracking_value = trace.get("tracking_rmse")
        tracking = math.inf if tracking_value is None else float(tracking_value)
        if (
            test_count != 5
            or not 0 <= success_count <= test_count
            or not math.isfinite(recorded_success_rate)
            or abs(recorded_success_rate - success_rate) > 1.0e-12
            or int(candidate.get("seed", -1)) != 92001
            or int(candidate.get("policy_inference_seed", -1)) != 92001
            or candidate.get("episode_inference_seeds")
            != [92001 + index for index in range(5)]
            or candidate.get("episode_ids")
            != [f"92001:{index}" for index in range(5)]
            or candidate.get("mode") != "randomized_validation"
            or candidate.get("domain_randomization_profile")
            != SIM_VALIDATION_DR_PROFILE
            or candidate.get("runtime_evaluation_mode") != "randomized"
            or not math.isfinite(jerk)
            or jerk < 0.0
            or not math.isfinite(acceleration)
            or acceleration < 0.0
            or tracking < 0.0
            or (
                tracking_value is not None
                and not math.isfinite(tracking)
            )
        ):
            raise ValueError(f"invalid temporal candidate evidence for {setting}")
        candidate_by_setting[setting] = candidate
    if set(candidate_by_setting) != TEMPORAL_SETTINGS:
        raise ValueError("temporal sweep did not cover the required settings")

    def selection_key(item: tuple[tuple[str, int], dict[str, Any]]) -> tuple[float, ...]:
        setting, candidate = item
        trace = candidate["trace"]
        tracking_value = trace["tracking_rmse"]
        return (
            -float(candidate["success_rate"]),
            float(trace["target_jerk_rms"]),
            float(trace["target_acceleration_rms"]),
            math.inf if tracking_value is None else float(tracking_value),
            float(TEMPORAL_PREFERENCE.index(setting)),
        )

    selected_setting, selected_candidate = min(
        candidate_by_setting.items(),
        key=selection_key,
    )
    selected = report.get("selected")
    if (
        not isinstance(selected, dict)
        or str(selected.get("temporal_lambda")) != selected_setting[0]
        or int(selected.get("execution_steps", -1)) != selected_setting[1]
        or abs(
            float(selected.get("success_rate", math.nan))
            - float(selected_candidate["success_rate"])
        )
        > 1.0e-12
    ):
        raise ValueError(
            "selected temporal setting is inconsistent with sweep metrics"
        )
    return {
        "temporal_lambda": selected_setting[0],
        "execution_steps": selected_setting[1],
        "success_rate": float(selected_candidate["success_rate"]),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_eef_fk_release_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Validate fixed EEF/FK gates without treating IK residual as equality."""

    if (
        audit.get("source_repo_id") != DATASET_REPO_ID
        or audit.get("source_revision") != DATASET_REVISION
        or audit.get("configured_eef_order") != ["left", "right"]
        or audit.get("eef_pose_format") != "xyz_euler_xyz_rad"
        or audit.get("eef_reference_frame") != "robot_root"
        or audit.get("pass") is not True
        or audit.get("action_fk_residual_pass") is not True
        or audit.get("frame_assignment_pass") is not True
    ):
        raise ValueError("EEF-FK audit violates its source coordinate contract")
    thresholds = audit.get("thresholds") or {}
    expected_thresholds = {
        "position_p95_m_max": EEF_FK_POSITION_P95_M_MAX,
        "rotation_p95_rad_max": EEF_FK_ROTATION_P95_RAD_MAX,
        "swapped_score_ratio_min": EEF_FK_SWAPPED_SCORE_RATIO_MIN,
    }
    if any(
        float(thresholds.get(name, math.nan)) != value
        for name, value in expected_thresholds.items()
    ):
        raise ValueError("EEF-FK release thresholds changed")
    if float(audit.get("swapped_to_configured_score_ratio", 0.0)) < (
        EEF_FK_SWAPPED_SCORE_RATIO_MIN
    ):
        raise ValueError("EEF left/right assignment is ambiguous")
    tool_transforms = audit.get("tool_transforms") or {}
    for side, expected in EEF_FK_TOOL_TRANSFORMS.items():
        actual = tool_transforms.get(side) or {}
        if (
            actual.get("parent_frame") != expected["parent_frame"]
            or actual.get("translation_m") != expected["translation_m"]
            or actual.get("quaternion_xyzw")
            != expected["quaternion_xyzw"]
        ):
            raise ValueError(f"EEF-FK tool transform changed for {side}")

    validation_metrics = audit.get("validation_metrics") or {}
    action_metrics = validation_metrics.get("action") or {}
    pooled_metrics: dict[str, dict[str, float]] = {}
    for side in ("left", "right"):
        side_metrics = action_metrics.get(side) or {}
        position_p95 = float(
            (side_metrics.get("position_error_m") or {}).get(
                "p95",
                math.nan,
            )
        )
        rotation_p95 = float(
            (side_metrics.get("rotation_error_rad") or {}).get(
                "p95",
                math.nan,
            )
        )
        if (
            not math.isfinite(position_p95)
            or position_p95 > EEF_FK_POSITION_P95_M_MAX
            or not math.isfinite(rotation_p95)
            or rotation_p95 > EEF_FK_ROTATION_P95_RAD_MAX
        ):
            raise ValueError(f"pooled EEF-FK residual gate failed for {side}")
        pooled_metrics[side] = {
            "position_p95_m": position_p95,
            "rotation_p95_rad": rotation_p95,
        }

    per_episode = audit.get("per_episode")
    if (
        not isinstance(per_episode, list)
        or [int(item.get("episode_index", -1)) for item in per_episode]
        != list(range(174))
    ):
        raise ValueError("EEF-FK audit must cover every episode 0..173")
    failed_episodes = [
        int(item["episode_index"])
        for item in per_episode
        if item.get("action_fk_residual_pass") is not True
    ]
    coverage = audit.get("coverage") or {}
    declared_failures = coverage.get(
        "episode_level_diagnostic_threshold_exceedances",
        coverage.get("failed_episode_indices"),
    )
    if (
        int(coverage.get("episode_count", -1)) != 174
        or declared_failures != failed_episodes
    ):
        raise ValueError("EEF-FK per-episode diagnostics are incomplete")
    mimic_gate = audit.get("mimic_source_episode_gate") or {}
    if (
        int(mimic_gate.get("eligible_count", -1))
        + int(mimic_gate.get("rejected_count", -1))
        != 174
        or int(mimic_gate.get("rejected_count", -1))
        != len(failed_episodes)
    ):
        raise ValueError("EEF-FK episode diagnostics disagree with the source gate")
    timing = audit.get("temporal_alignment") or {}
    if (
        timing.get("pass") is not True
        or int(timing.get("selected_offset_frames", -1)) != 0
        or float(timing.get("material_improvement_threshold", math.nan))
        != 0.05
    ):
        raise ValueError("EEF and joint teachers have incompatible timestamps")
    training_contract = audit.get("training_contract") or {}
    if (
        training_contract.get("eef_teacher") != "action.ee_action"
        or training_contract.get("joint_teacher")
        != "action.robot_q_desired"
        or training_contract.get("policy_action_mask_slots") != "0:46"
        or training_contract.get("teacher_pair_status")
        != "compatible_with_expected_ik_realization_residual"
    ):
        raise ValueError("EEF and arm teacher contract changed")
    return {
        "pooled_action_metrics": pooled_metrics,
        "episode_threshold_exceedance_count": len(failed_episodes),
        "episode_threshold_exceedances": failed_episodes,
        "selected_offset_frames": 0,
        "swapped_to_configured_score_ratio": float(
            audit["swapped_to_configured_score_ratio"]
        ),
    }


def valid_sim_candidate_evidence(
    comparison: dict[str, Any],
    *,
    candidate_hashes: dict[str, str],
) -> bool:
    """Validate the complete same-seed baseline/auxiliary simulator comparison."""
    try:
        expected_candidates = {"baseline", "auxiliary_progress"}
        if (
            comparison.get("schema_version")
            != "groot_n17_sim_candidate_comparison_v1"
            or comparison.get("test_split_used") is not False
            or int(comparison.get("seed", -1)) != 95001
            or int(comparison.get("policy_inference_seed", -1)) != 95001
            or comparison.get("episode_ids")
            != [f"95001:{index}" for index in range(5)]
            or comparison.get("offline_validation_episodes") != list(range(139, 156))
            or comparison.get("selection_data")
            != "same_seed_randomized_sim_validation"
            or comparison.get("domain_randomization_profile")
            != SIM_VALIDATION_DR_PROFILE
            or set(candidate_hashes) != expected_candidates
        ):
            return False
        candidates = comparison.get("candidates") or {}
        if set(candidates) != expected_candidates:
            return False
        for name, expected_hash in candidate_hashes.items():
            candidate = candidates[name]
            success_count = int(candidate.get("success_count", -1))
            if (
                candidate.get("model_safetensors_sha256") != expected_hash
                or int(candidate.get("test_count", -1)) != 5
                or not 0 <= success_count <= 5
                or int(candidate.get("seed", -1)) != 95001
                or int(candidate.get("policy_inference_seed", -1)) != 95001
                or candidate.get("episode_inference_seeds")
                != [95001 + index for index in range(5)]
                or candidate.get("episode_ids")
                != [f"95001:{index}" for index in range(5)]
                or candidate.get("mode") != "randomized_validation"
                or candidate.get("domain_randomization_profile")
                != SIM_VALIDATION_DR_PROFILE
                or candidate.get("runtime_evaluation_mode") != "randomized"
                or candidate.get("offline_validation_episodes")
                != list(range(139, 156))
                or candidate.get("progress_enabled")
                is not (name == "auxiliary_progress")
            ):
                return False
        return (
            comparison.get("selected")
            == expected_sim_candidate_selection(comparison)
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_checkpoint_contract(checkpoint_root: str | Path) -> dict[str, Any]:
    """Fail closed when the pinned source files or semantic layout drift."""
    root = Path(checkpoint_root)
    hashes: dict[str, str] = {}
    for filename, expected in EXPECTED_SHA256.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned GR00T contract file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{filename} SHA-256 mismatch: expected {expected}, got {actual}")
        hashes[filename] = actual

    config = _read_json(root / "config.json")
    processor = _read_json(root / "processor_config.json")
    embodiment_ids = _read_json(root / "embodiment_id.json")
    statistics = _read_json(root / "statistics.json")
    modality = processor["processor_kwargs"]["modality_configs"][EMBODIMENT_TAG]

    _expect(config.get("model_type") == "Gr00tN1d7", "checkpoint is not GR00T N1.7")
    _expect(config.get("max_state_dim") == PACKED_STATE_DIM, "packed state must stay 132-D")
    _expect(config.get("max_action_dim") == PACKED_ACTION_DIM, "packed action must stay 132-D")
    _expect(config.get("action_horizon") == ACTION_HORIZON, "action horizon must stay 40")
    _expect(embodiment_ids.get(EMBODIMENT_TAG) == EMBODIMENT_ID, "G1 embodiment id must stay 25")
    _expect(modality["video"]["delta_indices"] == VIDEO_DELTA_INDICES, "video history must be [-20, 0]")
    _expect(modality["video"]["modality_keys"] == ["ego_view"], "official source must have ego_view")
    _expect(modality["state"]["delta_indices"] == STATE_DELTA_INDICES, "state history must stay [0]")
    _expect(modality["state"]["modality_keys"] == STATE_MODALITY_KEYS, "49-D state group order changed")
    _expect(modality["action"]["delta_indices"] == ACTION_DELTA_INDICES, "action horizon layout changed")
    _expect(modality["action"]["modality_keys"] == ACTION_MODALITY_KEYS, "53-D action group order changed")
    _expect(EMBODIMENT_TAG in statistics, "missing G1 statistics")

    return {
        "repo_id": BASE_MODEL_REPO_ID,
        "revision": BASE_MODEL_REVISION,
        "embodiment_tag": EMBODIMENT_TAG,
        "embodiment_id": EMBODIMENT_ID,
        "logical_state_dim": LOGICAL_STATE_DIM,
        "logical_action_dim": LOGICAL_ACTION_DIM,
        "packed_state_dim": PACKED_STATE_DIM,
        "packed_action_dim": PACKED_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "valid_action_dim": VALID_ACTION_DIM,
        "official_video_delta_indices": VIDEO_DELTA_INDICES,
        "policy_video_keys": POLICY_VIDEO_KEYS,
        "sha256": hashes,
    }


def validate_furniture_training_candidate(
    checkpoint_root: str | Path,
    *,
    expected_progress_enabled: bool | None = None,
) -> dict[str, Any]:
    """Validate an H100 candidate before held-out simulator selection."""
    root = Path(checkpoint_root)
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete Furniture-GR00T candidate; missing {missing}"
        )
    config = _read_json(root / "config.json")
    _expect(config.get("type") == "furniture_groot", "candidate type changed")
    _expect(
        config.get("base_model_path") == BASE_MODEL_REPO_ID,
        "candidate base model repository changed",
    )
    revision = config.get("base_model_revision")
    _expect(
        revision == BASE_MODEL_REVISION,
        "candidate base model revision changed",
    )
    _expect(
        all(config.get(key) == value for key, value in EXPECTED_TUNING_SCOPE.items()),
        "candidate tuning scope changed",
    )
    _expect(config.get("max_state_dim") == PACKED_STATE_DIM, "candidate state packing changed")
    _expect(config.get("max_action_dim") == PACKED_ACTION_DIM, "candidate action packing changed")
    _expect(config.get("chunk_size") == ACTION_HORIZON, "candidate horizon changed")
    _expect(config.get("valid_action_dim") == VALID_ACTION_DIM, "candidate valid action mask changed")
    _expect(config.get("use_relative_actions") is True, "candidate must use relative actions")
    _expect(
        set(config.get("relative_exclude_joints") or ())
        == RELATIVE_ACTION_EXCLUSIONS,
        "candidate relative-action exclusions changed",
    )
    _expect(
        config.get("action_decode_transform") is None,
        "candidate contains a simulator-only action transform",
    )
    if expected_progress_enabled is not None:
        _expect(
            config.get("progress_enabled") is expected_progress_enabled,
            "candidate progress-head mode changed",
        )

    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    _expect(
        (inputs.get("observation.state") or {}).get("shape")
        == [LOGICAL_STATE_DIM],
        "candidate state feature must be 49-D",
    )
    camera_keys = {
        str(key)
        for key in inputs
        if str(key).startswith("observation.images.")
    }
    expected_camera_keys = {
        f"observation.images.{name}" for name in POLICY_VIDEO_KEYS
    }
    _expect(camera_keys == expected_camera_keys, "candidate policy cameras changed")
    for key in expected_camera_keys:
        _expect(
            (inputs.get(key) or {}).get("shape") == [3, 480, 640],
            f"candidate camera shape changed: {key}",
        )
    _expect(
        (outputs.get("action") or {}).get("shape") == [LOGICAL_ACTION_DIM],
        "candidate logical action must remain 53-D",
    )
    return {
        "model_safetensors_sha256": sha256_file(root / "model.safetensors"),
        "config_sha256": sha256_file(root / "config.json"),
        "progress_enabled": bool(config.get("progress_enabled")),
        "logical_state_dim": LOGICAL_STATE_DIM,
        "logical_action_dim": LOGICAL_ACTION_DIM,
        "packed_action_dim": PACKED_ACTION_DIM,
        "valid_action_dim": VALID_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "policy_cameras": list(POLICY_VIDEO_KEYS),
    }


def validate_finalized_furniture_checkpoint(
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    """Verify immutable training provenance and executable adapter artifacts."""
    root = Path(checkpoint_root)
    required = (
        "config.json",
        "model.safetensors",
        "training_manifest.json",
        "training_run_record.json",
        "source_snapshot_manifest.json",
        "dex1_g1_synergy.json",
        "eef_fk_audit.json",
        "progress_manifest.json",
        "progress.jsonl",
        "visual_rotation_manifest.json",
        "visual_rotation.jsonl",
        "orientation_contact_sheet.jpg",
        "orientation_contact_sheet.approved",
        "sim_candidate_selection.json",
        "sim_evaluation_manifest.json",
        "sim_release_evaluation.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"checkpoint is not finalized; missing release artifacts: {missing}"
        )

    config = _read_json(root / "config.json")
    manifest = _read_json(root / "training_manifest.json")
    run_record = _read_json(root / "training_run_record.json")
    source_snapshot = _read_json(root / "source_snapshot_manifest.json")
    eef_fk_audit = _read_json(root / "eef_fk_audit.json")
    eef_fk_validation = validate_eef_fk_release_audit(eef_fk_audit)
    progress_manifest = _read_json(root / "progress_manifest.json")
    visual_manifest = _read_json(root / "visual_rotation_manifest.json")
    sim_comparison = _read_json(root / "sim_candidate_selection.json")
    sim_bundle_manifest = _read_json(root / "sim_evaluation_manifest.json")
    sim_release = _read_json(root / "sim_release_evaluation.json")
    sim_bundle = root / "sim_evaluation"
    temporal_selection_path = (
        sim_bundle / "release" / "temporal_selection.json"
    )
    temporal_selection = _read_json(temporal_selection_path)
    validated_temporal = validate_temporal_selection_report(
        temporal_selection
    )
    fixed_scene = sim_release.get("fixed_scene") or {}
    unseen_dr = sim_release.get("unseen_dr") or {}
    selected_temporal = sim_release.get("selected_temporal_setting") or {}
    selected_temporal_lambda = str(selected_temporal.get("temporal_lambda"))
    selected_execution_steps = int(selected_temporal.get("execution_steps", -1))
    temporal_setting_evidence = (
        selected_temporal_lambda in {"none", "-0.25", "-0.1", "0"}
        and selected_execution_steps in {5, 10, 20}
        and validated_temporal["temporal_lambda"]
        == selected_temporal_lambda
        and validated_temporal["execution_steps"]
        == selected_execution_steps
        and sim_release.get("temporal_validation") == temporal_selection
        and sim_release.get("temporal_validation_sha256")
        == sha256_file(temporal_selection_path)
        and sim_release.get("scripted_controller_tracking")
        == temporal_selection.get("scripted_controller_tracking")
        and str(fixed_scene.get("temporal_lambda")) == selected_temporal_lambda
        and int(fixed_scene.get("execution_steps", -1))
        == selected_execution_steps
        and str(unseen_dr.get("temporal_lambda")) == selected_temporal_lambda
        and int(unseen_dr.get("execution_steps", -1))
        == selected_execution_steps
    )
    fixed_seed_evidence = (
        fixed_scene.get("seed") == 93001
        and fixed_scene.get("policy_inference_seed") == 93001
        and fixed_scene.get("episode_inference_seeds") == [93001, 93002, 93003]
        and fixed_scene.get("episode_ids")
        == [f"93001:{index}" for index in range(3)]
        and fixed_scene.get("mode") == "nominal"
        and fixed_scene.get("domain_randomization_profile")
        == SIM_FIXED_DR_PROFILE
        and fixed_scene.get("runtime_evaluation_mode") == "nominal"
    )
    unseen_seed_evidence = (
        unseen_dr.get("seed") == 94001
        and unseen_dr.get("policy_inference_seed") == 94001
        and unseen_dr.get("episode_inference_seeds")
        == [94001 + index for index in range(50)]
        and unseen_dr.get("episode_ids")
        == [f"94001:{index}" for index in range(50)]
        and unseen_dr.get("mode") == "unseen_dr"
        and unseen_dr.get("domain_randomization_profile")
        == SIM_UNSEEN_DR_PROFILE
        and unseen_dr.get("runtime_evaluation_mode") == "unseen_dr"
    )
    model_hash = sha256_file(root / "model.safetensors")
    config_hash = sha256_file(root / "config.json")
    synergy_hash = sha256_file(root / "dex1_g1_synergy.json")
    eef_fk_audit_hash = sha256_file(root / "eef_fk_audit.json")
    source_snapshot_hash = sha256_file(root / "source_snapshot_manifest.json")
    progress_manifest_hash = sha256_file(root / "progress_manifest.json")
    visual_manifest_hash = sha256_file(root / "visual_rotation_manifest.json")
    selection = manifest.get("candidate_selection") or {}
    candidate_hashes = selection.get("candidate_hashes") or {}
    sim_candidate_evidence = (
        set(candidate_hashes) == {"baseline", "auxiliary_progress"}
        and valid_sim_candidate_evidence(
            sim_comparison,
            candidate_hashes={
                str(name): str(value) for name, value in candidate_hashes.items()
            },
        )
        and sim_comparison.get("selected") == selection.get("selected")
        and candidate_hashes.get(selection.get("selected")) == model_hash
        and config.get("progress_enabled")
        is (selection.get("selected") == "auxiliary_progress")
    )

    _expect(config.get("type") == "furniture_groot", "checkpoint type must be furniture_groot")
    _expect(
        config.get("base_model_path") == BASE_MODEL_REPO_ID,
        "checkpoint base model repository changed",
    )
    _expect(
        config.get("base_model_revision") == BASE_MODEL_REVISION,
        "checkpoint base model revision changed",
    )
    _expect(
        all(config.get(key) == value for key, value in EXPECTED_TUNING_SCOPE.items()),
        "checkpoint tuning scope changed",
    )
    checkpoint_record = manifest.get("checkpoint") or {}
    _expect(
        checkpoint_record.get("model_safetensors_sha256") == model_hash,
        "model.safetensors hash differs from the finalized manifest",
    )
    _expect(
        checkpoint_record.get("config_sha256") == config_hash,
        "config.json hash differs from the finalized manifest",
    )
    dataset = manifest.get("dataset") or {}
    _expect(dataset.get("repo_id") == DATASET_REPO_ID, "training dataset repository changed")
    _expect(dataset.get("revision") == DATASET_REVISION, "training dataset revision changed")
    _expect(
        dataset.get("task_instruction") == "flip table"
        and dataset.get("task_index") == 0,
        "training task instruction changed",
    )
    _expect(
        dataset.get("counts") == {"train": 139, "validation": 17, "test": 18},
        "training dataset split counts changed",
    )
    contract = manifest.get("contract") or {}
    expected_contract = {
        "logical_state_dim": LOGICAL_STATE_DIM,
        "logical_action_dim": LOGICAL_ACTION_DIM,
        "packed_state_dim": PACKED_STATE_DIM,
        "packed_action_dim": PACKED_ACTION_DIM,
        "valid_action_dim": VALID_ACTION_DIM,
        "action_horizon": ACTION_HORIZON,
        "policy_cameras": POLICY_VIDEO_KEYS,
        "head_right_used": False,
        "progress_in_action": False,
        "task_instruction": "flip table",
    }
    _expect(
        all(contract.get(key) == value for key, value in expected_contract.items()),
        "finalized training contract changed",
    )
    expected_progress_shape = [ACTION_HORIZON, 1] if config.get("progress_enabled") else None
    _expect(
        contract.get("progress_head_shape") == expected_progress_shape,
        "auxiliary progress head shape changed",
    )
    base_contract = run_record.get("contract") or {}
    _expect(
        base_contract.get("repo_id") == BASE_MODEL_REPO_ID
        and base_contract.get("revision") == BASE_MODEL_REVISION,
        "training record does not identify the pinned GR00T N1.7 source",
    )
    _expect(
        base_contract.get("sha256") == EXPECTED_SHA256,
        "pinned GR00T N1.7 source-file hashes changed",
    )
    _expect(
        (run_record.get("training_scope") or {}).get("config_flags")
        == EXPECTED_TUNING_SCOPE,
        "training record does not prove the reviewed trainable scope",
    )
    source_record = run_record.get("source") or {}
    source_files = source_snapshot.get("files") or {}
    source_bundle_name = source_snapshot.get("bundle_root")
    source_bundle = root / str(source_bundle_name)
    bundle_files = (
        {
            path.relative_to(source_bundle).as_posix()
            for path in source_bundle.rglob("*")
            if path.is_file()
        }
        if source_bundle.is_dir()
        else set()
    )
    _expect(
        source_snapshot.get("schema_version") == "groot_n17_source_snapshot_v2"
        and source_snapshot.get("scope")
        == "release_time_training_inference_and_evaluation_source"
        and source_snapshot.get("captured_at_utc")
        == source_record.get("snapshot_captured_at_utc")
        and source_snapshot.get("scope") == source_record.get("snapshot_scope")
        and source_bundle_name == "source_snapshot"
        and source_snapshot.get("git_head") == source_record.get("git_head")
        and source_snapshot.get("file_count") == source_record.get(
            "snapshot_file_count"
        )
        and source_snapshot.get("file_count") == len(source_files)
        and bundle_files == set(source_files)
        and source_record.get("snapshot_manifest")
        == "source_snapshot_manifest.json"
        and source_record.get("snapshot_manifest_sha256")
        == source_snapshot_hash,
        "training source snapshot is incomplete or differs from the training record",
    )
    for relative, expected_hash in source_files.items():
        relative_path = Path(relative)
        _expect(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            "training source snapshot contains an unsafe path",
        )
        _expect(
            sha256_file(source_bundle / relative_path) == expected_hash,
            f"training source snapshot hash changed: {relative}",
        )
    dex1_record = run_record.get("dex1_adapter") or {}
    _expect(synergy_hash == DEX1_SYNERGY_SHA256, "Dex1 synergy artifact hash changed")
    _expect(
        dex1_record.get("sha256") == DEX1_SYNERGY_SHA256,
        "training record Dex1 synergy hash changed",
    )
    eef_fk_record = run_record.get("eef_fk_audit") or {}
    _expect(
        eef_fk_record.get("sha256") == eef_fk_audit_hash,
        "EEF-FK audit hash differs from the training record",
    )
    _expect(
        eef_fk_record.get("source_repo_id") == DATASET_REPO_ID
        and eef_fk_record.get("source_revision") == DATASET_REVISION
        and eef_fk_record.get("episode_count") == 174
        and eef_fk_record.get("action_fk_residual_pass") is True
        and eef_fk_record.get("frame_assignment_pass") is True
        and eef_fk_record.get("selected_offset_frames") == 0
        and eef_fk_record.get("teacher_pair_status")
        == "compatible_with_expected_ik_realization_residual"
        and eef_fk_record.get("fixed_release_gate")
        == eef_fk_validation,
        "EEF and joint teachers lack a passing coordinate/time audit",
    )
    sidecar_record = run_record.get("sidecars") or {}
    sidecar_artifacts = sidecar_record.get("artifacts") or {}
    expected_sidecar_artifacts = {
        name: sha256_file(root / name)
        for name in (
            "progress.jsonl",
            "visual_rotation.jsonl",
            "orientation_contact_sheet.jpg",
            "orientation_contact_sheet.approved",
        )
    }
    contact_review = sidecar_record.get("contact_sheet_review") or {}
    contact_sheet_hash = expected_sidecar_artifacts[
        "orientation_contact_sheet.jpg"
    ]
    _expect(
        sidecar_record.get("progress_manifest_sha256")
        == progress_manifest_hash
        and sidecar_record.get("visual_rotation_manifest_sha256")
        == visual_manifest_hash
        and sidecar_artifacts == expected_sidecar_artifacts
        and progress_manifest.get("schema_version")
        == "flip_table_progress_sidecar_manifest_v1"
        and progress_manifest.get("dataset_repo_id") == DATASET_REPO_ID
        and progress_manifest.get("dataset_revision") == DATASET_REVISION
        and progress_manifest.get("annotation_file") == "progress.jsonl"
        and progress_manifest.get("annotation_sha256")
        == expected_sidecar_artifacts["progress.jsonl"]
        and (progress_manifest.get("summary") or {}).get("episode_count")
        == 174
        and visual_manifest.get("schema_version")
        == "flip_table_visual_rotation_manifest_v1"
        and visual_manifest.get("dataset_repo_id") == DATASET_REPO_ID
        and visual_manifest.get("dataset_revision") == DATASET_REVISION
        and visual_manifest.get("episode_count") == 174
        and visual_manifest.get("sidecar_sha256")
        == expected_sidecar_artifacts["visual_rotation.jsonl"]
        and visual_manifest.get("contact_sheet")
        == "orientation_contact_sheet.jpg"
        and visual_manifest.get("contact_sheet_sha256") == contact_sheet_hash
        and visual_manifest.get("contact_sheet_human_review_required") is True
        and visual_manifest.get("policy_input") is False
        and contact_review.get("required") is True
        and contact_review.get("approved_sha256") == contact_sheet_hash
        and (root / "orientation_contact_sheet.approved")
        .read_text(encoding="utf-8")
        .strip()
        == contact_sheet_hash,
        "progress/visual sidecars or contact-sheet approval changed",
    )
    _expect(
        selection.get("selection_data")
        == "offline_validation_plus_same_seed_sim_validation",
        "candidate selection did not use the declared offline and simulator validation gates",
    )
    simulation_record = run_record.get("simulation_evaluation") or {}
    recorded_bundle_files = sim_bundle_manifest.get("files") or {}
    actual_bundle_files = (
        {
            path.relative_to(sim_bundle).as_posix()
            for path in sim_bundle.rglob("*")
            if path.is_file()
        }
        if sim_bundle.is_dir()
        else set()
    )
    _expect(
        sim_candidate_evidence
        and sim_release.get("schema_version")
        == "team_ramen_groot_n17_release_evaluation/v1"
        and sim_release.get("candidate_name") == selection.get("selected")
        and sim_release.get("model_safetensors_sha256") == model_hash
        and int(fixed_scene.get("test_count", -1)) == 3
        and int(fixed_scene.get("success_count", -1)) == 3
        and fixed_seed_evidence
        and int(unseen_dr.get("test_count", -1)) == 50
        and int(unseen_dr.get("success_count", -1)) >= 40
        and unseen_seed_evidence
        and temporal_setting_evidence
        and (sim_release.get("release_goal") or {}).get("unseen_dr_passed")
        is True
        and sim_release.get("claim_scope")
        == "simulator evaluation only; no Sim-to-Real success claim"
        and simulation_record.get("selected_candidate")
        == selection.get("selected")
        and simulation_record.get("fixed_scene_dr_profile")
        == SIM_FIXED_DR_PROFILE
        and simulation_record.get("unseen_dr_profile")
        == SIM_UNSEEN_DR_PROFILE
        and simulation_record.get("candidate_comparison_sha256")
        == sha256_file(root / "sim_candidate_selection.json")
        and simulation_record.get("release_evaluation_sha256")
        == sha256_file(root / "sim_release_evaluation.json")
        and simulation_record.get("temporal_validation_sha256")
        == sha256_file(temporal_selection_path)
        and simulation_record.get("bundle_manifest_sha256")
        == sha256_file(root / "sim_evaluation_manifest.json")
        and sim_bundle_manifest.get("schema_version")
        == "groot_n17_sim_evaluation_bundle_v1"
        and sim_bundle_manifest.get("root") == "sim_evaluation"
        and sim_bundle_manifest.get("file_count") == len(recorded_bundle_files)
        and sim_bundle_manifest.get("file_count")
        == simulation_record.get("bundle_file_count")
        and sim_bundle_manifest.get("total_bytes")
        == simulation_record.get("bundle_total_bytes")
        and actual_bundle_files == set(recorded_bundle_files),
        "finalized checkpoint lacks passing same-seed simulator evidence",
    )
    for relative, metadata in recorded_bundle_files.items():
        relative_path = Path(relative)
        _expect(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            "simulator evaluation bundle contains an unsafe path",
        )
        path = sim_bundle / relative_path
        _expect(
            path.stat().st_size == int(metadata["bytes"])
            and sha256_file(path) == metadata["sha256"],
            f"simulator evaluation artifact hash changed: {relative}",
        )
    _expect(bool(manifest.get("wandb_url")), "finalized manifest is missing the W&B run")
    return {
        "model_safetensors_sha256": model_hash,
        "config_sha256": config_hash,
        "dex1_synergy_sha256": synergy_hash,
        "eef_fk_audit_sha256": eef_fk_audit_hash,
        "source_snapshot_manifest_sha256": source_snapshot_hash,
        "dataset_revision": DATASET_REVISION,
        "base_model_revision": BASE_MODEL_REVISION,
        "temporal_lambda_label": selected_temporal_lambda,
        "temporal_lambda": (
            None
            if selected_temporal_lambda == "none"
            else float(selected_temporal_lambda)
        ),
        "execution_steps": selected_execution_steps,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
