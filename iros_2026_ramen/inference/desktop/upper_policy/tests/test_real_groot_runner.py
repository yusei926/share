from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from inference.desktop.upper_policy.groot_pick_leg_contract import (
    CAMERA_KEYS,
    CAMERA_ROLE_TO_KEY,
    DEX1_DATASET_OPEN_VALUE,
    MODEL_ACTION_HORIZON,
    MODEL_REVISION,
    REAL_ROOT_PROXY_XYZ_WXYZ,
    TASK_TEXT,
    camera_payloads,
    compose_model_state,
    extract_executable_action,
    validate_checkpoint_metadata,
)
from inference.desktop.upper_policy.run_pick_leg_groot import (
    validate_action_chunk,
    validate_state_distribution,
)
from model.subtask_policy_training.deployment.real_groot_n17_worker import (
    _decode_rgb,
)


def _write_metadata(root: Path) -> None:
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "Gr00tN1d7",
                "action_horizon": 40,
                "model_name": "nvidia/Cosmos-Reason2-2B",
            }
        )
    )
    (root / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_kwargs": {
                    "use_relative_action": True,
                    "modality_configs": {
                        "new_embodiment": {
                            "video": {
                                "delta_indices": [0],
                                "modality_keys": [
                                    "cam_0",
                                    "cam_1",
                                    "cam_2",
                                    "cam_3",
                                ],
                            },
                            "state": {
                                "delta_indices": [0],
                                "modality_keys": ["robot_q", "hand"],
                            },
                            "action": {
                                "delta_indices": list(range(16)),
                                "modality_keys": ["robot_q", "hand"],
                                "action_configs": [
                                    {
                                        "rep": "ABSOLUTE",
                                        "type": "NON_EEF",
                                        "format": "DEFAULT",
                                        "state_key": None,
                                    },
                                    {
                                        "rep": "ABSOLUTE",
                                        "type": "NON_EEF",
                                        "format": "DEFAULT",
                                        "state_key": None,
                                    },
                                ],
                            },
                            "language": {
                                "delta_indices": [0],
                                "modality_keys": [
                                    "annotation.human.task_description"
                                ],
                            },
                        }
                    },
                }
            }
        )
    )
    stat = lambda width: {  # noqa: E731
        "mean": [0.0] * width,
        "std": [1.0] * width,
        "min": [-1.0] * width,
        "max": [1.0] * width,
        "q01": [-1.0] * width,
        "q99": [1.0] * width,
    }
    (root / "statistics.json").write_text(
        json.dumps(
            {
                "new_embodiment": {
                    "state": {"robot_q": stat(36), "hand": stat(2)},
                    "action": {"robot_q": stat(36), "hand": stat(2)},
                    "relative_action": {},
                }
            }
        )
    )
    (root / "embodiment_id.json").write_text(
        json.dumps({"new_embodiment": 10})
    )


def test_checkpoint_contract_is_exact(tmp_path: Path) -> None:
    _write_metadata(tmp_path)
    contract = validate_checkpoint_metadata(tmp_path)
    assert contract["model_revision"] == MODEL_REVISION
    assert contract["task"] == TASK_TEXT
    assert contract["state_dim"] == 38
    assert contract["decoded_action_dim"] == 38
    assert contract["executable_action_dim"] == 16
    assert contract["lower_body_command_dimensions"] == 0
    assert contract["camera_keys"] == list(CAMERA_KEYS)


def test_checkpoint_rejects_relative_actions(tmp_path: Path) -> None:
    _write_metadata(tmp_path)
    path = tmp_path / "processor_config.json"
    config = json.loads(path.read_text())
    config["processor_kwargs"]["modality_configs"]["new_embodiment"]["action"][
        "action_configs"
    ][0]["rep"] = "RELATIVE"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="absolute"):
        validate_checkpoint_metadata(tmp_path)


def test_model_state_preserves_full_unitree_body_order() -> None:
    body = np.linspace(-1.4, 1.4, 29)
    state = compose_model_state(body, [0.25, 0.75])
    np.testing.assert_allclose(state[:7], REAL_ROOT_PROXY_XYZ_WXYZ)
    np.testing.assert_allclose(state[7:36], body)
    np.testing.assert_allclose(
        state[36:], [0.25 * DEX1_DATASET_OPEN_VALUE, 0.75 * DEX1_DATASET_OPEN_VALUE]
    )


def test_action_mapping_drops_root_legs_and_waist() -> None:
    actions = np.zeros((MODEL_ACTION_HORIZON, 38), dtype=np.float64)
    actions[:, :22] = np.arange(22)[None, :] + 1000.0
    expected_arms = np.arange(MODEL_ACTION_HORIZON * 14).reshape(
        MODEL_ACTION_HORIZON, 14
    )
    actions[:, 22:36] = expected_arms
    actions[:, 36:] = [1.25, 3.5]
    executable = extract_executable_action(actions)
    np.testing.assert_allclose(executable[:, :14], expected_arms)
    np.testing.assert_allclose(
        executable[:, 14:], np.tile([1.25, 3.5], (MODEL_ACTION_HORIZON, 1))
    )
    # Changing every non-executable dimension cannot affect hardware output.
    actions[:, :22] *= -999.0
    np.testing.assert_allclose(extract_executable_action(actions), executable)


def test_camera_mapping_matches_training_order() -> None:
    source = {
        "head_left": b"head-left",
        "head_right": b"head-right",
        "left_wrist": b"left-wrist",
        "right_wrist": b"right-wrist",
    }
    mapped = camera_payloads(source)
    assert tuple(mapped) == CAMERA_KEYS
    for role, key in CAMERA_ROLE_TO_KEY.items():
        assert mapped[key] == source[role]


def test_worker_decodes_policy_images_as_chw() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    decoded = _decode_rgb(encoded.tobytes(), "head_left")
    assert tuple(decoded.shape) == (3, 480, 640)
    # BGR red becomes RGB channel 0, not a silent BGR input.
    assert float(decoded[0].float().mean()) > 250.0
    assert float(decoded[2].float().mean()) < 5.0


class _Safety:
    arm_position_lower_rad = [-10.0] * 14
    arm_position_upper_rad = [10.0] * 14


class _Config:
    safety = _Safety()


def test_executable_validation_never_accepts_38d_hardware_action() -> None:
    with pytest.raises(ValueError, match=r"\[16,16\]"):
        validate_action_chunk(
            np.zeros((16, 38)),
            measured_arm=np.zeros(14),
            config=_Config(),
            initial_delta_limit_rad=0.2,
            step_delta_limit_rad=0.2,
        )


def test_executable_validation_checks_only_absolute_arms_and_hands() -> None:
    chunk = np.zeros((16, 16), dtype=np.float64)
    chunk[:, 14:] = 2.25
    result = validate_action_chunk(
        chunk,
        measured_arm=np.zeros(14),
        config=_Config(),
        initial_delta_limit_rad=0.2,
        step_delta_limit_rad=0.2,
    )
    assert result["initial_arm_delta_max_rad"] == 0.0
    assert result["dex1_min_fraction"] == pytest.approx(0.5)


def test_read_only_validation_reports_initial_delta_without_weakening_actuation() -> None:
    chunk = np.zeros((16, 16), dtype=np.float64)
    chunk[:, 0] = 0.75
    with pytest.raises(ValueError, match="first GR00T arm target"):
        validate_action_chunk(
            chunk,
            measured_arm=np.zeros(14),
            config=_Config(),
            initial_delta_limit_rad=0.5,
            step_delta_limit_rad=0.2,
        )
    report = validate_action_chunk(
        chunk,
        measured_arm=np.zeros(14),
        config=_Config(),
        initial_delta_limit_rad=0.5,
        step_delta_limit_rad=0.2,
        enforce_initial_delta=False,
    )
    assert report["initial_arm_delta_max_rad"] == pytest.approx(0.75)


def _state_observation(body: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        body_joint_position_rad=np.asarray(body, dtype=np.float64),
        dex1_opening_fraction=(0.0, 0.0),
        camera_capture_monotonic_ns={
            "head_left": 1_000_000_000,
            "head_right": 1_001_000_000,
            "left_wrist": 1_002_000_000,
            "right_wrist": 1_003_000_000,
        },
    )


def _narrow_stat_dimension(root: Path, index: int) -> None:
    path = root / "statistics.json"
    payload = json.loads(path.read_text())
    robot_q = payload["new_embodiment"]["state"]["robot_q"]
    robot_q["mean"][index] = 0.15
    robot_q["std"][index] = 0.02
    robot_q["q01"][index] = 0.10
    robot_q["q99"][index] = 0.20
    path.write_text(json.dumps(payload))


def test_context_only_lower_body_tail_inside_raw_support_is_diagnostic(
    tmp_path: Path,
) -> None:
    _write_metadata(tmp_path)
    # Model-state dimension 8 is robot_q/body index 1: left hip roll.
    _narrow_stat_dimension(tmp_path, 8)
    report = validate_state_distribution(_state_observation(np.zeros(29)), tmp_path)
    assert report["state_context_tail_indices"] == [8]
    assert report["state_context_tail_count"] == 1
    assert report["state_context_max_abs_z"] == pytest.approx(7.5)


def test_context_only_lower_body_outside_raw_support_is_diagnostic(
    tmp_path: Path,
) -> None:
    _write_metadata(tmp_path)
    body = np.zeros(29)
    body[1] = -2.0
    report = validate_state_distribution(_state_observation(body), tmp_path)
    assert 8 in report["state_distribution_warning_indices"]
    assert report["training_distribution_action_modified"] is False


def test_executable_arm_support_is_diagnostic_and_can_be_deferred_before_staging(
    tmp_path: Path,
) -> None:
    _write_metadata(tmp_path)
    # Model-state dimension 22 is robot_q/body index 15: left shoulder pitch.
    _narrow_stat_dimension(tmp_path, 22)
    observation = _state_observation(np.zeros(29))
    report = validate_state_distribution(observation, tmp_path)
    assert 22 in report["state_distribution_warning_indices"]
    assert report["training_distribution_action_modified"] is False
    report = validate_state_distribution(
        observation,
        tmp_path,
        validate_executable_state=False,
    )
    assert report["state_executable_validation_enabled"] is False
    assert 22 not in report["state_distribution_warning_indices"]
