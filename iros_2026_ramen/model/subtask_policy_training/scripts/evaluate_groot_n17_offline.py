"""Offline chunk-reset evaluation for the contract-preserving GR00T N1.7 policy.

The evaluator feeds recorded observations to the policy and re-anchors every
predicted chunk at the recorded state. It is a model contract and imitation
quality check, not a simulator or real-robot success evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, TRAINING_ROOT / "lerobot_policy_furniture_groot"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lerobot_policy_furniture_groot  # noqa: F401
from lerobot_policy_furniture_groot.modeling_furniture_groot import FurnitureGrootPolicy

from model.subtask_policy_training.gr00t.dex1_hand_synergy import hand_to_dex1
from model.subtask_policy_training.gr00t.g1_full_body_mapping import (
    GROOT_N17_NATIVE_ACTION_HORIZON,
    GROOT_N17_PACKED_ACTION_DIM,
    GROOT_N17_PACKED_STATE_DIM,
    GROOT_N17_VALID_ACTION_DIM,
    REAL_G1_RELATIVE_EEF_ACTION_DIM,
    REAL_G1_RELATIVE_EEF_ACTION_SLICES,
    REAL_G1_RELATIVE_EEF_STATE_DIM,
    REAL_G1_RELATIVE_EEF_STATE_SLICES,
    STANDARD_POLICY_VIDEO_KEYS,
)

FPS = 30.0
PHYSICAL_ARM_DIM = 14
PHYSICAL_ACTION_DIM = 16
UINT32_MODULUS = 2**32
OFFLINE_EPISODE_SEED_STRIDE = 1_000_003


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated episode indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-split",
        choices=("train", "validation", "test"),
        default="test",
        help="Declared split that every requested episode must belong to.",
    )
    parser.add_argument("--write-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-chunks-per-episode", type=int)
    return parser.parse_args()


def parse_episode_indices(value: str) -> list[int]:
    try:
        result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--episodes must be comma-separated integers") from exc
    if not result or any(index < 0 for index in result):
        raise ValueError("--episodes must contain non-negative indices")
    return result


def offline_chunk_inference_seed(
    *,
    base_seed: int,
    episode_index: int,
    chunk_ordinal: int,
) -> int:
    """Return an order-independent uint32 seed for one flow-matching chunk."""
    if not 0 <= base_seed < UINT32_MODULUS:
        raise ValueError("base seed must be a uint32")
    if episode_index < 0 or chunk_ordinal < 0:
        raise ValueError("episode index and chunk ordinal must be non-negative")
    return (
        base_seed
        + episode_index * OFFLINE_EPISODE_SEED_STRIDE
        + chunk_ordinal
    ) % UINT32_MODULUS


def seed_inference(seed: int) -> None:
    """Seed every RNG consumed by preprocessing or flow-matching inference."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_episode_bounds(dataset_root: Path) -> dict[int, tuple[int, int]]:
    bounds: dict[int, tuple[int, int]] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        columns = ["episode_index", "dataset_from_index", "dataset_to_index"]
        for row in pq.read_table(path, columns=columns).to_pylist():
            bounds[int(row["episode_index"])] = (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
            )
    if not bounds:
        raise FileNotFoundError(f"no episode metadata under {dataset_root}")
    return bounds


def load_episode_orientation_groups(dataset_root: Path) -> dict[int, str]:
    groups: dict[int, str] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "curation_orientation_cluster"],
        )
        for row in table.to_pylist():
            episode_index = int(row["episode_index"])
            if episode_index in groups:
                raise ValueError(f"duplicate orientation metadata for episode {episode_index}")
            groups[episode_index] = str(int(row["curation_orientation_cluster"]))
    if sorted(groups) != list(range(174)):
        raise ValueError("orientation metadata must contain exactly episodes 0..173")
    return groups


def load_split(dataset_root: Path) -> dict[str, set[int]]:
    path = dataset_root / "meta" / "team_ramen_episode_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: {int(index) for index in payload["splits"][name]["episode_indices"]}
        for name in ("train", "validation", "test")
    }


def validate_checkpoint_contract(config: Any) -> None:
    expected = {
        "type": "furniture_groot",
        "chunk_size": GROOT_N17_NATIVE_ACTION_HORIZON,
        "max_state_dim": GROOT_N17_PACKED_STATE_DIM,
        "max_action_dim": GROOT_N17_PACKED_ACTION_DIM,
        "valid_action_dim": GROOT_N17_VALID_ACTION_DIM,
    }
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected:
        raise ValueError(f"checkpoint violates the pinned GR00T contract: {actual} != {expected}")
    if tuple(config.observation_delta_indices) != (-20, 0):
        raise ValueError("checkpoint must use the official [-20,0] observation history")


def action_to_physical(action: Sequence[float]) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32)
    if values.shape != (REAL_G1_RELATIVE_EEF_ACTION_DIM,):
        raise ValueError(f"expected 53-D logical action, got {values.shape}")
    left_arm = values[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_arm"])]
    right_arm = values[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_arm"])]
    left_hand = values[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["left_hand"])]
    right_hand = values[slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES["right_hand"])]
    return np.concatenate(
        (
            left_arm,
            right_arm,
            np.asarray(
                [
                    hand_to_dex1(left_hand, side="left", kind="action"),
                    hand_to_dex1(right_hand, side="right", kind="action"),
                ],
                dtype=np.float32,
            ),
        )
    )


def state_arms(state: Sequence[float]) -> np.ndarray:
    values = np.asarray(state, dtype=np.float32)
    if values.shape != (REAL_G1_RELATIVE_EEF_STATE_DIM,):
        raise ValueError(f"expected 49-D state, got {values.shape}")
    return np.concatenate(
        (
            values[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["left_arm"])],
            values[slice(*REAL_G1_RELATIVE_EEF_STATE_SLICES["right_arm"])],
        )
    )


def compute_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    states: np.ndarray,
    *,
    progress_prediction: np.ndarray | None = None,
    progress_target: np.ndarray | None = None,
    progress_valid: np.ndarray | None = None,
) -> dict[str, float | None]:
    expected = (len(target), REAL_G1_RELATIVE_EEF_ACTION_DIM)
    if predicted.shape != expected or target.shape != expected:
        raise ValueError(f"predicted and target must both be {expected}")
    if states.shape != (len(target), REAL_G1_RELATIVE_EEF_STATE_DIM):
        raise ValueError("state trace shape does not match action trace")

    metrics: dict[str, float | None] = {}
    for group in ("left_wrist_eef_9d", "right_wrist_eef_9d", "left_hand", "right_hand", "left_arm", "right_arm"):
        group_slice = slice(*REAL_G1_RELATIVE_EEF_ACTION_SLICES[group])
        error = predicted[:, group_slice] - target[:, group_slice]
        metrics[f"{group}_mae"] = float(np.mean(np.abs(error)))
        metrics[f"{group}_rmse"] = float(np.sqrt(np.mean(np.square(error))))

    predicted_physical = np.stack([action_to_physical(row) for row in predicted])
    target_physical = np.stack([action_to_physical(row) for row in target])
    arm_error = predicted_physical[:, :PHYSICAL_ARM_DIM] - target_physical[:, :PHYSICAL_ARM_DIM]
    dex_error = predicted_physical[:, PHYSICAL_ARM_DIM:] - target_physical[:, PHYSICAL_ARM_DIM:]
    metrics.update(
        {
            "physical_arm_rmse_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
            "physical_arm_p95_abs_error_rad": float(np.quantile(np.abs(arm_error), 0.95)),
            "dex1_mae": float(np.mean(np.abs(dex_error))),
            "dex1_open_closed_accuracy": float(
                np.mean(
                    (predicted_physical[:, PHYSICAL_ARM_DIM:] < 2.25)
                    == (target_physical[:, PHYSICAL_ARM_DIM:] < 2.25)
                )
            ),
            "predicted_arm_range_rad": float(np.max(np.ptp(predicted_physical[:, :PHYSICAL_ARM_DIM], axis=0))),
            "predicted_dex1_range": float(np.max(np.ptp(predicted_physical[:, PHYSICAL_ARM_DIM:], axis=0))),
            "target_arm_range_rad": float(np.max(np.ptp(target_physical[:, :PHYSICAL_ARM_DIM], axis=0))),
            "target_dex1_range": float(np.max(np.ptp(target_physical[:, PHYSICAL_ARM_DIM:], axis=0))),
        }
    )
    current_arms = np.stack([state_arms(row) for row in states])
    displacement = np.abs(predicted_physical[:, :PHYSICAL_ARM_DIM] - current_arms)
    metrics["arm_max_abs_displacement_from_recorded_state_rad"] = float(np.max(displacement))
    metrics["stationary_frame_fraction"] = float(
        np.mean(
            (np.max(displacement, axis=1) < 0.02)
            & (np.max(np.abs(dex_error), axis=1) < 0.1)
        )
    )

    if progress_prediction is not None:
        if progress_target is None or progress_valid is None:
            raise ValueError("progress target and mask are required with progress predictions")
        valid = progress_valid.astype(bool)
        metrics["progress_valid_fraction"] = float(np.mean(valid))
        metrics["progress_mae"] = (
            float(np.mean(np.abs(progress_prediction[valid] - progress_target[valid])))
            if np.any(valid)
            else None
        )
        decreases = np.diff(progress_prediction, axis=1) < -1e-4
        pair_valid = valid[:, :-1] & valid[:, 1:]
        metrics["progress_monotonicity_violation_fraction"] = (
            float(np.mean(decreases[pair_valid])) if np.any(pair_valid) else None
        )
    return metrics


def compute_orientation_group_metrics(
    episode_traces: dict[int, dict[str, np.ndarray]],
    episode_groups: dict[int, str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, np.ndarray]]]] = {}
    for episode_index, trace in episode_traces.items():
        if episode_index not in episode_groups:
            raise KeyError(f"missing orientation group for episode {episode_index}")
        grouped.setdefault(episode_groups[episode_index], []).append(
            (episode_index, trace)
        )

    result: dict[str, dict[str, Any]] = {}
    for group, entries in sorted(grouped.items(), key=lambda item: int(item[0])):
        entries.sort(key=lambda item: item[0])
        result[group] = {
            "episodes": [episode_index for episode_index, _ in entries],
            "aggregate": compute_metrics(
                np.concatenate(
                    [trace["predicted_action"] for _, trace in entries]
                ),
                np.concatenate([trace["target_action"] for _, trace in entries]),
                np.concatenate([trace["state"] for _, trace in entries]),
            ),
        }
    return result


def _latest_vector(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[-1]
    if value.ndim == 1:
        return value
    raise ValueError(f"expected vector or temporal vectors, got {tuple(value.shape)}")


def evaluate_episode(
    *,
    dataset: Any,
    bounds: tuple[int, int],
    policy: FurnitureGrootPolicy,
    preprocessor: Any,
    postprocessor: Any,
    execution_steps: int,
    max_chunks: int | None,
    episode_index: int,
    base_seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    from lerobot.utils.collate import lerobot_collate_fn

    start, end = bounds
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    states: list[np.ndarray] = []
    frame_indices: list[int] = []
    progress_predictions: list[np.ndarray] = []
    progress_targets: list[np.ndarray] = []
    progress_masks: list[np.ndarray] = []
    latencies_ms: list[float] = []
    inference_seeds: list[int] = []
    initial_chunk_predicted: np.ndarray | None = None
    initial_chunk_target: np.ndarray | None = None
    initial_state: np.ndarray | None = None

    chunk_starts = list(range(start, end, execution_steps))
    if max_chunks is not None:
        chunk_starts = chunk_starts[:max_chunks]
    for chunk_ordinal, frame_index in enumerate(chunk_starts):
        inference_seed = offline_chunk_inference_seed(
            base_seed=base_seed,
            episode_index=episode_index,
            chunk_ordinal=chunk_ordinal,
        )
        seed_inference(inference_seed)
        sample = dataset[frame_index]
        target_chunk = sample["action"]
        if target_chunk.shape != (GROOT_N17_NATIVE_ACTION_HORIZON, REAL_G1_RELATIVE_EEF_ACTION_DIM):
            raise ValueError(f"dataset returned invalid action chunk {tuple(target_chunk.shape)}")
        batch = lerobot_collate_fn([sample])
        processed = preprocessor(batch)
        policy.reset()
        begin = time.perf_counter()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(processed)
            decoded = postprocessor(prediction)
            progress = policy.predict_progress(processed) if policy.config.progress_enabled else None
        latencies_ms.append((time.perf_counter() - begin) * 1000.0)
        if not isinstance(decoded, torch.Tensor):
            raise TypeError(f"postprocessor returned {type(decoded).__name__}, expected Tensor")
        decoded_np = decoded[0].detach().cpu().float().numpy()
        if initial_chunk_predicted is None:
            initial_chunk_predicted = np.stack(
                [action_to_physical(row) for row in decoded_np]
            )
            initial_chunk_target = np.stack(
                [action_to_physical(row) for row in target_chunk.detach().cpu().float().numpy()]
            )
            initial_state = _latest_vector(sample["observation.state"]).detach().cpu().float().numpy()
        count = min(execution_steps, end - frame_index)
        predictions.extend(decoded_np[:count])
        targets.extend(target_chunk[:count].detach().cpu().float().numpy())
        current_state = _latest_vector(sample["observation.state"]).detach().cpu().float().numpy()
        states.extend(np.repeat(current_state[None], count, axis=0))
        frame_indices.extend(range(frame_index, frame_index + count))
        inference_seeds.extend([inference_seed] * count)
        if progress is not None:
            progress_predictions.append(progress[0].detach().cpu().float().numpy())
            progress_targets.append(
                _latest_vector(sample["observation.progress_horizon"])
                .detach()
                .cpu()
                .float()
                .numpy()[..., None]
            )
            progress_masks.append(
                _latest_vector(sample["observation.progress_mask"])
                .detach()
                .cpu()
                .bool()
                .numpy()[..., None]
            )

    predicted_np = np.asarray(predictions, dtype=np.float32)
    target_np = np.asarray(targets, dtype=np.float32)
    states_np = np.asarray(states, dtype=np.float32)
    progress_prediction_np = np.asarray(progress_predictions, dtype=np.float32) if progress_predictions else None
    progress_target_np = np.asarray(progress_targets, dtype=np.float32) if progress_targets else None
    progress_mask_np = np.asarray(progress_masks, dtype=bool) if progress_masks else None
    metrics = compute_metrics(
        predicted_np,
        target_np,
        states_np,
        progress_prediction=progress_prediction_np,
        progress_target=progress_target_np,
        progress_valid=progress_mask_np,
    )
    if initial_chunk_predicted is None or initial_chunk_target is None or initial_state is None:
        raise ValueError("episode evaluation produced no chunks")
    initial_arms = state_arms(initial_state)
    metrics["initial_chunk_arm_max_abs_displacement_rad"] = float(
        np.max(np.abs(initial_chunk_predicted[:, :14] - initial_arms))
    )
    metrics["initial_chunk_target_arm_max_abs_displacement_rad"] = float(
        np.max(np.abs(initial_chunk_target[:, :14] - initial_arms))
    )
    metrics["initial_chunk_dex1_range"] = float(
        np.max(np.ptp(initial_chunk_predicted[:, 14:], axis=0))
    )
    metrics["initial_chunk_target_dex1_range"] = float(
        np.max(np.ptp(initial_chunk_target[:, 14:], axis=0))
    )
    metrics["chunk_inference_ms_mean"] = float(np.mean(latencies_ms))
    metrics["chunk_inference_ms_p95"] = float(np.quantile(latencies_ms, 0.95))
    metrics["evaluated_chunks"] = float(len(chunk_starts))
    trace = {
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "state": states_np,
        "predicted_action": predicted_np,
        "target_action": target_np,
        "predicted_physical_action": np.stack([action_to_physical(row) for row in predicted_np]),
        "target_physical_action": np.stack([action_to_physical(row) for row in target_np]),
        "inference_seed": np.asarray(inference_seeds, dtype=np.uint32),
    }
    if progress_prediction_np is not None:
        trace.update(
            {
                "progress_prediction": progress_prediction_np,
                "progress_target": progress_target_np,
                "progress_valid": progress_mask_np,
            }
        )
    return metrics, trace


def rgb_uint8(sample: dict[str, Any], key: str) -> np.ndarray:
    value = sample[key]
    if value.ndim == 4:
        value = value[-1]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"{key} must be CHW or temporal TCHW RGB")
    array = value.detach().cpu().permute(1, 2, 0).numpy()
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return array


def write_video(dataset: Any, trace: dict[str, np.ndarray], output_path: Path) -> None:
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (960, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {output_path}")
    try:
        for offset, frame_index in enumerate(trace["frame_index"]):
            sample = dataset[int(frame_index)]
            head = cv2.resize(
                cv2.cvtColor(rgb_uint8(sample, STANDARD_POLICY_VIDEO_KEYS[0]), cv2.COLOR_RGB2BGR),
                (640, 480),
                interpolation=cv2.INTER_AREA,
            )
            wrists = []
            for key in STANDARD_POLICY_VIDEO_KEYS[1:]:
                image = cv2.cvtColor(rgb_uint8(sample, key), cv2.COLOR_RGB2BGR)
                wrists.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
            frame = np.hstack((head, np.vstack(wrists)))
            pred = trace["predicted_physical_action"][offset]
            target = trace["target_physical_action"][offset]
            arm_rmse = float(np.sqrt(np.mean(np.square(pred[:14] - target[:14]))))
            dex_mae = float(np.mean(np.abs(pred[14:] - target[14:])))
            cv2.rectangle(frame, (0, 0), (960, 60), (0, 0, 0), thickness=-1)
            cv2.putText(
                frame,
                "OFFLINE CHUNK-RESET: recorded RGB, not closed-loop success",
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"frame={int(frame_index):05d} arm_rmse={arm_rmse:.4f} rad dex1_mae={dex_mae:.4f}",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (80, 255, 80),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    episodes = parse_episode_indices(args.episodes)
    if args.execution_steps <= 0 or args.execution_steps > GROOT_N17_NATIVE_ACTION_HORIZON:
        raise ValueError("--execution-steps must be in [1,40]")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested device")
    if not 0 <= args.seed < UINT32_MODULUS:
        raise ValueError("--seed must be a uint32")
    seed_inference(args.seed)

    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors

    split = load_split(args.dataset_root)
    disallowed = sorted(set(episodes) - split[args.evaluation_split])
    if disallowed:
        raise ValueError(
            f"episodes must belong to the declared {args.evaluation_split} split: {disallowed}"
        )

    config = PreTrainedConfig.from_pretrained(str(args.model_dir))
    config.device = args.device
    validate_checkpoint_contract(config)
    policy = FurnitureGrootPolicy.from_pretrained(
        str(args.model_dir), config=config, strict=True
    ).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(args.model_dir)
    )

    marker_path = args.dataset_root / "meta" / "team_ramen_training_view.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    repo_id = str(marker["source_repo_id"])
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, meta)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    bounds = load_episode_bounds(args.dataset_root)
    orientation_groups = load_episode_orientation_groups(args.dataset_root)
    missing = sorted(set(episodes) - set(bounds))
    if missing:
        raise ValueError(f"episodes missing from dataset: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "groot_n17_offline_chunk_reset_v2",
        "evaluation_type": "offline_chunk_reset_not_closed_loop",
        "model_dir": str(args.model_dir.resolve()),
        "model_safetensors_sha256": sha256_file(
            args.model_dir / "model.safetensors"
        ),
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": episodes,
        "declared_split": args.evaluation_split,
        "execution_steps": args.execution_steps,
        "randomness": {
            "base_seed": args.seed,
            "scope": "one deterministic seed per episode chunk",
            "formula": (
                "(base_seed + episode_index * episode_stride + chunk_ordinal) "
                "mod uint32_modulus"
            ),
            "episode_stride": OFFLINE_EPISODE_SEED_STRIDE,
            "uint32_modulus": UINT32_MODULUS,
        },
        "contract": {
            "state_dim": REAL_G1_RELATIVE_EEF_STATE_DIM,
            "logical_action_dim": REAL_G1_RELATIVE_EEF_ACTION_DIM,
            "packed_action_dim": GROOT_N17_PACKED_ACTION_DIM,
            "valid_action_dim": GROOT_N17_VALID_ACTION_DIM,
            "horizon": GROOT_N17_NATIVE_ACTION_HORIZON,
            "cameras": list(STANDARD_POLICY_VIDEO_KEYS),
            "observation_delta_indices": list(config.observation_delta_indices),
        },
        "episodes_report": {},
    }
    all_predicted: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    episode_traces: dict[int, dict[str, np.ndarray]] = {}
    for episode in episodes:
        metrics, trace = evaluate_episode(
            dataset=dataset,
            bounds=bounds[episode],
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            execution_steps=args.execution_steps,
            max_chunks=args.max_chunks_per_episode,
            episode_index=episode,
            base_seed=args.seed,
        )
        np.savez_compressed(args.output_dir / f"episode_{episode:04d}_trace.npz", **trace)
        if args.write_videos:
            write_video(
                dataset,
                trace,
                args.output_dir / f"episode_{episode:04d}_offline_chunk_reset.mp4",
            )
        report["episodes_report"][str(episode)] = metrics
        episode_traces[episode] = trace
        all_predicted.append(trace["predicted_action"])
        all_target.append(trace["target_action"])
        all_states.append(trace["state"])

    report["aggregate"] = compute_metrics(
        np.concatenate(all_predicted),
        np.concatenate(all_target),
        np.concatenate(all_states),
    )
    report["orientation_group_report"] = compute_orientation_group_metrics(
        episode_traces,
        orientation_groups,
    )
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
