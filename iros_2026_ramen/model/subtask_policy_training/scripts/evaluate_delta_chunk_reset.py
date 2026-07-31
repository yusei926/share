"""Offline chunk-reset evaluation for absolute or chunk-relative arm policies.

This intentionally evaluates recorded observations only. Each action chunk starts
from the recorded state, so the result is not a closed-loop robot or simulator
success rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.subtask_policy_training.action_representation import (
    ACTION_DIM,
    ARM_DIM,
    ABSOLUTE_TARGET,
    decode_action_chunk,
    semantics as action_semantics,
    validate_representation,
)
from model.subtask_policy_training.native_delta_policy import (
    NativeACTConfig,
    VIDEO_BACKEND,
    VIDEO_TIMESTAMP_TOLERANCE_S,
    configure_serial_video_decode,
    denormalize,
    load_native_act_checkpoint,
    normalize,
    normalizer_from_stats,
)


CAMERA_KEYS = (
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated held-out episode indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-videos", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_episode_indices(value: str) -> list[int]:
    try:
        result = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("--episodes must be comma-separated integers") from exc
    if not result or any(index < 0 for index in result):
        raise ValueError("--episodes must contain unique non-negative indices")
    return result


def load_episode_bounds(dataset_root: Path) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for path in sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path, columns=["episode_index", "dataset_from_index", "dataset_to_index"]).to_pylist():
            result[int(row["episode_index"])] = (
                int(row["dataset_from_index"]),
                int(row["dataset_to_index"]),
            )
    if not result:
        raise FileNotFoundError(f"no episode metadata under {dataset_root}")
    return result


def load_policy_rows(dataset_root: Path) -> dict[int, tuple[list[float], list[float]]]:
    result: dict[int, tuple[list[float], list[float]]] = {}
    for path in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(path, columns=["index", "observation.state", "action"])
        for row in table.to_pylist():
            result[int(row["index"])] = (
                [float(value) for value in row["observation.state"]],
                [float(value) for value in row["action"]],
            )
    if not result:
        raise FileNotFoundError(f"no policy parquet data under {dataset_root}")
    return result


def compute_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != ACTION_DIM:
        raise ValueError("predicted and target must be [frames, 16] with matching shape")
    arm_error = predicted[:, :ARM_DIM] - target[:, :ARM_DIM]
    arm_abs = np.abs(arm_error)
    gripper_error = predicted[:, ARM_DIM:] - target[:, ARM_DIM:]
    final_step_error = arm_error[-1:]
    transition_f1 = gripper_transition_f1(predicted[:, ARM_DIM:], target[:, ARM_DIM:])
    return {
        "arm_rmse_rad": float(np.sqrt(np.mean(np.square(arm_error)))),
        "arm_p95_abs_error_rad": float(np.quantile(arm_abs, 0.95)),
        "arm_final_step_rmse_rad": float(np.sqrt(np.mean(np.square(final_step_error)))),
        "gripper_mae": float(np.mean(np.abs(gripper_error))),
        "gripper_transition_f1": transition_f1,
        "action_jerk_rad_s3": action_jerk(predicted[:, :ARM_DIM]),
    }


def gripper_transition_f1(predicted: np.ndarray, target: np.ndarray) -> float:
    if len(target) < 2:
        return 1.0
    ranges = np.ptp(target, axis=0)
    thresholds = np.maximum(0.02, ranges * 0.05)
    truth = np.any(np.abs(np.diff(target, axis=0)) >= thresholds, axis=1)
    predicted_events = np.any(np.abs(np.diff(predicted, axis=0)) >= thresholds, axis=1)
    true_positive = int(np.sum(truth & predicted_events))
    false_positive = int(np.sum(~truth & predicted_events))
    false_negative = int(np.sum(truth & ~predicted_events))
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else float(2 * true_positive / denominator)


def action_jerk(arm_targets: np.ndarray) -> float:
    if len(arm_targets) < 4:
        return 0.0
    jerk = np.diff(arm_targets, n=3, axis=0) * FPS**3
    return float(np.mean(np.linalg.norm(jerk, axis=1)))


class NativeACTEvaluator:
    """Inference adapter for the two-frame, separate-camera native ACT checkpoint."""

    def __init__(self, model_dir: Path, device: str):
        self.device = torch.device(device)
        self.model, stats = load_native_act_checkpoint(model_dir, device=self.device)
        self.config = self.model.config
        self.action_representation = model_action_representation(model_dir)
        self.state_stats = normalizer_from_stats(stats, "observation.state", device=self.device)
        self.action_stats = normalizer_from_stats(stats, "action", device=self.device)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        from model.subtask_policy_training.scripts.train_native_act_delta import prepare_images

        images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=2).to(self.device)
        state = batch["observation.state"].to(self.device).float()
        images = prepare_images(images, training=False)
        normalized = self.model.predict_action_chunk(images, normalize(state, self.state_stats))
        return denormalize(normalized, self.action_stats)


class NativeDiffusionEvaluator:
    """EMA Diffusion Policy adapter that evaluates its complete 16-step horizon."""

    def __init__(self, model_dir: Path, device: str):
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        self.device = torch.device(device)
        config = PreTrainedConfig.from_pretrained(str(model_dir / "_diffusion_config"))
        config.device = str(self.device)
        if bool(getattr(config, "clip_sample", True)):
            raise ValueError(
                "z-score action normalization requires DiffusionConfig.clip_sample=false"
            )
        self.policy = DiffusionPolicy.from_pretrained(str(model_dir), config=config, strict=True).eval()
        self.config = config
        self.action_representation = model_action_representation(model_dir)
        stats = json.loads((model_dir / "normalization.json").read_text(encoding="utf-8"))
        self.state_stats = normalizer_from_stats(stats, "observation.state", device=self.device)
        self.action_stats = normalizer_from_stats(stats, "action", device=self.device)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        from lerobot.utils.constants import OBS_IMAGES
        from model.subtask_policy_training.scripts.train_native_act_delta import prepare_images

        images = torch.stack([batch[key] for key in CAMERA_KEYS], dim=2).to(self.device)
        state = batch["observation.state"].to(self.device).float()
        images = prepare_images(images, training=False)
        encoded_batch = {
            "observation.state": normalize(state, self.state_stats),
            OBS_IMAGES: images,
        }
        condition = self.policy.diffusion._prepare_global_conditioning(encoded_batch)
        normalized_actions = self.policy.diffusion.conditional_sample(state.shape[0], global_cond=condition)
        return denormalize(normalized_actions, self.action_stats)


def model_action_representation(model_dir: Path) -> str:
    path = model_dir / "train_config.json"
    if not path.is_file():
        return ABSOLUTE_TARGET
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_representation(payload.get("action_representation", ABSOLUTE_TARGET))


def load_policy(model_dir: Path, device: str) -> tuple[Any, Any, Any]:
    config_payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config_payload.get("type") == NativeACTConfig.type:
        return NativeACTEvaluator(model_dir, device), None, None
    if config_payload.get("type") == "flip_table_native_diffusion_chunk_relative":
        return NativeDiffusionEvaluator(model_dir, device), None, None
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(str(model_dir))
    config.device = device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(str(model_dir), config=config, strict=True)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(model_dir))
    return policy, preprocessor, postprocessor


def observation_batch(dataset: Any, *, frame_index: int, episode_start: int, n_obs_steps: int) -> dict[str, torch.Tensor]:
    observations = [dataset[max(episode_start, frame_index - offset)] for offset in range(n_obs_steps - 1, -1, -1)]
    batch: dict[str, torch.Tensor] = {}
    for key in ("observation.state", *CAMERA_KEYS):
        values = [sample[key] for sample in observations]
        stacked = torch.stack(values)
        batch[key] = stacked.unsqueeze(0)
    return batch


def predict_chunk(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    batch: dict[str, torch.Tensor],
    *,
    n_obs_steps: int,
) -> np.ndarray:
    if isinstance(policy, (NativeACTEvaluator, NativeDiffusionEvaluator)):
        return policy.predict(batch)[0].detach().cpu().float().numpy()
    if n_obs_steps == 1:
        batch = {key: value[:, 0] for key, value in batch.items()}
    processed = preprocessor(batch)
    policy.reset()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(processed)
        raw_chunk = postprocessor(chunk)
    if not isinstance(raw_chunk, torch.Tensor):
        raise TypeError(f"postprocessor returned {type(raw_chunk).__name__}, expected Tensor")
    return raw_chunk[0].detach().cpu().float().numpy()


def evaluate_episode(
    *,
    dataset: Any,
    rows: dict[int, tuple[list[float], list[float]]],
    bounds: tuple[int, int],
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    action_representation: str,
    chunk_size: int,
    n_obs_steps: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    start, end = bounds
    predicted_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    state_frames: list[np.ndarray] = []
    latencies_ms: list[float] = []
    for chunk_start in range(start, end, chunk_size):
        count = min(chunk_size, end - chunk_start)
        state = rows[chunk_start][0]
        target_actions = [rows[index][1] for index in range(chunk_start, chunk_start + count)]
        batch = observation_batch(
            dataset,
            frame_index=chunk_start,
            episode_start=start,
            n_obs_steps=n_obs_steps,
        )
        begin = time.perf_counter()
        predicted_model_actions = predict_chunk(
            policy,
            preprocessor,
            postprocessor,
            batch,
            n_obs_steps=n_obs_steps,
        )
        latencies_ms.append((time.perf_counter() - begin) * 1000.0)
        if predicted_model_actions.shape[0] < count or predicted_model_actions.shape[1] != ACTION_DIM:
            raise ValueError(f"policy returned invalid action chunk shape {predicted_model_actions.shape}")
        reference_state = torch.as_tensor(state, dtype=torch.float32).reshape(1, 1, -1)
        predicted_frames.extend(
            decode_action_chunk(
                torch.as_tensor(predicted_model_actions[:count], dtype=torch.float32).unsqueeze(0),
                reference_state,
                action_representation,
            )[0]
            .cpu()
            .numpy()
        )
        target_frames.extend(np.asarray(target_actions, dtype=np.float32))
        state_frames.extend(
            np.asarray([rows[index][0] for index in range(chunk_start, chunk_start + count)], dtype=np.float32)
        )
    predicted = np.asarray(predicted_frames, dtype=np.float32)
    target = np.asarray(target_frames, dtype=np.float32)
    metrics = compute_metrics(predicted, target)
    initial_count = min(chunk_size, len(predicted))
    initial_reference = np.asarray(state_frames[0][3 : 3 + ARM_DIM], dtype=np.float32)
    metrics["initial_chunk_arm_max_abs_displacement_rad"] = float(
        np.max(np.abs(predicted[:initial_count, :ARM_DIM] - initial_reference))
    )
    metrics["initial_chunk_gripper_range"] = float(
        np.max(np.ptp(predicted[:initial_count, ARM_DIM:], axis=0))
    )
    metrics["chunk_inference_ms_mean"] = float(np.mean(latencies_ms))
    metrics["chunk_inference_ms_p95"] = float(np.quantile(latencies_ms, 0.95))
    return metrics, {"state": np.asarray(state_frames), "predicted": predicted, "target": target}


def rgb_uint8(sample: dict[str, Any], key: str) -> np.ndarray:
    value = sample[key]
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] != 3
        or value.shape[1] <= 0
        or value.shape[2] <= 0
    ):
        raise ValueError(f"{key} must be uint8 CHW RGB, got {getattr(value, 'shape', None)}")
    return value.detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8, copy=False)


def write_offline_rollout_video(
    *,
    dataset: Any,
    bounds: tuple[int, int],
    trace: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Render recorded observations with prediction error; this is not a closed-loop rollout."""
    import cv2

    start, end = bounds
    predicted = trace["predicted"]
    target = trace["target"]
    if len(predicted) != end - start or len(target) != end - start:
        raise ValueError("trace length does not match episode bounds")
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
        for offset, frame_index in enumerate(range(start, end)):
            sample = dataset[frame_index]
            head = cv2.resize(
                cv2.cvtColor(rgb_uint8(sample, CAMERA_KEYS[0]), cv2.COLOR_RGB2BGR),
                (640, 480),
                interpolation=cv2.INTER_AREA,
            )
            left = cv2.resize(
                cv2.cvtColor(rgb_uint8(sample, CAMERA_KEYS[1]), cv2.COLOR_RGB2BGR),
                (320, 240),
                interpolation=cv2.INTER_AREA,
            )
            right = cv2.resize(
                cv2.cvtColor(rgb_uint8(sample, CAMERA_KEYS[2]), cv2.COLOR_RGB2BGR),
                (320, 240),
                interpolation=cv2.INTER_AREA,
            )
            panel = np.vstack((left, right))
            frame = np.hstack((head, panel))
            arm_rmse = float(np.sqrt(np.mean((predicted[offset, :ARM_DIM] - target[offset, :ARM_DIM]) ** 2)))
            gripper_mae = float(np.mean(np.abs(predicted[offset, ARM_DIM:] - target[offset, ARM_DIM:])))
            cv2.rectangle(frame, (0, 0), (960, 58), (0, 0, 0), thickness=-1)
            cv2.putText(
                frame,
                "OFFLINE CHUNK-RESET: recorded observations, not a closed-loop rollout",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"frame={offset:04d} arm_rmse={arm_rmse:.4f} rad  dex1_mae={gripper_mae:.4f}",
                (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
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
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is required for this evaluation")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if os.environ.get("FLIP_TABLE_SERIAL_VIDEO_DECODE") == "1":
        configure_serial_video_decode()
    bounds = load_episode_bounds(args.dataset_root)
    missing = [episode for episode in episodes if episode not in bounds]
    if missing:
        raise ValueError(f"episodes are absent from dataset metadata: {missing}")
    rows = load_policy_rows(args.dataset_root)
    marker_path = args.dataset_root / "meta" / "team_ramen_training_view.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    repo_id = str(marker.get("source_repo_id", "")).strip()
    if not repo_id:
        raise ValueError(f"training view marker has no source_repo_id: {marker_path}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=args.dataset_root,
        video_backend=VIDEO_BACKEND,
        return_uint8=True,
        tolerance_s=VIDEO_TIMESTAMP_TOLERANCE_S,
    )
    policy, preprocessor, postprocessor = load_policy(args.model_dir, args.device)
    action_representation = getattr(policy, "action_representation", model_action_representation(args.model_dir))
    chunk_size = min(16, int(getattr(policy.config, "chunk_size", getattr(policy.config, "horizon", 16))))
    n_obs_steps = int(
        getattr(policy.config, "observation_horizon", getattr(policy.config, "n_obs_steps", 1))
    )
    report: dict[str, Any] = {
        "evaluation_type": "offline_chunk_reset",
        "model_dir": str(args.model_dir.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "episodes": episodes,
        "seed": args.seed,
        "chunk_size": chunk_size,
        "n_obs_steps": n_obs_steps,
        "video_backend": VIDEO_BACKEND,
        "action_representation": action_representation,
        "action_contract": action_semantics(action_representation),
        "episodes_report": {},
    }
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        metrics, trace = evaluate_episode(
            dataset=dataset,
            rows=rows,
            bounds=bounds[episode],
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            action_representation=action_representation,
            chunk_size=chunk_size,
            n_obs_steps=n_obs_steps,
        )
        np.savez_compressed(args.output_dir / f"episode_{episode:04d}_trace.npz", **trace)
        if args.write_videos:
            write_offline_rollout_video(
                dataset=dataset,
                bounds=bounds[episode],
                trace=trace,
                output_path=args.output_dir / f"episode_{episode:04d}_offline_chunk_reset.mp4",
            )
        report["episodes_report"][str(episode)] = metrics
        all_predictions.append(trace["predicted"])
        all_targets.append(trace["target"])
    aggregate = compute_metrics(np.concatenate(all_predictions), np.concatenate(all_targets))
    report["aggregate"] = aggregate
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
