"""Audited execution loop built on the V1 Isaac Lab Mimic generator."""

from __future__ import annotations

import asyncio
import contextvars
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.datagen_info_pool import DataGenInfoPool
from isaaclab_mimic.datagen.generation import env_loop

from ..fk_audit import synthetic_action_fk_report
from ..provenance import CandidateLedger, CandidateRecord
from .isaaclab_compat import install_missing_mimic_pose_helpers
from .physical_snapshot import snapshot_physical_randomization


install_missing_mimic_pose_helpers()


_SELECTION_EVENTS: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("flip_table_mimic_selection_events", default=None)
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_demo_episode_indices(path: str | Path) -> dict[int, int]:
    """Read the immutable Mimic-demo to source-episode mapping."""

    mapping = {}
    with h5py.File(Path(path), "r") as stream:
        data = stream["data"]
        demo_names = sorted(data, key=lambda name: int(name.removeprefix("demo_")))
        for demo_name in demo_names:
            demo_index = int(demo_name.removeprefix("demo_"))
            episode_index = int(data[demo_name].attrs["source_episode_index"])
            mapping[demo_index] = episode_index
    if not mapping:
        raise ValueError("Mimic source HDF5 contains no demonstrations")
    return mapping


class AuditedDataGenerator(DataGenerator):
    """Record source selections without changing Isaac Lab Mimic behavior."""

    def select_source_demo(
        self,
        eef_name,
        eef_pose,
        object_pose,
        src_demo_current_subtask_boundaries,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        selected = super().select_source_demo(
            eef_name,
            eef_pose,
            object_pose,
            src_demo_current_subtask_boundaries,
            subtask_object_name,
            selection_strategy_name,
            selection_strategy_kwargs,
        )
        events = _SELECTION_EVENTS.get()
        if events is not None:
            event: dict[str, Any] = {
                "eef": str(eef_name),
                "selection_ordinal": sum(event["eef"] == eef_name for event in events),
                "mimic_demo_index": int(selected),
                "strategy": str(selection_strategy_name),
            }
            if subtask_object_name is not None:
                boundaries = np.asarray(src_demo_current_subtask_boundaries)
                source_start = int(boundaries[int(selected), 0])
                source_pose = self.src_demo_datagen_info_pool.datagen_infos[
                    int(selected)
                ].object_poses[subtask_object_name][source_start]
                current_pose = torch.as_tensor(object_pose)
                source_pose = torch.as_tensor(
                    source_pose,
                    dtype=current_pose.dtype,
                    device=current_pose.device,
                )
                delta_pose = current_pose @ torch.linalg.inv(source_pose)
                event.update(
                    {
                        "object_name": str(subtask_object_name),
                        "source_start_step": source_start,
                        "current_object_pose": current_pose.detach().cpu().tolist(),
                        "source_object_pose": source_pose.detach().cpu().tolist(),
                        "delta_object_pose": delta_pose.detach().cpu().tolist(),
                    }
                )
            events.append(event)
        return selected

    async def generate(self, *args, **kwargs):
        events: list[dict[str, Any]] = []
        token = _SELECTION_EVENTS.set(events)
        try:
            result = await super().generate(*args, **kwargs)
            result["source_selection_events"] = events
            return result
        finally:
            _SELECTION_EVENTS.reset(token)


@dataclass
class GenerationCounters:
    attempted: int = 0
    accepted: int = 0
    rejected: int = 0
    resumed: int = 0


def _seed_attempt(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _recorder_handler(env):
    handler = getattr(env.recorder_manager, "_dataset_file_handler", None)
    if handler is None or getattr(handler, "_hdf5_data_group", None) is None:
        raise RuntimeError("the pinned HDF5 recorder handler is unavailable")
    return handler


def _episode_group(env, attempt_index: int) -> h5py.Group:
    handler = _recorder_handler(env)
    demo_name = f"demo_{attempt_index}"
    if demo_name not in handler._hdf5_data_group:
        raise RuntimeError(f"recorder did not export the accepted episode {demo_name}")
    return handler._hdf5_data_group[demo_name]


def _dataset_by_feature(group: h5py.Group, key: str) -> h5py.Dataset:
    direct = group.get(key)
    if isinstance(direct, h5py.Dataset):
        return direct
    current: h5py.Group | h5py.Dataset = group
    for component in key.split("."):
        if not isinstance(current, h5py.Group) or component not in current:
            raise RuntimeError(f"accepted trajectory lacks numeric feature {key}")
        current = current[component]
    if not isinstance(current, h5py.Dataset):
        raise RuntimeError(f"accepted trajectory numeric feature {key} is not a dataset")
    return current


def _accepted_action_fk_report(
    *,
    env,
    attempt_index: int,
    urdf_path: Path,
    action_fk_contract: dict[str, Any],
) -> dict[str, Any]:
    group = _episode_group(env, attempt_index)
    numeric = group.get("dataset_numeric")
    if not isinstance(numeric, h5py.Group):
        raise RuntimeError("accepted trajectory lacks dataset_numeric")
    return synthetic_action_fk_report(
        robot_q_desired=np.asarray(
            _dataset_by_feature(numeric, "action.robot_q_desired"), dtype=np.float64
        ),
        ee_action=np.asarray(
            _dataset_by_feature(numeric, "action.ee_action"), dtype=np.float64
        ),
        urdf_path=urdf_path,
        frame_names={
            side: str(value)
            for side, value in action_fk_contract["fk_frames"].items()
        },
        tool_transforms=action_fk_contract["fk_tool_transforms"],
        position_p95_max=float(
            action_fk_contract["fk_action_validation_position_p95_m_max"]
        ),
        rotation_p95_max=float(
            action_fk_contract["fk_action_validation_rotation_p95_rad_max"]
        ),
    )


def _delete_exported_episode(env, attempt_index: int) -> None:
    handler = _recorder_handler(env)
    demo_name = f"demo_{attempt_index}"
    if demo_name in handler._hdf5_data_group:
        samples = int(handler._hdf5_data_group[demo_name].attrs.get("num_samples", 0))
        total = int(handler._hdf5_data_group.attrs.get("total", 0))
        if samples < 0 or total < samples:
            raise RuntimeError(
                f"invalid HDF5 totals while removing {demo_name}: total={total}, samples={samples}"
            )
        del handler._hdf5_data_group[demo_name]
        handler._hdf5_data_group.attrs["total"] = total - samples
        handler._demo_count = len(handler._hdf5_data_group)
        handler.flush()


def _trajectory_group_sha256(group) -> str:
    """Hash trajectory arrays and Isaac Lab episode attributes, not provenance attrs."""

    digest = hashlib.sha256()

    def update(current, prefix: str) -> None:
        for name in sorted(current.keys()):
            value = current[name]
            path = f"{prefix}/{name}"
            digest.update(path.encode("utf-8"))
            if isinstance(value, h5py.Group):
                digest.update(b"group\0")
                update(value, path)
            else:
                array = np.asarray(value)
                digest.update(b"dataset\0")
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(np.ascontiguousarray(array).tobytes())

    for name in ("num_samples", "seed", "success"):
        if name in group.attrs:
            digest.update(name.encode("ascii"))
            digest.update(str(group.attrs[name]).encode("utf-8"))
    update(group, "")
    return digest.hexdigest()


def _json_attr(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _episode_series(env, env_id: int, path: str) -> torch.Tensor:
    episode = env.recorder_manager._episodes.get(env_id)
    if episode is None:
        raise RuntimeError(f"recorder has no episode buffer for env {env_id}")
    value: Any = episode.data
    for component in path.split("/"):
        if not isinstance(value, dict) or component not in value:
            raise RuntimeError(f"recorder episode lacks {path}")
        value = value[component]
    if isinstance(value, list):
        if not value:
            raise RuntimeError(f"recorder episode series is empty: {path}")
        value = torch.stack(value)
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise RuntimeError(f"recorder episode series must be a finite tensor: {path}")
    if value.ndim >= 2 and value.shape[1] == 1:
        value = value[:, 0]
    return value.detach().to(device="cpu", dtype=torch.float64)


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError("cannot summarize an empty trajectory")
    return {
        "median": float(torch.quantile(flat, 0.5)),
        "p95": float(torch.quantile(flat, 0.95)),
        "max": float(torch.amax(flat)),
    }


def _path_length(positions: torch.Tensor) -> float:
    if positions.shape[0] < 2:
        return 0.0
    return float(torch.linalg.norm(torch.diff(positions, dim=0), dim=1).sum())


def _quarter_summaries(values: torch.Tensor) -> list[dict[str, float]]:
    return [_quantiles(chunk) for chunk in torch.tensor_split(values, 4) if chunk.numel()]


def _position_bounds_by_quarter(positions: torch.Tensor) -> list[dict[str, list[float]]]:
    return [
        {
            "min_m": torch.amin(chunk, dim=0).tolist(),
            "max_m": torch.amax(chunk, dim=0).tolist(),
        }
        for chunk in torch.tensor_split(positions, 4)
        if chunk.numel()
    ]


def _trajectory_diagnostics(env, env_id: int) -> dict[str, Any]:
    prefix = "dataset_numeric/"
    ee_state = _episode_series(env, env_id, prefix + "observation.state.ee_state")
    ee_action = _episode_series(env, env_id, prefix + "action.ee_action")
    hand_state = _episode_series(env, env_id, prefix + "observation.state.hand_state")
    hand_command = _episode_series(env, env_id, prefix + "action.hand_cmd")
    joint_state = _episode_series(env, env_id, prefix + "observation.state.robot_q_current")
    joint_target = _episode_series(env, env_id, prefix + "action.robot_q_desired")
    lengths = {
        len(value)
        for value in (ee_state, ee_action, hand_state, hand_command, joint_state, joint_target)
    }
    if len(lengths) != 1:
        raise RuntimeError(f"numeric recorder series lengths differ: {sorted(lengths)}")

    eef = {}
    for side, offset in (("left", 0), ("right", 6)):
        measured = ee_state[:, offset : offset + 3]
        target = ee_action[:, offset : offset + 3]
        position_error = torch.linalg.norm(target - measured, dim=1)
        eef[side] = {
            "target_start_xyz_euler_rad": ee_action[0, offset : offset + 6].tolist(),
            "target_end_xyz_euler_rad": ee_action[-1, offset : offset + 6].tolist(),
            "measured_start_xyz_euler_rad": ee_state[0, offset : offset + 6].tolist(),
            "measured_end_xyz_euler_rad": ee_state[-1, offset : offset + 6].tolist(),
            "target_path_length_m": _path_length(target),
            "measured_path_length_m": _path_length(measured),
            "target_position_bounds_m_by_quarter": _position_bounds_by_quarter(target),
            "position_error_m": _quantiles(position_error),
            "position_error_m_by_quarter": _quarter_summaries(position_error),
        }
    object_pose = env.get_object_poses(env_ids=[env_id]).get("white_table")
    if object_pose is None or object_pose.shape != (1, 4, 4):
        raise RuntimeError("Mimic environment exposes no final white-table pose")
    return {
        "control_steps": lengths.pop(),
        "final_white_table_pose_robot_root": object_pose[0].detach().cpu().tolist(),
        "eef": eef,
        "hand_command_range": {
            "min": torch.amin(hand_command, dim=0).tolist(),
            "max": torch.amax(hand_command, dim=0).tolist(),
        },
        "hand_tracking_abs_error": _quantiles(torch.abs(hand_command - hand_state)),
        "body_joint_tracking_abs_error_rad": _quantiles(
            torch.abs(joint_target[:, 7:] - joint_state[:, 7:])
        ),
    }


def _write_episode_provenance(
    *,
    env,
    attempt_index: int,
    candidate: CandidateRecord,
    generation_payload: dict[str, Any],
    action_fk_report: dict[str, Any],
) -> str:
    handler = _recorder_handler(env)
    group = _episode_group(env, attempt_index)
    trajectory_sha256 = _trajectory_group_sha256(group)
    attrs = {
        "candidate_id": candidate.candidate_id,
        "trajectory_seed": candidate.trajectory_seed,
        "config_sha256": candidate.config_sha256,
        "runtime_digest": candidate.runtime_digest,
        "source_episode_indices_json": _json_attr(list(candidate.source_episode_indices)),
        "generation_payload_json": _json_attr(generation_payload),
        "action_fk_report_json": _json_attr(action_fk_report),
        "trajectory_sha256": trajectory_sha256,
    }
    for key, value in attrs.items():
        group.attrs[key] = value
    handler.flush()
    return trajectory_sha256


def _episode_provenance(
    env, attempt_index: int, *, clean_incomplete: bool
) -> dict[str, Any] | None:
    handler = _recorder_handler(env)
    demo_name = f"demo_{attempt_index}"
    if demo_name not in handler._hdf5_data_group:
        return None
    group = handler._hdf5_data_group[demo_name]
    required = {
        "candidate_id",
        "trajectory_seed",
        "config_sha256",
        "runtime_digest",
        "source_episode_indices_json",
        "generation_payload_json",
        "action_fk_report_json",
        "trajectory_sha256",
    }
    missing = required.difference(group.attrs)
    if missing:
        if clean_incomplete:
            _delete_exported_episode(env, attempt_index)
            return None
        raise RuntimeError(f"{demo_name} lacks resume provenance: {sorted(missing)}")
    stored_hash = str(group.attrs["trajectory_sha256"])
    actual_hash = _trajectory_group_sha256(group)
    if stored_hash != actual_hash:
        raise RuntimeError(f"{demo_name} trajectory hash changed")
    return {
        "demo_name": demo_name,
        "candidate_id": str(group.attrs["candidate_id"]),
        "trajectory_seed": int(group.attrs["trajectory_seed"]),
        "config_sha256": str(group.attrs["config_sha256"]),
        "runtime_digest": str(group.attrs["runtime_digest"]),
        "source_episode_indices": tuple(
            int(value) for value in json.loads(group.attrs["source_episode_indices_json"])
        ),
        "generation_payload": json.loads(group.attrs["generation_payload_json"]),
        "action_fk_report": json.loads(group.attrs["action_fk_report_json"]),
        "trajectory_sha256": stored_hash,
    }


def _reconcile_exported_episode(
    *, ledger: CandidateLedger, candidate: CandidateRecord, provenance: dict[str, Any]
) -> CandidateRecord:
    expected_identity = (
        candidate.candidate_id,
        candidate.trajectory_seed,
        candidate.config_sha256,
        candidate.runtime_digest,
    )
    actual_identity = (
        provenance["candidate_id"],
        provenance["trajectory_seed"],
        provenance["config_sha256"],
        provenance["runtime_digest"],
    )
    if actual_identity != expected_identity:
        raise RuntimeError(f"exported HDF5 identity mismatch for {candidate.candidate_id}")
    if candidate.status == "claimed":
        candidate = ledger.transition(
            candidate.candidate_id,
            "generated",
            provenance["generation_payload"],
            source_episode_indices=provenance["source_episode_indices"],
        )
    if candidate.status == "generated":
        if candidate.source_episode_indices != provenance["source_episode_indices"]:
            raise RuntimeError("exported source lineage differs from the candidate ledger")
        for key, value in provenance["generation_payload"].items():
            if candidate.payload.get(key) != value:
                raise RuntimeError(f"exported generation payload mismatch for {key}")
        candidate = ledger.transition(
            candidate.candidate_id,
            "validated",
            {
                "accepted_hdf5_demo": provenance["demo_name"],
                "trajectory_sha256": provenance["trajectory_sha256"],
                "action_fk_report": provenance["action_fk_report"],
            },
        )
    if candidate.status not in {"validated", "rendered", "exported"}:
        raise RuntimeError(
            f"accepted HDF5 conflicts with candidate status {candidate.status!r}"
        )
    if candidate.payload.get("trajectory_sha256") != provenance["trajectory_sha256"]:
        raise RuntimeError("candidate trajectory hash differs from the accepted HDF5")
    if candidate.payload.get("action_fk_report") != provenance["action_fk_report"]:
        raise RuntimeError("candidate action FK report differs from the accepted HDF5")
    return candidate


def _assert_retry_matches(
    candidate: CandidateRecord,
    source_episode_indices: tuple[int, ...],
    generation_payload: dict[str, Any],
) -> None:
    if candidate.source_episode_indices != source_episode_indices:
        raise RuntimeError("deterministic retry selected different source episodes")
    for key in (
        "mimic_generator_success",
        "source_selection_events",
        "acceptance_report",
        "trajectory_diagnostics",
        "physical_randomization",
    ):
        if candidate.payload.get(key) != generation_payload[key]:
            raise RuntimeError(f"deterministic retry changed {key}")


async def _run_audited_generator(
    *,
    env,
    env_id: int,
    reset_queue: asyncio.Queue,
    action_queue: asyncio.Queue,
    generator: AuditedDataGenerator,
    success_term,
    ledger: CandidateLedger,
    run_id: str,
    output_shard: str,
    start_attempt_index: int,
    attempt_count: int,
    base_seed: int,
    config_sha256: str,
    runtime_digest: str,
    source_demo_to_episode: dict[int, int],
    urdf_path: Path,
    action_fk_contract: dict[str, Any],
    counters: GenerationCounters,
) -> None:
    if env_id != 0 or env.num_envs != 1:
        raise ValueError("audited generation currently requires exactly one deterministic environment")
    for offset in range(attempt_count):
        attempt_index = start_attempt_index + offset
        trajectory_seed = base_seed + attempt_index
        candidate_id = f"{run_id}-attempt-{attempt_index:06d}"
        started_at = _utc_now()
        candidate = ledger.ensure_claim(
            CandidateRecord(
                candidate_id=candidate_id,
                status="claimed",
                source_episode_indices=(),
                trajectory_seed=trajectory_seed,
                config_sha256=config_sha256,
                runtime_digest=runtime_digest,
                payload={
                    "run_id": run_id,
                    "attempt_index": attempt_index,
                    "started_at_utc": started_at,
                    "output_shard": output_shard,
                },
            )
        )
        existing_episode = _episode_provenance(
            env,
            attempt_index,
            clean_incomplete=candidate.status in {"claimed", "generated"},
        )
        if existing_episode is not None:
            candidate = _reconcile_exported_episode(
                ledger=ledger, candidate=candidate, provenance=existing_episode
            )
        if candidate.status in {"validated", "rendered", "exported"}:
            counters.attempted += 1
            counters.accepted += 1
            counters.resumed += 1
            continue
        if candidate.status == "rejected":
            counters.attempted += 1
            counters.rejected += 1
            counters.resumed += 1
            continue
        if candidate.status not in {"claimed", "generated"}:
            raise RuntimeError(f"cannot resume candidate in status {candidate.status!r}")

        _seed_attempt(trajectory_seed)
        result = await generator.generate(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=reset_queue,
            env_action_queue=action_queue,
            pause_subtask=False,
            export_demo=False,
            motion_planner=None,
        )
        report = env.get_candidate_acceptance_report(env_id)
        generator_success = bool(result["success"])
        accepted = generator_success and bool(report["passed"])
        if not generator_success and report["passed"]:
            report["rejection_reasons"] = ["mimic_generator_never_observed_success"]
            report["passed"] = False
            accepted = False

        events = result.get("source_selection_events", [])
        selected_demo_indices = sorted({int(event["mimic_demo_index"]) for event in events})
        if not selected_demo_indices:
            raise RuntimeError("Mimic returned no source-demo selection event")
        try:
            source_episode_indices = tuple(
                sorted({source_demo_to_episode[index] for index in selected_demo_indices})
            )
        except KeyError as exc:
            raise RuntimeError(f"Mimic selected unknown source demo {exc.args[0]}") from exc

        generation_payload = {
            "finished_at_utc": _utc_now(),
            "mimic_generator_success": generator_success,
            "source_selection_events": events,
            "acceptance_report": report,
            "trajectory_diagnostics": _trajectory_diagnostics(env, env_id),
            "physical_randomization": snapshot_physical_randomization(env, env_id),
        }
        if candidate.status == "claimed":
            candidate = ledger.transition(
                candidate_id,
                "generated",
                generation_payload,
                source_episode_indices=source_episode_indices,
            )
        else:
            _assert_retry_matches(candidate, source_episode_indices, generation_payload)
            generation_payload = {
                key: candidate.payload[key]
                for key in (
                    "finished_at_utc",
                    "mimic_generator_success",
                    "source_selection_events",
                    "acceptance_report",
                    "trajectory_diagnostics",
                    "physical_randomization",
                )
            }

        env_ids = torch.tensor([env_id], dtype=torch.int64, device=env.device)
        env.recorder_manager.set_success_to_episodes(
            env_ids,
            torch.tensor([[accepted]], dtype=torch.bool, device=env.device),
        )
        env.recorder_manager.export_episodes(env_ids, demo_ids=[attempt_index])
        if accepted:
            action_fk_report = _accepted_action_fk_report(
                env=env,
                attempt_index=attempt_index,
                urdf_path=urdf_path,
                action_fk_contract=action_fk_contract,
            )
            if action_fk_report["pass"] is not True:
                _delete_exported_episode(env, attempt_index)
                ledger.transition(
                    candidate_id,
                    "rejected",
                    {
                        "action_fk_report": action_fk_report,
                        "rejection_reasons": ["synthetic_action_fk_residual_exceeded"],
                    },
                )
                counters.rejected += 1
                counters.attempted += 1
                continue
            trajectory_sha256 = _write_episode_provenance(
                env=env,
                attempt_index=attempt_index,
                candidate=candidate,
                generation_payload=generation_payload,
                action_fk_report=action_fk_report,
            )
            ledger.transition(
                candidate_id,
                "validated",
                {
                    "accepted_hdf5_demo": f"demo_{attempt_index}",
                    "trajectory_sha256": trajectory_sha256,
                    "action_fk_report": action_fk_report,
                },
            )
            counters.accepted += 1
        else:
            ledger.transition(
                candidate_id,
                "rejected",
                {"rejection_reasons": report["rejection_reasons"]},
            )
            counters.rejected += 1
        counters.attempted += 1


def run_generation(
    *,
    env,
    input_file: str | Path,
    success_term,
    ledger: CandidateLedger,
    run_id: str,
    output_shard: str,
    start_attempt_index: int,
    attempt_count: int,
    base_seed: int,
    config_sha256: str,
    runtime_digest: str,
    urdf_path: str | Path,
    action_fk_contract: dict[str, Any],
) -> dict[str, Any]:
    """Execute deterministic attempts using the current official Mimic engine."""

    if env.num_envs != 1:
        raise ValueError("num_envs must be one until cross-environment RNG isolation is proven")
    if attempt_count <= 0:
        raise ValueError("attempt_count must be positive")
    resolved_urdf = Path(urdf_path).resolve()
    if not resolved_urdf.is_file():
        raise FileNotFoundError(resolved_urdf)
    event_loop = asyncio.get_event_loop()
    reset_queue = asyncio.Queue()
    action_queue = asyncio.Queue()
    info_pool = DataGenInfoPool(env, env.cfg, env.device)
    info_pool.load_from_dataset_file(str(input_file))
    generator = AuditedDataGenerator(env=env, src_demo_datagen_info_pool=info_pool)
    counters = GenerationCounters()
    task = event_loop.create_task(
        _run_audited_generator(
            env=env,
            env_id=0,
            reset_queue=reset_queue,
            action_queue=action_queue,
            generator=generator,
            success_term=success_term,
            ledger=ledger,
            run_id=run_id,
            output_shard=output_shard,
            start_attempt_index=start_attempt_index,
            attempt_count=attempt_count,
            base_seed=base_seed,
            config_sha256=config_sha256,
            runtime_digest=runtime_digest,
            source_demo_to_episode=source_demo_episode_indices(input_file),
            urdf_path=resolved_urdf,
            action_fk_contract=action_fk_contract,
            counters=counters,
        )
    )
    gathered = asyncio.ensure_future(asyncio.gather(task))
    try:
        env_loop(
            env,
            reset_queue,
            action_queue,
            info_pool,
            event_loop,
            data_gen_tasks=gathered,
        )
        event_loop.run_until_complete(gathered)
    finally:
        if not gathered.done():
            gathered.cancel()
            with suppress(asyncio.CancelledError):
                event_loop.run_until_complete(gathered)
    return {
        "attempted": counters.attempted,
        "accepted": counters.accepted,
        "rejected": counters.rejected,
        "resumed": counters.resumed,
        "acceptance_rate": counters.accepted / counters.attempted,
        "source_demo_count": info_pool.num_datagen_infos,
    }
