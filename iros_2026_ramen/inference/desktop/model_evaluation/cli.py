"""Resolve, validate, and dry-run heterogeneous physical-G1 policies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys
from typing import Mapping
import warnings

import numpy as np

from .adapters import CanonicalObservation, adapter_for
from .artifacts import (
    download_plan,
    seal_local_artifacts,
    validate_prepared_artifacts,
)
from .registry import ModelSpec, load_registry
from .resolver import (
    UnsupportedModelError,
    inspect_hf_model,
    onboarding_manifest_draft,
    resolve_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    for name in ("resolve", "inspect"):
        command = subparsers.add_parser(name)
        _reference_args(command)
    inspect_hf = subparsers.add_parser("inspect-hf")
    _reference_args(inspect_hf)
    audit_team = subparsers.add_parser(
        "audit-team",
        help="Audit every model in an HF namespace without downloading weights.",
    )
    audit_team.add_argument("--namespace", default="Team-RAMEN")
    audit_team.add_argument("--output", type=Path)
    infer_offline = subparsers.add_parser(
        "infer-offline",
        help="Infer an offline-only contract; this can never enable actuation.",
    )
    _reference_args(infer_offline)
    for name in (
        "offline-download-plan",
        "offline-prepare",
        "offline-validate",
    ):
        command = subparsers.add_parser(name)
        _reference_args(command)
        command.add_argument("--local-dir", type=Path, required=True)
    inferred_run = subparsers.add_parser("inferred-model-dry-run")
    inferred_run.add_argument("--local-dir", type=Path, required=True)
    inferred_run.add_argument("--device", default="cuda:0")
    inferred_run.add_argument("--seed", type=int, default=42)
    inferred_run.add_argument(
        "--task", default="perform the demonstrated manipulation"
    )
    test_team = subparsers.add_parser(
        "test-team-offline",
        help="Audit and safely probe all eligible namespace models without robot I/O.",
    )
    test_team.add_argument("--namespace", default="Team-RAMEN")
    test_team.add_argument("--workspace", type=Path, required=True)
    test_team.add_argument("--output", type=Path, required=True)
    test_team.add_argument("--device", default="cuda:0")
    test_team.add_argument("--prepare", action="store_true")
    test_team.add_argument("--max-download-gb", type=float, default=1.0)
    onboard = subparsers.add_parser("onboard")
    _reference_args(onboard)
    onboard.add_argument("--output", type=Path)
    for name in ("download-plan", "prepare", "download", "seal", "validate-artifacts"):
        command = subparsers.add_parser(name)
        _reference_args(command)
        command.add_argument("--local-dir", type=Path, required=True)
    for name in ("adapter-dry-run", "dry-run"):
        command = subparsers.add_parser(name)
        _reference_args(command)
    offline = subparsers.add_parser("offline-model-dry-run")
    _reference_args(offline)
    offline.add_argument("--local-dir", type=Path, required=True)
    offline.add_argument("--bundle", type=Path, required=True)
    offline.add_argument("--device", default="cuda:0")
    offline.add_argument("--seed", type=int, default=42)
    real = subparsers.add_parser("real-command")
    _reference_args(real)
    real.add_argument("--local-dir", type=Path, required=True)
    real.add_argument("--actuate", action="store_true")
    real.add_argument("--max-seconds", type=float)
    real.add_argument("--device")
    return parser.parse_args()


def _reference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Local id, owner/repo, or huggingface.co URL")
    parser.add_argument(
        "--revision",
        help="HF branch/tag/commit to resolve; execution always records the full SHA.",
    )


def main() -> int:
    args = parse_args()
    if args.command == "list":
        for model_id, spec in load_registry().items():
            print(
                f"{model_id}\t{spec.family}\t{spec.repo_id}@{spec.revision[:12]}"
            )
        return 0
    if args.command == "inspect-hf":
        print(
            json.dumps(
                inspect_hf_model(args.model, revision=args.revision),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "audit-team":
        from .inferred import audit_namespace

        report = audit_namespace(args.namespace)
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        return 0
    if args.command == "infer-offline":
        from .inferred import infer_offline_contract

        contract = infer_offline_contract(args.model, revision=args.revision)
        print(json.dumps(contract.to_mapping(), ensure_ascii=False, indent=2))
        return 0
    if args.command in {
        "offline-download-plan",
        "offline-prepare",
        "offline-validate",
    }:
        from .inferred import infer_offline_contract
        from .inferred_artifacts import (
            download_plan as inferred_download_plan,
            prepare as inferred_prepare,
            validate as validate_inferred,
        )

        contract = infer_offline_contract(args.model, revision=args.revision)
        if args.command == "offline-download-plan":
            result = inferred_download_plan(contract, args.local_dir)
        elif args.command == "offline-prepare":
            result = inferred_prepare(contract, args.local_dir)
        else:
            result = validate_inferred(args.local_dir, expected=contract)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "inferred-model-dry-run":
        from .inferred_offline import run_inferred_offline

        print(
            json.dumps(
                run_inferred_offline(
                    args.local_dir,
                    device=args.device,
                    seed=args.seed,
                    task=args.task,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "test-team-offline":
        from .batch_offline import test_namespace_offline, write_report

        if not np.isfinite(args.max_download_gb) or args.max_download_gb < 0:
            raise ValueError("--max-download-gb must be finite and non-negative")
        report = test_namespace_offline(
            namespace=args.namespace,
            workspace=args.workspace,
            device=args.device,
            prepare_missing=args.prepare,
            max_download_bytes=int(args.max_download_gb * 1_000_000_000),
        )
        write_report(report, args.output)
        print(json.dumps(report["status_counts"], ensure_ascii=False, indent=2))
        print(args.output.expanduser().resolve())
        return 0
    if args.command == "onboard":
        draft = onboarding_manifest_draft(args.model, revision=args.revision)
        text = json.dumps(draft, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(text, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        return 0

    resolved = resolve_model(
        args.model,
        revision=args.revision,
        allow_network=("/" in args.model or args.revision is not None),
    )
    spec = resolved.spec
    if args.command == "resolve":
        print(json.dumps(resolved.to_mapping(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "inspect":
        result = asdict(spec)
        result["canonical_output"] = spec.canonical_output
        result["lower_body_command_dimensions"] = 0
        result["resolution_source"] = resolved.resolution_source
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "download-plan":
        print(
            json.dumps(
                download_plan(spec, args.local_dir), ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.command in {"prepare", "download"}:
        if args.command == "download":
            warnings.warn("`download` is deprecated; use `prepare`", stacklevel=1)
        from huggingface_hub import snapshot_download

        snapshot_download(**download_plan(spec, args.local_dir))
        print(
            json.dumps(
                seal_local_artifacts(args.local_dir, spec),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "seal":
        print(
            json.dumps(
                seal_local_artifacts(args.local_dir, spec),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "validate-artifacts":
        print(
            json.dumps(
                validate_prepared_artifacts(args.local_dir, spec),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "real-command":
        validate_prepared_artifacts(args.local_dir, spec)
        print(
            real_command(
                spec,
                args.local_dir,
                actuate=args.actuate,
                max_seconds=args.max_seconds,
                device=args.device,
            )
        )
        return 0
    if args.command == "offline-model-dry-run":
        from .offline_model import run_offline_model

        print(
            json.dumps(
                run_offline_model(
                    spec,
                    local_dir=args.local_dir,
                    bundle=args.bundle,
                    device=args.device,
                    seed=args.seed,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command in {"adapter-dry-run", "dry-run"}:
        if args.command == "dry-run":
            warnings.warn(
                "`dry-run` checks only the adapter; use `adapter-dry-run`",
                stacklevel=1,
            )
        print(
            json.dumps(
                adapter_dry_run(spec),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise AssertionError(args.command)


def real_command(
    spec: ModelSpec,
    local_dir: Path,
    *,
    actuate: bool = False,
    max_seconds: float | None = None,
    device: str | None = None,
) -> str:
    """Build but never execute the sealed common launcher command."""
    argv = [
        "pixi",
        "run",
        "-e",
        "model-eval",
        "python",
        "-m",
        "inference.desktop.model_evaluation.launch",
        spec.repo_id,
        "--revision",
        spec.revision,
        "--local-dir",
        str(local_dir),
    ]
    if max_seconds is not None:
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("--max-seconds must be finite and positive")
        argv += ["--max-seconds", str(max_seconds)]
    if device is not None:
        argv += ["--device", device]
    if actuate:
        argv.append("--actuate")
    return " ".join(shlex.quote(value) for value in argv)


def runner_argv(
    spec: ModelSpec,
    local_dir: Path,
    *,
    interface: str,
    image_server_ip: str,
    actuate: bool = False,
    max_seconds: float | None = None,
    device: str | None = None,
    safety_limits: Mapping[str, float] | None = None,
    log_path: Path | None = None,
) -> list[str]:
    """Build model-specific argv from trusted local family data only."""
    from .artifacts import checkpoint_path

    checkpoint = checkpoint_path(local_dir, spec)
    argv = [
        "python",
        "-m",
        spec.runner,
        "--interface",
        interface,
        "--image-server-ip",
        image_server_ip,
        "--checkpoint",
        str(checkpoint),
        "--worker-script",
        spec.worker,
        "--task",
        spec.task,
        "--model-repo-id",
        spec.repo_id,
        "--model-revision",
        spec.revision,
    ]
    if spec.expected_model_sha256 is not None:
        argv += ["--expected-checkpoint-sha256", spec.expected_model_sha256]
    if max_seconds is not None:
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("--max-seconds must be finite and positive")
        argv += ["--max-seconds", str(max_seconds)]
    if device is not None:
        argv += ["--device", device]
    if log_path is not None:
        argv += ["--log", str(log_path)]
    # Chunk execution is part of the sealed model contract, not an operator
    # tuning knob.  ACT checkpoints here declare n_action_steps=30; forwarding
    # the registry value prevents the runner from silently replanning after a
    # legacy 8-step prefix. Other families keep their own reviewed defaults.
    if spec.family in {
        "act_absolute_joint16_v1",
        "groot_absolute_joint_v1",
        "groot_relative_eef_v1",
    }:
        argv += ["--action-execution-steps", str(spec.execution_steps)]
    # Only these model-independent motion limits may cross the sealed common
    # launcher boundary. The upper bounds are the highest reviewed defaults
    # used by any registered real runner; arbitrary runner arguments and
    # unsafe larger values remain impossible through this entry point.
    safety_bounds = {
        "pre_motion_arm_velocity_rad_s": (
            "--pre-motion-arm-velocity-rad-s", 0.05, 0.5
        ),
        "pre_motion_arm_acceleration_rad_s2": (
            "--pre-motion-arm-acceleration-rad-s2", 0.1, 1.0
        ),
        "pre_motion_waypoint_tolerance_rad": (
            "--pre-motion-waypoint-tolerance-rad", 0.01, 0.10
        ),
        "pre_motion_stage_timeout_s": (
            "--pre-motion-stage-timeout-s", 1.0, 30.0
        ),
        "policy_arm_velocity_rad_s": (
            "--policy-arm-velocity-rad-s", 0.05, 1.0
        ),
        "policy_arm_acceleration_rad_s2": (
            "--policy-arm-acceleration-rad-s2", 0.1, 4.0
        ),
        "policy_hand_velocity_fraction_s": (
            "--policy-hand-velocity-fraction-s", 0.05, 1.0
        ),
        "policy_hand_acceleration_fraction_s2": (
            "--policy-hand-acceleration-fraction-s2", 0.1, 4.0
        ),
    }
    overrides = dict(safety_limits or {})
    unknown = set(overrides) - set(safety_bounds)
    if unknown:
        raise ValueError(f"unsupported safety limit overrides: {sorted(unknown)}")
    for name, (flag, minimum, maximum) in safety_bounds.items():
        if name not in overrides:
            continue
        value = float(overrides[name])
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(
                f"{flag} must be finite in the reviewed range "
                f"[{minimum},{maximum}]"
            )
        argv += [flag, str(value)]
    if actuate:
        argv.append("--actuate")
    return argv


def adapter_dry_run(spec: ModelSpec) -> dict[str, object]:
    """Exercise dimensions and mappings without importing physical transports."""
    modules_before = set(sys.modules)
    adapter = adapter_for(spec)
    cameras = {role: f"synthetic-{role}".encode() for role in spec.camera_roles}
    observation = CanonicalObservation(
        body_joint_position_rad=np.zeros(29, dtype=np.float64),
        dex1_opening_fraction=np.asarray([0.25, 0.75]),
        eef_xyz_euler=np.zeros(12, dtype=np.float64),
        camera_jpeg=cameras,
    )
    state = adapter.model_state(observation)
    native = adapter.synthetic_native_action()
    action = adapter.canonical_action(native, observation)
    forbidden = sorted(
        name
        for name in set(sys.modules) - modules_before
        if name.startswith(("unitree_sdk2py", "cyclonedds"))
    )
    if forbidden:
        raise RuntimeError(f"adapter dry-run imported physical transports: {forbidden}")
    return {
        "model_id": spec.model_id,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "family": spec.family,
        "state_shape": list(state.shape),
        "native_action_shape": list(native.shape),
        "canonical_action_shape": list(action.shape),
        "canonical_output": spec.canonical_output,
        "lower_body_command_dimensions": 0,
        "model_weights_loaded": False,
        "robot_command_sent": False,
        "dds_initialized": False,
        "physical_transport_imported": False,
    }


def cli_entry() -> int:
    try:
        return main()
    except (
        FileNotFoundError,
        KeyError,
        UnsupportedModelError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_entry())
