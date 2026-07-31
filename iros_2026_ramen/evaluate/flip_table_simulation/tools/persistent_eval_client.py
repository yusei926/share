#!/usr/bin/env python3
"""Submit a bounded evaluation job to a persistent Isaac worker."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import shutil
import time
import uuid


SCHEMA_VERSION = "team_ramen_persistent_evaluation_job/v1"
ENV_LINE = re.compile(r"^export ([A-Z0-9_]+)=(.*)$")
ENV_ASSIGNMENT = re.compile(r"^([A-Z0-9_]+)=(.*)$")
SUPPORTED_POLICIES = {
    "RecordedJointTargetPolicy",
    "RecordedFullBodyTargetPolicy",
    "CvRuleBasedPolicy",
    "ScriptedJointPolicy",
    "AvpTeleopPolicy",
    "TeleopPerformanceBenchmarkPolicy",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-env", type=Path)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-name", choices=sorted(SUPPORTED_POLICIES))
    parser.add_argument("--time-out-limit", type=int)
    parser.add_argument(
        "--environment",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add one allowlisted worker environment value; may be repeated.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    return parser.parse_args()


def _read_generated_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = ENV_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"runtime env contains an unsupported line: {line!r}")
        key, quoted = match.groups()
        try:
            value = ast.literal_eval(quoted)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"runtime env value for {key} is invalid") from exc
        if not isinstance(value, str):
            raise ValueError(f"runtime env value for {key} must be a string")
        values[key] = value
    return values


def _relative_to(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{path} must be inside persistent root {root}") from exc


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment_overrides(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        match = ENV_ASSIGNMENT.fullmatch(value)
        if match is None:
            raise ValueError(f"environment override must be KEY=VALUE: {value!r}")
        key, setting = match.groups()
        parsed[key] = setting
    return parsed


def main() -> int:
    args = _args()
    if args.seed < 0 or args.timeout_s <= 0:
        raise ValueError("seed must be non-negative and timeout must be positive")
    root = args.persistent_root.resolve()
    output_dir = args.output_dir.resolve()
    output_relpath = _relative_to(root, output_dir)
    ready = root / "persistent_jobs" / "ready.json"
    if not ready.is_file():
        raise RuntimeError("persistent Isaac worker is not ready; run persistent_eval.sh start first")
    ready_payload = json.loads(ready.read_text(encoding="utf-8"))
    if ready_payload.get("state") not in {"ready", "running"}:
        raise RuntimeError(f"persistent Isaac worker has invalid state: {ready_payload.get('state')!r}")

    environment = (
        _read_generated_environment(args.runtime_env.resolve()) if args.runtime_env is not None else {}
    )
    environment.update(_environment_overrides(args.environment))
    policy_name = args.policy_name or environment.get("FLIP_TABLE_POLICY_NAME")
    if policy_name not in SUPPORTED_POLICIES:
        raise ValueError("--policy-name is required unless runtime-env supplies a supported policy")
    job_id = f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    replay_path = environment.get("FLIP_TABLE_REPLAY_ACTION_PATH")
    if replay_path is not None:
        action_path = Path(replay_path).resolve()
        if not action_path.is_file():
            raise FileNotFoundError(action_path)
        input_dir = root / "persistent_jobs" / "inputs" / job_id
        input_dir.mkdir(parents=True, exist_ok=False)
        copied_action = input_dir / "replay_actions.json"
        shutil.copy2(action_path, copied_action)
        environment["FLIP_TABLE_REPLAY_ACTION_PATH"] = _relative_to(root, copied_action)
    timeout_value = args.time_out_limit or environment.pop("FLIP_TABLE_TIME_OUT_LIMIT", None)
    if timeout_value is None:
        raise ValueError("--time-out-limit is required unless runtime-env provides it")
    timeout_steps = int(timeout_value)
    if timeout_steps < 1:
        raise ValueError("time-out-limit must be positive")
    environment.pop("FLIP_TABLE_SIM_OUTPUT_DIR", None)
    environment.pop("FLIP_TABLE_POLICY_NAME", None)
    environment.pop("FLIP_TABLE_TEST_NUM", None)
    job = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "policy_name": policy_name,
        "seed": args.seed,
        "time_out_limit": timeout_steps,
        "output_relpath": output_relpath,
        "environment": environment,
    }
    _atomic_json(root / "persistent_jobs" / "queue" / f"{job_id}.job.json", job)
    print(json.dumps({"job_id": job_id, "state": "queued", "output_dir": str(output_dir)}, indent=2))
    if not args.wait:
        return 0
    deadline = time.monotonic() + args.timeout_s
    completed = root / "persistent_jobs" / "completed" / f"{job_id}.job.json"
    failed = root / "persistent_jobs" / "failed" / f"{job_id}.job.json"
    while time.monotonic() < deadline:
        if completed.is_file():
            print(completed.read_text(encoding="utf-8"))
            return 0
        if failed.is_file():
            raise RuntimeError(failed.read_text(encoding="utf-8"))
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for persistent job {job_id}")


if __name__ == "__main__":
    raise SystemExit(main())
