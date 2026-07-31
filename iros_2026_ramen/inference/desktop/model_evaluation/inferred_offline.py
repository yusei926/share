"""Launch the isolated inferred-model probe and verify its safety report."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from .inferred_artifacts import validate


REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PYTHON = REPO_ROOT / "model/subtask_policy_training/.venv/bin/python"
WORKER = (
    REPO_ROOT
    / "model/subtask_policy_training/scripts/inferred_offline_probe.py"
)


def run_inferred_offline(
    local_dir: Path,
    *,
    device: str = "cuda:0",
    seed: int = 42,
    task: str = "perform the demonstrated manipulation",
) -> dict[str, Any]:
    """Run one synthetic inference; never initialize live hardware transports."""
    validate(local_dir)
    if not MODEL_PYTHON.is_file():
        raise FileNotFoundError(
            "model/subtask_policy_training/.venv is unavailable"
        )
    result = subprocess.run(
        [
            str(MODEL_PYTHON),
            str(WORKER),
            "--local-dir",
            str(local_dir.expanduser().resolve()),
            "--device",
            device,
            "--seed",
            str(seed),
            "--task",
            task,
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "inferred offline probe failed "
            f"(exit={result.returncode}):\n{result.stderr[-8000:]}"
        )
    marker = "__IROS_RAMEN_OFFLINE_REPORT__"
    report_lines = [
        line.removeprefix(marker)
        for line in result.stdout.splitlines()
        if line.startswith(marker)
    ]
    try:
        if len(report_lines) != 1:
            raise json.JSONDecodeError("missing unique report marker", "", 0)
        report = json.loads(report_lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"inferred offline probe returned invalid JSON:\n{result.stdout[-4000:]}"
        ) from exc
    required_false = (
        "actuation_allowed",
        "robot_command_sent",
        "dds_initialized",
        "physical_transport_imported",
        "physical_mapping_verified",
    )
    changed = [key for key in required_false if report.get(key) is not False]
    if changed:
        raise RuntimeError(f"offline safety report is not fail-closed: {changed}")
    return report
