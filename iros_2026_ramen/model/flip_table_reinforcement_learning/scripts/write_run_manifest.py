#!/usr/bin/env python3
"""Write a reproducibility manifest before a flip-table experiment starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone


ENV_PREFIXES = (
    "FLIP_TABLE_",
    "ROBOFINALS_",
    "NVIDIA_",
    "OMNI_KIT_",
)
IGNORED_TREE_PARTS = {"__pycache__", ".pytest_cache", "outputs"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    count = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path)
        if any(part in IGNORED_TREE_PARTS for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        count += 1
    return {"path": str(path.resolve()), "file_count": count, "sha256": digest.hexdigest()}


def _command(*command: str, cwd: Path | None = None) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc)}
    output = result.stdout.strip() or result.stderr.strip()
    return {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "output": output,
    }


def _input_record(raw: str) -> tuple[str, dict[str, object]]:
    if "=" not in raw:
        raise ValueError(f"--input must be LABEL=PATH, got {raw!r}")
    label, raw_path = raw.split("=", 1)
    path = Path(raw_path).expanduser()
    record: dict[str, object] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _sha256_file(path),
            }
        )
    return label, record


def _git_state(repository: Path) -> dict[str, object]:
    commit = _command("git", "rev-parse", "HEAD", cwd=repository)
    branch = _command("git", "branch", "--show-current", cwd=repository)
    status = _command("git", "status", "--short", cwd=repository)
    diff = _command("git", "diff", "--binary", "HEAD", "--", ".", cwd=repository)
    diff_output = str(diff.get("output", ""))
    return {
        "commit": commit.get("output") if commit.get("available") else None,
        "branch": branch.get("output") if branch.get("available") else None,
        "status_short": status.get("output", ""),
        "tracked_diff_sha256": hashlib.sha256(diff_output.encode("utf-8")).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--policy-mode", required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--source-root", action="append", type=Path, default=[])
    args = parser.parse_args()

    inputs = dict(_input_record(value) for value in args.input)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "stage": args.stage,
        "policy_mode": args.policy_mode,
        "working_directory": str(Path.cwd()),
        "git": _git_state(args.repository.resolve()),
        "source_trees": [_tree_digest(path.resolve()) for path in args.source_root],
        "inputs": inputs,
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(ENV_PREFIXES)
        },
        "hardware": {
            "gpu": _command(
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ),
            "cpu": _command("lscpu"),
            "memory": _command("free", "-b"),
            "storage": _command("df", "-B1", str(args.repository.resolve())),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)
    print(f"[flip_table_rl] run manifest: {args.output}")


if __name__ == "__main__":
    main()
