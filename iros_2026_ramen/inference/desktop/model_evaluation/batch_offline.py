"""Batch audit and offline-only probing for every model in an HF namespace."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from .cli import adapter_dry_run
from .inferred import audit_namespace, infer_offline_contract
from .inferred_artifacts import prepare, validate
from .inferred_offline import run_inferred_offline
from .registry import load_registry


def test_namespace_offline(
    *,
    namespace: str,
    workspace: Path,
    device: str,
    prepare_missing: bool,
    max_download_bytes: int,
) -> dict[str, Any]:
    """Probe all eligible models without importing or initializing robot I/O."""
    audit = audit_namespace(namespace)
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    registry_by_repo = {spec.repo_id: spec for spec in load_registry().values()}
    results: list[dict[str, Any]] = []
    for item in audit["models"]:
        repo_id = str(item["repo_id"])
        started = time.monotonic()
        result: dict[str, Any] = {
            "repo_id": repo_id,
            "revision": item.get("revision"),
            "category": item["category"],
            "robot_command_sent": False,
            "dds_initialized": False,
            "actuation_allowed": False,
        }
        try:
            if item["category"] == "registered_physical":
                spec = registry_by_repo[repo_id]
                result.update(
                    {
                        "status": "registered_adapter_passed",
                        "test_level": "adapter_dimensions_only",
                        "report": adapter_dry_run(spec),
                    }
                )
            elif not item.get("weight_load_supported"):
                result.update(
                    {
                        "status": "structure_only",
                        "test_level": "metadata",
                        "issues": item.get("issues", []),
                    }
                )
            elif int(item.get("total_download_bytes", 0)) > max_download_bytes:
                result.update(
                    {
                        "status": "skipped_download_limit",
                        "test_level": "metadata_and_contract",
                        "required_download_bytes": item["total_download_bytes"],
                        "download_limit_bytes": max_download_bytes,
                    }
                )
            else:
                contract = infer_offline_contract(
                    repo_id, revision=str(item["revision"])
                )
                local_dir = workspace / _safe_name(repo_id)
                lock = local_dir / ".iros_ramen_inferred_offline_lock.json"
                if lock.is_file():
                    validate(local_dir, expected=contract)
                elif prepare_missing:
                    prepare(contract, local_dir)
                else:
                    result.update(
                        {
                            "status": "not_prepared",
                            "test_level": "metadata_and_contract",
                            "local_dir": str(local_dir),
                        }
                    )
                    results.append(result)
                    continue
                report = run_inferred_offline(local_dir, device=device)
                result.update(
                    {
                        "status": "weight_inference_passed",
                        "test_level": "synthetic_weight_inference",
                        "local_dir": str(local_dir),
                        "report": report,
                    }
                )
        except Exception as exc:
            result.update(
                {
                    "status": "weight_inference_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        result["elapsed_s"] = time.monotonic() - started
        results.append(result)

    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": "team_ramen_namespace_offline_test/v1",
        "namespace": namespace,
        "model_count": len(results),
        "status_counts": counts,
        "prepare_missing": prepare_missing,
        "max_download_bytes": max_download_bytes,
        "device": device,
        "safety": {
            "robot_command_sent": False,
            "dds_initialized": False,
            "actuation_allowed": False,
            "live_camera_opened": False,
        },
        "results": results,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _safe_name(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)
