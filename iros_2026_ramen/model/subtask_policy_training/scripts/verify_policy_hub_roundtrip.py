"""Clean-download a private policy and repeat its offline chunk-reset evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

try:
    from .upload_policy import validate_model_dir
except ImportError:  # Executed directly by the shell release runner.
    from upload_policy import validate_model_dir


_NONDETERMINISTIC_METRICS = {
    "chunk_inference_ms_mean",
    "chunk_inference_ms_p95",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_offline_evaluation_equivalence(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-5,
) -> None:
    """Require a clean Hub download to reproduce every policy-quality metric."""
    for key in (
        "schema_version",
        "evaluation_type",
        "model_safetensors_sha256",
        "episodes",
        "declared_split",
        "execution_steps",
        "randomness",
        "contract",
    ):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"roundtrip evaluation changed {key}")
    for key in ("aggregate", "orientation_group_report", "episodes_report"):
        _assert_metric_tree_close(
            expected.get(key),
            actual.get(key),
            path=key,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )


def _assert_metric_tree_close(
    expected: Any,
    actual: Any,
    *,
    path: str,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"roundtrip evaluation changed type at {path}")
        expected_keys = set(expected) - _NONDETERMINISTIC_METRICS
        actual_keys = set(actual) - _NONDETERMINISTIC_METRICS
        if expected_keys != actual_keys:
            raise ValueError(
                f"roundtrip evaluation changed keys at {path}: "
                f"{sorted(expected_keys)} != {sorted(actual_keys)}"
            )
        for key in sorted(expected_keys):
            _assert_metric_tree_close(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"roundtrip evaluation changed list at {path}")
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _assert_metric_tree_close(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        return
    if (
        isinstance(expected, (int, float))
        and not isinstance(expected, bool)
        and isinstance(actual, (int, float))
        and not isinstance(actual, bool)
    ):
        if not (
            math.isfinite(float(expected))
            and math.isfinite(float(actual))
            and math.isclose(
                float(expected),
                float(actual),
                abs_tol=absolute_tolerance,
                rel_tol=relative_tolerance,
            )
        ):
            raise ValueError(
                f"roundtrip evaluation changed metric at {path}: "
                f"{expected!r} != {actual!r}"
            )
        return
    if expected != actual:
        raise ValueError(
            f"roundtrip evaluation changed value at {path}: "
            f"{expected!r} != {actual!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--upload-receipt",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    api = HfApi()
    info = api.model_info(args.repo_id)
    if not info.private:
        raise RuntimeError(f"refusing to verify public model repository {args.repo_id}")
    checkpoint_revision = str(info.sha)
    root = args.output_root.resolve()
    if root.exists():
        shutil.rmtree(root)
    model_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="model",
            revision=checkpoint_revision,
            local_dir=root / "model",
        )
    )
    validate_model_dir(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("type") == "furniture_groot":
        evaluator = Path(__file__).with_name("evaluate_groot_n17_offline.py")
    else:
        evaluator = Path(__file__).with_name("evaluate_delta_chunk_reset.py")
    subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--model-dir",
            str(model_dir),
            "--dataset-root",
            str(args.dataset_root.resolve()),
            "--episodes",
            args.episodes,
            "--output-dir",
            str(root / "evaluation"),
        ],
        check=True,
    )
    evaluation_report = root / "evaluation" / "report.json"
    expected_evaluation = json.loads(
        (model_dir / "evaluation_report.json").read_text(encoding="utf-8")
    )
    roundtrip_evaluation = json.loads(
        evaluation_report.read_text(encoding="utf-8")
    )
    validate_offline_evaluation_equivalence(
        expected_evaluation,
        roundtrip_evaluation,
    )
    training_manifest = json.loads(
        (model_dir / "training_manifest.json").read_text(encoding="utf-8")
    )
    receipt = {
        "schema_version": "groot_n17_hf_roundtrip_v1",
        "repo_id": args.repo_id,
        "checkpoint_revision": checkpoint_revision,
        "model_safetensors_sha256": sha256(model_dir / "model.safetensors"),
        "recorded_model_safetensors_sha256": (
            training_manifest.get("checkpoint") or {}
        ).get("model_safetensors_sha256"),
        "recorded_evaluation_report_sha256": sha256(
            model_dir / "evaluation_report.json"
        ),
        "evaluation_report_sha256": sha256(evaluation_report),
        "evaluation_metrics_reproduced": True,
        "episodes": [int(value) for value in args.episodes.split(",")],
        "clean_snapshot_path": str(model_dir),
    }
    if (
        receipt["model_safetensors_sha256"]
        != receipt["recorded_model_safetensors_sha256"]
    ):
        raise ValueError("clean-downloaded model hash differs from its training manifest")
    receipt_path = root / "hub_roundtrip_manifest.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.upload_receipt:
        commit = api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(receipt_path),
            path_in_repo=receipt_path.name,
            commit_message="Add clean-download GR00T evaluation receipt",
        )
        receipt["receipt_commit"] = commit.oid
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"checkpoint_revision: {checkpoint_revision}")
    print(f"roundtrip_receipt: {receipt_path}")


if __name__ == "__main__":
    main()
