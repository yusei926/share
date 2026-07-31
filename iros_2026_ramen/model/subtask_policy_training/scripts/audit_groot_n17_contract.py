#!/usr/bin/env python3
"""Validate the pinned GR00T N1.7 contract and Dex1 synergy adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gr00t.dex1_hand_synergy import (  # noqa: E402
    dex1_to_hand,
    hand_to_dex1,
    load_synergy_manifest,
    synergy_manifest_sha256,
)
from gr00t.n17_contract import (  # noqa: E402
    BASE_MODEL_REPO_ID,
    BASE_MODEL_REVISION,
    validate_checkpoint_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Optional raw flip-table dataset used to audit mapped Dex1 hand distributions.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.checkpoint_root
    if root is None:
        root = Path(
            snapshot_download(
                BASE_MODEL_REPO_ID,
                revision=BASE_MODEL_REVISION,
                allow_patterns=[
                    "config.json",
                    "processor_config.json",
                    "statistics.json",
                    "embodiment_id.json",
                ],
            )
        )
    report = validate_checkpoint_contract(root)
    report["dex1_synergy_sha256"] = synergy_manifest_sha256()
    report["dex1_roundtrip_max_abs_error"] = validate_dex1_roundtrip()
    if args.dataset_root is not None:
        report["dex1_official_q01_q99_audit"] = audit_dex1_official_ranges(
            args.dataset_root,
            root / "statistics.json",
        )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)


def validate_dex1_roundtrip() -> float:
    load_synergy_manifest()
    errors: list[float] = []
    for side in ("left", "right"):
        for kind in ("state", "action"):
            for index in range(101):
                source = 4.5 * index / 100
                hand = dex1_to_hand(source, side=side, kind=kind)
                recovered = hand_to_dex1(hand, side=side, kind=kind)
                errors.append(abs(source - recovered))
    maximum = max(errors)
    if maximum > 1e-4:
        raise ValueError(f"Dex1 synergy roundtrip error {maximum} exceeds 1e-4")
    return maximum


def audit_dex1_official_ranges(
    dataset_root: Path,
    statistics_path: Path,
    *,
    maximum_outside_fraction: float = 0.05,
) -> dict[str, object]:
    """Verify every mapped Dex1 dimension against the pinned official hand range."""
    parquet_paths = sorted(dataset_root.glob("data/**/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no source parquet files under {dataset_root}")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))[
        "real_g1_relative_eef_relative_joints"
    ]
    mapped: dict[tuple[str, str], list[np.ndarray]] = {
        (kind, side): []
        for kind in ("state", "action")
        for side in ("left", "right")
    }
    frame_count = 0
    for path in parquet_paths:
        table = pq.read_table(
            path,
            columns=["observation.state.hand_state", "action.hand_cmd"],
        )
        frame_count += len(table)
        for kind, column in (
            ("state", "observation.state.hand_state"),
            ("action", "action.hand_cmd"),
        ):
            dex1 = np.asarray(table[column].to_pylist(), dtype=np.float64)
            if dex1.ndim != 2 or dex1.shape[1] != 2 or not np.isfinite(dex1).all():
                raise ValueError(f"{column} must contain finite [N,2] Dex1 values")
            for index, side in enumerate(("left", "right")):
                mapped[(kind, side)].append(
                    np.asarray(
                        [
                            dex1_to_hand(value, side=side, kind=kind)
                            for value in dex1[:, index]
                        ],
                        dtype=np.float64,
                    )
                )

    groups: dict[str, object] = {}
    for (kind, side), chunks in mapped.items():
        values = np.concatenate(chunks, axis=0)
        official = statistics[kind][f"{side}_hand"]
        q01 = np.asarray(official["q01"], dtype=np.float64)
        q99 = np.asarray(official["q99"], dtype=np.float64)
        outside = np.mean((values < q01) | (values > q99), axis=0)
        if np.any(outside > maximum_outside_fraction):
            raise ValueError(
                f"{kind}.{side}_hand mapped range exceeds the "
                f"{maximum_outside_fraction:.1%} limit: {outside.tolist()}"
            )
        groups[f"{kind}.{side}_hand"] = {
            "outside_fraction_per_dimension": outside.tolist(),
            "maximum_outside_fraction": float(np.max(outside)),
            "sample_count": int(len(values)),
        }
    return {
        "dataset_root": str(dataset_root.resolve()),
        "frame_count": frame_count,
        "threshold": maximum_outside_fraction,
        "clipping_used": False,
        "groups": groups,
    }


if __name__ == "__main__":
    main()
