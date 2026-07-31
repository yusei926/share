#!/usr/bin/env python3
"""Rank one shared arm actuator profile from fixed calibration reports.

The selector is deliberately an offline diagnostic.  It consumes reports from
source intervals whose visual/static eligibility is recorded separately.  It
never modifies a replay, task, or policy input.  It prevents a profile from
being chosen because it happened to fit only one demonstration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from data.flip_table_data_augmentation.io_utils import atomic_write_json

MAX_VISUAL_STATIC_TRANSLATION_P95_M = 0.005
MAX_VISUAL_STATIC_ROTATION_P95_DEG = 0.75


def _finite(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_report(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "team_ramen_flip_table_actuator_identification/v1":
        raise ValueError(f"{path} is not an actuator-identification report")
    interval = document.get("source_frame_interval")
    if not isinstance(interval, dict):
        raise ValueError(f"{path} does not declare its source interval")
    match = document.get("sim_to_real_encoder_match")
    real_arms = document.get("real", {}).get("group_summaries", {}).get("arms")
    sim_arms = document.get("sim", {}).get("group_summaries", {}).get("arms")
    if not isinstance(match, dict) or not isinstance(real_arms, dict) or not isinstance(sim_arms, dict):
        raise ValueError(f"{path} is missing arm response metrics")
    return {
        "path": str(path),
        "episode_index": int(document["source_episode_index"]),
        "interval": interval,
        "samples": int(match["samples"]),
        "q_rmse_rad": _finite(match["upper_body_rmse_rad"], f"{path} q RMSE"),
        "q_p95_abs_error_rad": _finite(match["upper_body_p95_abs_error_rad"], f"{path} q p95"),
        "real_time_constant_s": _finite(
            real_arms["median_time_constant_s"], f"{path} real arm time constant"
        ),
        "sim_time_constant_s": _finite(
            sim_arms["median_time_constant_s"], f"{path} simulated arm time constant"
        ),
        "real_delay_s": _finite(real_arms["median_delay_samples"], f"{path} real arm delay")
        / _finite(document["real"]["hz"], f"{path} real Hz"),
        "sim_delay_s": _finite(sim_arms["median_delay_samples"], f"{path} simulated arm delay")
        / _finite(document["sim"]["hz"], f"{path} simulated Hz"),
    }


def _load_visual_static_evidence(path: Path, episode_index: int) -> dict[str, Any]:
    """Load a source-only precision gate for a claimed static interval.

    A simulator contact flag cannot establish that the real demonstration was
    contact-free.  A strict stereo-CAD precision report is still not a force
    measurement, but it is the minimum evidence required before arm-only
    response fitting may be promoted beyond a provisional diagnostic.
    """

    document = json.loads(path.read_text(encoding="utf-8"))
    stereo = document.get("stereo_agreement")
    if not isinstance(stereo, dict):
        raise ValueError(f"{path} is not a source CAD-alignment report")
    paired = stereo.get("accepted_paired_frames")
    translation = stereo.get("accepted_translation_p95_m")
    rotation = stereo.get("accepted_rotation_p95_deg")
    if not isinstance(paired, int) or paired < 0:
        raise ValueError(f"{path} has invalid accepted stereo-pair count")
    translation_value = _finite(translation, f"{path} accepted translation p95")
    rotation_value = _finite(rotation, f"{path} accepted rotation p95")
    passed = (
        document.get("accepted_for_fixed_scene_proposal") is True
        and paired >= 3
        and translation_value <= MAX_VISUAL_STATIC_TRANSLATION_P95_M
        and rotation_value <= MAX_VISUAL_STATIC_ROTATION_P95_DEG
    )
    return {
        "episode_index": episode_index,
        "path": str(path),
        "accepted_paired_frames": paired,
        "accepted_translation_p95_m": translation_value,
        "accepted_rotation_p95_deg": rotation_value,
        "passes_strict_static_precision_gate": passed,
        "thresholds": {
            "accepted_paired_frames_min": 3,
            "accepted_translation_p95_m_max": MAX_VISUAL_STATIC_TRANSLATION_P95_M,
            "accepted_rotation_p95_deg_max": MAX_VISUAL_STATIC_ROTATION_P95_DEG,
        },
    }


def summarize(
    candidates: dict[str, list[Path]],
    visual_static_evidence: dict[int, Path],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    summaries: dict[str, Any] = {}
    for name, paths in candidates.items():
        if len(paths) < 2:
            raise ValueError(f"candidate {name!r} requires reports from at least two episodes")
        reports = [_load_report(path) for path in paths]
        episodes = [entry["episode_index"] for entry in reports]
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"candidate {name!r} repeats a source episode")
        missing_evidence = sorted(set(episodes) - set(visual_static_evidence))
        if missing_evidence:
            raise ValueError(
                f"candidate {name!r} lacks visual static evidence for episodes {missing_evidence}"
            )
        evidence = [_load_visual_static_evidence(visual_static_evidence[episode], episode) for episode in episodes]
        tau_relative_error = [
            abs(entry["sim_time_constant_s"] - entry["real_time_constant_s"])
            / max(entry["real_time_constant_s"], 1.0e-6)
            for entry in reports
        ]
        delay_abs_error = [abs(entry["sim_delay_s"] - entry["real_delay_s"]) for entry in reports]
        q_rmse = [entry["q_rmse_rad"] for entry in reports]
        q_p95 = [entry["q_p95_abs_error_rad"] for entry in reports]
        # A profile must satisfy the joint gate on every calibration episode;
        # ties are resolved by a scale-free response-time mismatch, then RMSE.
        summaries[name] = {
            "reports": reports,
            "episodes": episodes,
            "joint_gate_passes_all_calibration_episodes": max(q_rmse) <= 0.03,
            "visual_static_gate_passes_all_calibration_episodes": all(
                entry["passes_strict_static_precision_gate"] for entry in evidence
            ),
            "visual_static_evidence": evidence,
            "q_rmse_rad": {"median": _quantile(q_rmse, 0.5), "max": max(q_rmse)},
            "q_p95_abs_error_rad": {"median": _quantile(q_p95, 0.5), "max": max(q_p95)},
            "time_constant_relative_error": {
                "median": _quantile(tau_relative_error, 0.5),
                "max": max(tau_relative_error),
            },
            "delay_abs_error_s": {"median": _quantile(delay_abs_error, 0.5), "max": max(delay_abs_error)},
        }
    ranked = sorted(
        summaries,
        key=lambda name: (
            not summaries[name]["joint_gate_passes_all_calibration_episodes"],
            summaries[name]["time_constant_relative_error"]["median"],
            summaries[name]["q_rmse_rad"]["median"],
            summaries[name]["q_rmse_rad"]["max"],
            name,
        ),
    )
    eligible = [
        name
        for name in ranked
        if summaries[name]["joint_gate_passes_all_calibration_episodes"]
        and summaries[name]["visual_static_gate_passes_all_calibration_episodes"]
    ]
    return {
        "schema_version": "team_ramen_flip_table_actuator_profile_selection/v1",
        "policy_use": "forbidden: offline simulator-actuator calibration only",
        "selection_rule": (
            "Each candidate must use two or more distinct source intervals with separately retained "
            "real-RGB/CAD static or contact-exclusion evidence. Reject profiles exceeding 0.03 rad "
            "calibration q RMSE on any episode; rank remaining profiles by median relative arm "
            "time-constant mismatch, then median and maximum q RMSE."
        ),
        "candidates": summaries,
        "ranking": ranked,
        "eligible_candidates": eligible,
        "recommended_candidate": eligible[0] if eligible else None,
        "decision": "profile_eligible_for_freeze" if eligible else "no_profile_eligible_for_freeze",
        "limitations": [
            "This selector does not establish contact freedom; inspect its linked visual/static evidence.",
            "This does not identify contact parameters or prove held-out performance.",
            "The selected profile must remain frozen before contact and held-out validation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=REPORT",
        help="repeat for every candidate/episode report",
    )
    parser.add_argument(
        "--visual-static-evidence",
        action="append",
        required=True,
        metavar="EPISODE=REPORT",
        help="strict source CAD-alignment evidence for each calibration episode",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates: dict[str, list[Path]] = {}
    for value in args.candidate:
        name, separator, report = value.partition("=")
        if not separator or not name or not report:
            raise ValueError("--candidate must be NAME=REPORT")
        candidates.setdefault(name, []).append(Path(report).expanduser().resolve())
    evidence: dict[int, Path] = {}
    for value in args.visual_static_evidence:
        episode_text, separator, report = value.partition("=")
        if not separator or not episode_text or not report:
            raise ValueError("--visual-static-evidence must be EPISODE=REPORT")
        try:
            episode = int(episode_text)
        except ValueError as exc:
            raise ValueError("--visual-static-evidence episode must be an integer") from exc
        if episode < 0 or episode in evidence:
            raise ValueError("--visual-static-evidence episodes must be unique non-negative integers")
        evidence[episode] = Path(report).expanduser().resolve()
    document = summarize(candidates, evidence)
    atomic_write_json(args.output.expanduser().resolve(), document)
    print(json.dumps({"recommended_candidate": document["recommended_candidate"], "ranking": document["ranking"]}))


if __name__ == "__main__":
    main()
