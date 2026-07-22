#!/usr/bin/env python3
"""Write the reproducibility report for one CV rule-based evaluation trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("FLIP_TABLE_")
    }


def _result(output_dir: Path, exit_status: int | None) -> str:
    result_path = output_dir / "eval_results.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return (
            f"- Process exit status: `{exit_status}`\n"
            f"- Episodes: `{result.get('test_count', 'unknown')}`\n"
            f"- Successes: `{result.get('success_count', 'unknown')}`\n"
            f"- Success rate: `{result.get('success_rate', 'unknown')}`\n"
        )
    if exit_status is None:
        return "- Evaluation is running.\n"
    return f"- Process exit status: `{exit_status}`; `eval_results.json` was not produced.\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exit-status", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = _environment()
    settings_text = "\n".join(f"- `{key}={value}`" for key, value in settings.items())
    previous_diff = os.environ.get(
        "FLIP_TABLE_TRIAL_DIFF",
        "No explicit differential was supplied; this run uses the current checked-out defaults.",
    )
    report = f"""# Flip-table CV rule-based trial

## Result

{_result(output_dir, args.exit_status)}
## Difference from the previous trial

{previous_diff}

## Complete algorithm

1. Read only the real-robot-compatible observations: G1 left head RGB, left and right D405 RGB, upper-body joint positions, and Dex1 finger joint positions. The fixed global camera is recorded for review but is never read by the policy.
2. Segment the white tabletop from the black workbench in left-head RGB. Fit the known 0.58 m by 0.42 m tabletop rectangle with calibrated monocular PnP and reconstruct all four CAD leg attachments from its corners. Associate visible shafts with those attachments and never substitute a rear shaft for an occluded front shaft.
3. Select the root-frame-nearest leg as the alignment pivot. Approach it with the corresponding open hand, then use that D405 RGB stream to center the shaft and advance it between the fingers. Keep the hand open through the insertion and authorize closure only after consecutive fresh RGB frames show the shaft centered and deep enough. Confirm enclosure from both Dex1 finger encoders, retrying bounded Cartesian offsets when needed.
4. Keep the pivot enclosed and use the other hand to grasp the visible near tabletop edge. Its D405 RGB lower-edge estimate closes the remaining depth error and gates closure. Pull when a short edge is already nearer the robot; for a long-edge presentation, push, recenter the assembly, and then pull around the held pivot until a short edge faces the robot.
5. Release and clear both hands, relocalize the tabletop from head-left RGB, and approach the near-left leg. Repeat the open insertion, fresh-frame wrist-RGB alignment, and symmetric finger-encoder enclosure checks before starting the flip.
6. Preserve the accepted grasp correction in the moving left-tool frame, then roll the wrist and grasp point counterclockwise toward robot-left through a 90-degree rigid arc around the supported tabletop edge. Advance only while measured wrist forward kinematics follows the command and both finger encoders continue to indicate enclosure.
7. Use the right D405 RGB to approach and enclose the raised tabletop edge, release the left hand only after the right enclosure gate passes, and push toward robot-left beyond the balance point for the second 90-degree roll.
8. Open both hands, retreat, and let the object settle. Rate-limit Cartesian position, orientation, and hand commands before IK, then rate- and acceleration-limit the resulting upper-body joint targets against measured encoder state so IK branch changes cannot create one-cycle jumps.

No demonstration EEF trajectory, learned model, simulator object pose, simulator contact signal, segmentation ground truth, teleport, or direct object manipulation is used by policy control or branching. Simulator-only object state is restricted to post-run success scoring and diagnostics.

## Domain randomization

Each randomized episode samples the room materials and distractor visibility/pose, lighting, policy-camera mount perturbations, robot base and upper-body initial pose, white-table position and full-circle yaw, and physically plausible hand/table/workbench friction and restitution. The tabletop center remains on the workbench but its corners may overhang. The workbench and review camera stay world-fixed. Randomization values are recorded by the simulator logs and the environment snapshot below.

## Runtime settings

{settings_text}

## Artifacts

- `eval_results.json`: aggregate success result
- `test_*/record_video.mp4`: global and policy-camera video
- `test_*/action_state_trace.jsonl`: commands, measured joint/FK state, CV observations, gates, and success trace
- `test_*/camera_frames/`: selected unmodified camera frames when enabled
"""
    (output_dir / "ALGORITHM.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
