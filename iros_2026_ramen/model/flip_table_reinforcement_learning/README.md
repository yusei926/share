# Flip-table simulation baselines

This directory contains the maintained simulator-side baselines for the
IROS 2026 RAMEN `flip_table` subtask. It runs only against the organizer image
`paperc/robofinals:RoboFinals-IKEA-V1` and applies the repository's Dex1-1
compatibility overlay at runtime.

The previous fixed-trajectory and CEM contact-search experiments did not
produce a task-successful teacher. Their candidate files and launch paths were
removed deliberately. Git history and the execution summary preserve the
decision; they are not supported execution paths.

## Runtime contract

Deployable policies use only real-robot observations:

- head-left RGB and left/right D405 RGB at 640x480;
- waist 3, arm 14, and Dex1 2 observed states (19-D total);
- a bounded 16-D absolute target containing arm 14 and Dex1 2 only.

The two Dex1 values follow the source dataset convention: `0.0=closed` and
`4.5=open`. Conversion to simulator joint positions or normalized actuator
commands must preserve this polarity.

The organizer's unmodified G1 decoupled WBC owns the floating base, legs, and
waist in simulation. It runs with 200 Hz physics and 50 Hz control, zero
navigation velocity, a 0.74 m base-height command, and zero torso RPY. The real
robot uses Regular mode for the same ownership boundary. `fixed_diagnostic`
exists only for fault isolation; training, teacher generation, replay, and
evaluation require `balanced_wbc`. Object pose, contacts, segmentation,
global-camera images, and other simulator-only signals are restricted to
offline success checks and diagnostics.

## Maintained commands

Run inside the organizer container, or use `run_train_local.sh` on a Linux host
with Docker and NVIDIA Container Toolkit. The launcher creates an immutable
copy of the two organizer robot files before applying repository overlays.

```bash
cd /workspace/iros_2026_ramen
model/flip_table_reinforcement_learning/run_train_in_container.sh audit_contract
model/flip_table_reinforcement_learning/run_train_in_container.sh audit_partial_reset
model/flip_table_reinforcement_learning/run_train_in_container.sh smoke
```

`audit_contract` checks the 16-D action routing, 60-second WBC stand stability,
reset isolation, table assembly, contact reporting, and known-command tracking. `smoke` replays
the source action prior only as an environment health check; it is not a policy
success claim.

For legacy comparisons, `evaluate` runs a PPO checkpoint and
`evaluate_rlpd_stage` runs a Flow checkpoint or a Flow+RLPD checkpoint. These
baselines are retained for measurement, not as the recommended path to solve
the task. Every command writes a run manifest and stores generated files below
the ignored `outputs/` directory.

## Source calibration utilities

`teacher/` and the corresponding scripts prepare an auditable source-table
calibration from recorded head stereo or D405 IR stereo. They are offline-only
tools for aligning demonstrations and never become policy inputs. They require
human-verified physical correspondences and reject insufficient or inconsistent
geometry.

Legacy 19-D demonstrations are converted without modifying the source: the
three waist action columns are dropped, and the conversion version plus source
hash are recorded. Existing 19-D-output checkpoints are intentionally not
compatible and must be retrained.
