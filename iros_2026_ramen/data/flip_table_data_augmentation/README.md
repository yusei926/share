# Flip-table data augmentation

This directory owns Issue #70: collect successful Apple Vision Pro (AVP)
teleoperation demonstrations in the organizer's RoboFinals V1 simulator,
extend them with Isaac Lab Mimic, render accepted trajectories with Omniverse
Replicator, and publish a provenance-tracked LeRobotDataset v3. Cosmos and
image-only augmentation are intentionally outside this pipeline.

## Dataset contract

The immutable real source is:

- repository: `Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1`
- revision: `10a6ec05f9993b8d59faad2957e47153b0f15f37`
- size: 531 episodes, 290,941 frames, 30 Hz

The private release target is
`Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_augmented_2`.

Policy data remains limited to real G1 + Dex1-1 signals:

- `cam_0`: head-left RGB, 640x480
- `cam_2`: left D405 RGB, 640x480
- `cam_3`: right D405 RGB, 640x480
- the existing six numeric features, with upper-body output only

The simulator AVP view remains the unmodified head-left/head-right stereo pair
for low latency. In the real backend only, the same central stereo pair is
surrounded in both eyes by a left D405 panel with the measured left seven arm
angles and a right D405 panel with the measured right seven arm angles. The
wrist panels are deliberately duplicated monocular HUDs, not synthetic stereo;
the head images remain the only stereoscopic camera content. The default
official TeleVuer `ego` mode retains Apple Vision Pro pass-through around this
compact operator window. In a real session, `HANDS READY` must be visible in
that HUD before `r` can arm tracking; an early `r` is deliberately ignored so
headset acquisition cannot arm the robot later by surprise. This display-only
composition never changes the policy images or recorded camera payloads.
The real launcher also opens a non-blocking Desktop monitor by default. Its
2x2 layout shows head-left, head-right, left wrist, and right wrist at the same
time, with the fourteen measured arm angles over the corresponding wrist
tiles. It is display-only and drops preview updates rather than applying
backpressure to AVP IK or control. Set
`FLIP_TABLE_TELEOP_DESKTOP_PREVIEW=false` only for a headless run.
Head-right, global images, object pose,
segmentation, contact and simulator state are never policy inputs. Simulator
GT is retained only in diagnostic sidecars for offline phase annotation and
success validation.

## Production path

1. Verify the source, FK, camera calibration, XR/TeleVuer revisions, and
   `paperc/robofinals:RoboFinals-IKEA-V1` image digest.
2. Collect 30 successful AVP simulator demonstrations: 10 each in `mild`,
   `medium`, and `full` DR profiles. The profiles randomize full table yaw,
   robot pose, realistic contact terms, camera mount/intrinsics/noise and
   latency, lighting, room, and foreground props.
3. Audit the raw collection. It rejects schema, timestamp, camera, sidecar,
   policy-isolation, yaw-diversity, and DR-stratum violations.
4. Convert accepted sim demos to phase-indexed HDF5. Object pose and contact
   are used only offline for phase boundaries and validation, then excluded
   from the HDF5; SAM/FoundationPose is not used for these simulator
   demonstrations.
5. Run exactly 100 Mimic pilot trials and require at least 50 physical
   successes. Then generate at least 2,000 distinct, physically validated
   trajectories using PINK and Dex1 actions.
6. Reject every candidate that fails physics, safety, provenance, or FK gates.
   Rejected trajectories never reach rendering or export.
7. Render each accepted Mimic trajectory with at least two independently
   sampled appearance/camera variants. Direct sim teleop stays one unmodified
   visual realization.
8. Assemble direct sim teleop, Mimic, and real episodes; validate locally;
   stage, clean-download validate, and publish the private LeRobot v3 release.

Dataset splits are made by physical trajectory lineage. Appearance variants of
one trajectory can never cross train, validation, and test splits.

## Layout

```text
flip_table_data_augmentation/
  teleop/
    shared/      backend-neutral state/watchdog vocabulary
    real/        physical G1 runner and official-motion safety
    sim/         Isaac runner, socket backend, and simulation safety
    contracts.py shared 14D arm + 2D Dex1 target/observation schema
  mimic/         Teleop conversion, PINK generation, and recording
  replicator/    Accepted-trajectory replay and appearance randomization
  export/        LeRobot v3 assembly, validation, and HF publication
  scripts/       Setup, conversion, gates, rendering, and release entrypoints
  configs/       Pinned runtime, randomization, and release settings
  tests/         Unit and cross-contract tests
  outputs/       Generated artifacts; never source-controlled
```

`teleop/configs/teleop_v1.json` owns the shared XR, V1, joint-envelope, clock,
operator, policy, DR, and collection contract. Real and Sim do not share an
actuator or safety-filter implementation: `teleop/real/` follows Unitree's
250 Hz arm_sdk and 200 Hz Dex1 motion contracts, while `teleop/sim/` applies Isaac's 50 Hz
action limits. `configs/pipeline_v1.json` owns Mimic,
rendering, export, and the HF release contract. Secrets such as `HF_TOKEN` are
environment variables only.

## AVP teleop collection

On the operator PC, run once per machine:

```bash
data/flip_table_data_augmentation/setup_teleop_runtime.sh
inference/desktop/xr/generate_avp_tls.sh
```

The RTX 5090 simulator is reached over Tailscale SSH. The launcher syncs an
isolated staging directory, verifies the V1 digest, starts an SSH tunnel, and
does not modify the workstation checkout.

The same command can be run directly on the RTX 5090 workstation. In `auto`
mode the launcher detects its local RTX 5090 and Docker runtime, skips
Tailscale/SSH, and runs the simulator and AVP server on that machine. Run the
two one-time setup commands above on each machine used as the AVP host. The
launcher accepts either the local `tv` environment or the workstation's
`xr-teleop` environment. `FLIP_TABLE_SIM_EXECUTION=local|remote` is available
as an explicit override; the normal command remains `run_sim_teleop.sh`.

The simulator remains running after a simulated AVP session by default. The
next invocation with the same teleop config, DR profile, and seed reconnects
without Isaac Sim's cold startup. Each new AVP connection safely resets the
environment before it is accepted. Stop the matching persistent simulator
explicitly when finished:

```bash
FLIP_TABLE_TELEOP_DR_PROFILE=mild FLIP_TABLE_TELEOP_SEED=101 \
  data/flip_table_data_augmentation/stop_teleop_sim.sh
```

Set `FLIP_TABLE_TELEOP_RESTART_SIMULATOR=true` before `run_sim_teleop.sh` to
replace a running instance after code or environment changes. Set
`FLIP_TABLE_TELEOP_KEEP_SIMULATOR_RUNNING=false` for one-shot behavior.

Before an AVP collection session, the following non-actuating probe confirms
the tunnel and true head-stereo stream without exposing the real backend:

```bash
FLIP_TABLE_TELEOP_TRANSPORT_PROBE=true \
  FLIP_TABLE_TELEOP_PROBE_FRAMES=180 \
  FLIP_TABLE_TELEOP_DR_PROFILE=mild FLIP_TABLE_TELEOP_SEED=100 \
  data/flip_table_data_augmentation/run_sim_teleop.sh
```

The following small-motion probe exercises the same 30 Hz socket, safety
filter, 16-D arm/Dex1 action, and both Dex1 targets before a headset session.
It moves only the two wrist-roll joints by 0.12 rad and saves measured spans
under `outputs/flip_table_teleop/probes/`:

```bash
FLIP_TABLE_TELEOP_CONTROL_PROBE=true \
  FLIP_TABLE_TELEOP_DR_PROFILE=mild FLIP_TABLE_TELEOP_SEED=101 \
  data/flip_table_data_augmentation/run_sim_teleop.sh
```

Long transport probes fail below 28 Hz. The control probe additionally gates
live stereo before and after the collection toggle at 28 Hz,
verifies the organizer WBC remains upright without fixing the root, legs, or
waist, bounds unintended arm motion and tracking error, and exercises the full
continuous opening range of both Dex1 hands. Its first two seconds are an IDLE
warm-up, matching the interval before an operator presses `r`.

The launchers isolate TeleVuer and the official G1 IK from each backend
transport. Only head stereo, measured arm/hand state, and real-compatible
14-D arm plus two Dex1 targets cross that process boundary. Commands run at
30 Hz, the WBC action loop at 50 Hz, and interactive sim physics at 200 Hz.
The real G1 `rt/arm_sdk` publisher is a separate official-protocol loop at
250 Hz; the simulator rate must not be reused as its DDS publish rate.
The real runner never imports the simulator backend/filter, and the simulator
runner never imports Unitree DDS or the real backend. The old
`run_teleop.sh real|sim` form remains only as a compatibility dispatcher.

The assembled white table retains the original visual mesh and all exposed
grasp/support surfaces. Its collision representation removes only internal
thread colliders that overlap after assembly and destabilize the reset. The
mass is fixed at 1.596 kg; contact, lighting, room, camera, robot pose, and table
pose remain profile-randomized. Recorded policy images are RGB 640x480,
quality-95 JPEG with 4:4:4 chroma and are never recompressed by the raw writer.

The launcher prints the AVP URL only after the simulator camera bridge is
ready. A cold Isaac Sim start can take several minutes. The watchdog separates
WebSocket and bilateral hand liveness, enters measured-pose hold immediately
on invalid input, and requires a new stable anchor after recovery. Five
distinct bilateral frames must remain within 15 mm and eight degrees before a
requested anchor becomes active. The organizer diagnostic video is encoded
only while recording.

```bash
FLIP_TABLE_TELEOP_DR_PROFILE=mild FLIP_TABLE_TELEOP_SEED=101 \
  data/flip_table_data_augmentation/run_sim_teleop.sh
```

Set `FLIP_TABLE_TELEOP_XR_DISPLAY_MODE=immersive` only when a full-field robot
view is required. The default `ego` mode is the collection mode; TeleVuer's
`pass-through`-only mode is intentionally excluded because it hides both robot
camera images.

Controls are `r` to track/pause/resume, `s` to start recording and then save,
and `d` to discard/reset. Both `q` (the left pedal) and real-backend `Ctrl+C`
are accepted during either tracking or HOLD: they stop tracking, send an
explicit IDLE/QUIT transition, blend arm ownership back to the regular
controller over the pinned upstream XR controller's approximately two-second
release interval, and exit only after weight=0 is published. In simulation,
`q` safely ends only the current AVP job.
While tracking, one `r` press pauses and holds the current arm/Dex1 target;
it does not release arm authority or follow subsequently moving hands. Press
`r` again to request a fresh anchor, then hold both hands still. Tracking
resumes only after the stable bilateral window has been accepted. Every
re-anchor clears the official IK warm start and moving-filter history. After
the first no-motion target, TeleVuer's absolute left/right wrist poses are
passed directly to the pinned official G1_29 IK; no extra relative-offset or
deadband coordinate mapping is applied.
The real arm path forwards both official IK outputs (`q` and RNEA
feedforward torque) to `rt/arm_sdk`. It retains the official IK moving average
and official measured-relative global 20→30 rad/s arm scaling, but does not
stack the simulator's per-joint velocity/acceleration smoother on top. IK is
returned before JPEG decode, HUD composition, and AVP/Desktop rendering.
If either AVP heartbeat becomes genuinely stale, the watchdog holds the last
safe pose, stops adding recording frames, and pauses tracking. When hands are
visible again and the HUD says `HANDS READY`, press `r` once to request a fresh
anchor. An early `r` is ignored rather than queued. A recovered hand stream
never resumes motion without this explicit action.
Saving an episode does not disarm tracking or release `rt/arm_sdk`; it only
atomically finalizes the files. A transient camera outage pauses on the last
applied arm/Dex1 target, discards an in-progress recording, and requires a new
`r` after all fresh streams recover. The Orin launcher publishes every acquired
frame once (never duplicate-padding a nominal FPS) and recovers only the failed
D405 pipeline instead of terminating the head and other wrist streams.
The non-actuating real preflight measures three seconds of unique frames and
refuses any camera below 28 Hz. Save-time validation also rejects a real
episode when its sample rate is below 28 Hz or any camera's consecutive JPEG
duplicate fraction exceeds 5%; rejected data is retained only under
`raw/rejected/`.
Each simulator launch saves the desktop-side timing and hand-tracking output
to `outputs/flip_table_teleop/runtime/<run>/operator.log`. It also writes
`operator_session_report.json`, which independently gates the 28 Hz camera,
AVP hand-tracking, and TRACK-command paths; observed motion of both arms and
both Dex1 openings; tracking error and latency; recording/reset; and a safe
`q` shutdown. The report additionally records the fixed assembled-table mass,
sampled contact coefficients, maximum four-finger contact forces, and maximum
Dex1 drive force against the physical 20 N per-finger limit. For an acceptance
run, press `r`, move both arms, open and close
the left and right hands independently through a useful range, press `s` then
`d` to exercise recording and reset, press `r` once more, and finish with `q`.
The run is accepted only when the report's `passed` field is `true`.

The 20 N value is the limit of each Dex1 prismatic actuator, not a guaranteed
surface-normal grasp force. The simulator stores actuator and finger-contact
maxima in the report; neither signal is exposed to teleoperation or a policy.

Before the simulator sends its first operator image, it holds the measured
joint pose for one second and rejects any randomized reset in which the white
table moves on its own. It also enforces a 0.20 m minimum root clearance from
the black workbench. The organizer workbench measures 0.75 x 1.80 x 0.76 m,
with its top surface at world Z = 0.762294 m. The G1 root default is 0.78 m.
Collect ten successful episodes for every DR profile, then require this audit
to pass:

```bash
python data/flip_table_data_augmentation/scripts/audit_teleop_collection.py \
  --raw-root outputs/flip_table_teleop/raw \
  --output outputs/flip_table_teleop/collection_audit.json
```

`run_real_teleop.sh` uses official `rt/arm_sdk` and Dex1 DDS after
non-actuating DDS, image, and AVP preflights. `G1_DDS_INTERFACE` and
`G1_IMAGE_SERVER_IP` are required explicitly; `AVP_DESKTOP_IP` is also required
when more than one Desktop IPv4 interface is active. The read-only preflight
requires high-level FSM 501/mode 0 and a repository-wide G1 controller lock.
It publishes nothing until the
operator presses `r`. Operator/policy commands contain zero lower-body
dimensions. The required `rt/arm_sdk` packet mirrors the official 35-slot
G1_29 whole-body snapshot while overwriting only arm indices 15..28; Regular
mode retains balance, waist/leg, and locomotion ownership. Normal lower-body
motion is recorded diagnostically rather than treated as an arm fault.
Camera timestamps are per-stream host receive times marked approximate because
the pinned TeleImager protocol does not expose hardware capture timestamps.

## Mimic and release gates

The primary entrypoints are:

```bash
python data/flip_table_data_augmentation/scripts/convert_teleop_episode.py --help
python data/flip_table_data_augmentation/scripts/package_teleop_episode.py --help
python data/flip_table_data_augmentation/scripts/export_teleop_mimic_source.py --help
python data/flip_table_data_augmentation/scripts/run_mimic_generation.py --help
python data/flip_table_data_augmentation/scripts/render_accepted_trajectories.py --help
python data/flip_table_data_augmentation/scripts/verify_mimic_release_gate.py --help
```

`verify_mimic_release_gate.py` requires the exact config/runtime manifest, at
least 50 pilot successes, and 2,000 ledger-backed trajectories with physical
DR evidence and FK validation. It cannot be passed by a video-only claim.

The stock Isaac Lab Mimic retargeter is deliberately blocked for
sim-teleoperation source HDF5 until an audited RGB-only table-pose adapter is
available. Its regular object-pose path reads simulator state during trajectory
planning, which is incompatible with this project's Sim-to-Real constraint.
The gate remains available for real-data sources whose object pose was inferred
from real sensor data, never simulator GT.

The final HF release additionally requires LeRobot metadata/Parquet/video
validation, a content manifest, transactional staging upload, and clean
re-download validation before and after publishing `main`.

## Training comparison

After a clean checkout of the released augmented dataset exists, write the
fixed ACT/Flow Matching comparison plan:

```bash
model/subtask_policy_training/.venv/bin/python \
  model/subtask_policy_training/scripts/run_augmentation_benchmark.py \
  --dataset-root /path/to/IROS2026_RAMEN_suzuki_flip_table_augmented_2 \
  --output-root outputs/flip_table_augmented_benchmark
```

Review `benchmark_plan.json`, then add `--execute`. Each policy uses one seed
and one update budget across these three conditions:

- `real_only`: 100% real
- `real_sim_teleop`: 90% real and 10% direct sim teleop
- `real_sim_teleop_mimic`: 50% real, 10% direct sim teleop, and 40% Mimic

The sampler balances physical trajectory lineages before appearance variants,
preventing repeated renders from overweighting a trajectory. Runs log to
`iros2026-ramen-flip-table`; model publishing is intentionally disabled by the
comparison launcher.

## Legacy source-annotation utilities

`object_pose/` and the Grounded SAM/FoundationPose scripts are retained for
research and audit of the real-only source. They are not on the approved
sim-teleop-to-Mimic production path above.

## References

- [Isaac Lab Mimic](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/augmented_imitation.html)
- [Isaac Lab Mimic API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_mimic/index.html)
- [Omniverse Replicator](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/index.html)
