# Flip-table Real-to-Sim Calibration

This package prepares an immutable calibration bundle from the pinned real
LeRobot dataset. It does not modify the dataset and does not use table pose,
contact, segmentation, or global-camera signals as a policy input.

`prepare.py` audits every numeric row, selects one anchor, two calibration, and
five held-out validation episodes, then writes the deployable 16-D
arms-plus-Dex1 replay stream. The organizer WBC remains responsible for the
floating base, legs, and waist. Root and lower-body labels are retained as
reference; they must never be applied as per-frame teleports during a replay.

`replay.py materialize --full-body-diagnostic` additionally writes the recorded
body29-plus-Dex1 stream for one narrow controller-identification experiment.
That path deliberately leaves the root dynamic and reports root drift; it is
not a policy, a deployment path, or valid camera/contact/held-out evidence.
It exists to reject the tempting but invalid assumption that a sequence of
recorded joint-position targets can replace G1's real balance controller.

The calibration bundle records measured raw-MCAP head stereo and D405 intrinsics
when the pinned raw metadata is available. Link extrinsics and contact physics
remain fitted quantities and must include uncertainty in the final report.

## What the dataset can identify

The pinned LeRobot dataset contains synchronized **raw** RGB, head-stereo intrinsics,
D405 intrinsics, robot encoders, commands, and end-effector labels. It does
**not** contain a camera-to-link transform, table 6-DoF pose, contact force,
or a material/friction measurement. Those omissions matter: an RGB silhouette
score alone cannot tell whether a residual came from a camera mount, the
table's reset pose, or an occluding arm.

Before changing a head or wrist mount, run the source-FK projection audit on
the same frames. It uses only real encoders plus the pinned intrinsic model and
is an independent check that visible wrist links project onto the real arms:

```bash
conda run -n tv python -m data.flip_table_data_augmentation.scripts.audit_source_camera_projection \
  --source-root ~/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/<revision> \
  --episode-index <episode> --frames 0 10 <later-frame> \
  --urdf outputs/flip_table_real_to_sim/<run_id>/assets/g1_29dof_with_hand.urdf \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/source_arm_projection_<episode>
```

The automated visibility check is necessary but not sufficient; its overlay is
an explicit visual-review artifact. Do not replace a measured head mount with
a perturbation that only improves a single-frame PnP, mask-IoU, or edge score.

The existing dataset can nevertheless estimate a **fixed pre-flip scene**
without seeing all four tabletop corners in every frame. The V1 assembled CAD
mesh supplies the outer rim and four leg axes;
`source_cad_alignment.py` registers those primitives directly against the raw
head-left/head-right RGB edge and white-support evidence, using robot-FK head
poses and the measured stereo transform. Each eye and frame is independently
scored, then a robust temporal consensus is formed. This does not fabricate
unobserved image corners and uses no simulator ground truth.

For a V1 scene candidate, first convert that fit to a workbench-local reset
with `source_scene_candidate.py`, then derive a torso-local stereo-mount delta
from the V1 diagnostic trace with `source_head_mount_candidate.py`. Finally,
run `source_projection_conformance.py`. Its pass gate compares CAD
reprojections, camera pose, and fixed-table pose; it deliberately does not use
RGB mask IoU or texture similarity as a geometry gate because the real black
workbench, exposure, reflections, and robot material differ from V1. The
source fit remains valid when individual corners are occluded, because the
accepted source pose was obtained from multi-frame rim/leg evidence before
the four CAD corners are projected for the final metric check.

`source_scene_candidate.py` must read the trace row synchronized to source
frame zero (normally terminal warmup step `119`), not reset step `0`.
The floating base can settle during warmup; using step `0` aliases that WBC
motion into the table offset. The source CAD frame and the V1 assembled-table
articulation may also have an unidentifiable vertical-origin difference. The
candidate therefore identifies only workbench-plane `x/y` and yaw, fixes its
vertical offset to zero, and records the V1 trace's **reset-time** local robot
root only to prevent the ordinary placement heuristic from moving G1 when the
table offset changes. It never writes the post-warmup dynamic root pose and
never teleports the root after reset. V1's supported table height and
WBC-owned floating base remain intact. The dynamic root and vertical
differences are retained as diagnostic report fields.

```bash
PYTHONPATH=$PWD conda run -n tv python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.source_projection_conformance \
  --source-alignment outputs/flip_table_real_to_sim/<run_id>/source_cad_alignment.json \
  --sim-trace outputs/flip_table_real_to_sim/<run_id>/candidate/test_0/action_state_trace.jsonl \
  --source-frame 0 --sim-step 119 \
  --output outputs/flip_table_real_to_sim/<run_id>/source_projection_conformance.json
```

For a metric camera-to-link update beyond the existing FK-validated mount, a
hand-eye sequence with a rigid ChArUco/AprilTag board is still required. The
dataset has no independent robot-root reference for such an update. This is
separate from the CAD fixed-scene fit and is not a policy modality.

Before interpreting a wrist mask residual as a mount error, run the source
state-timing audit.  It compares encoder-FK from `robot_q_current[t+offset]`
against `ee_state[t]` across a bounded offset sweep.  The result is diagnostic
only: it must never shift RGB/video frames, rewrite labels, alter replay
timing, or reach a policy.  A reliable camera update requires the zero-offset
or independently measured timing model to be established first.

```bash
FLIP_TABLE_AUG_RUNTIME_MODE=docker \
FLIP_TABLE_AUG_OUTPUTS="$PWD/outputs" \
data/flip_table_data_augmentation/scripts/run_object_pose_runtime.sh state-timing \
  --source-root /root/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/<revision> \
  --episode-index <episode> \
  --urdf /workspace/robofinals/robofinals/core/mdp/actions/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf \
  --output /outputs/flip_table_real_to_sim/<run_id>/state_timing_<episode>.json
```

The fixed-scene CAD fit does not identify contact parameters. Contact fitting
needs a table trajectory accepted by a CAD/stereo residual gate across multiple
frames. A raw white-pixel mask is not such a trajectory: arms and specular
highlights can make its apparent depth jump even while the physical table is
static. The simulator may record its own table pose/contact trace for offline
comparison, but that trace must never enter a policy input, reward, planner,
or runtime branch.

The target is IKEA UTTER article `603.577.37`. Its published dimensions are
22 7/8 x 16 1/2 x 16 7/8 inches (approximately 0.581 x 0.419 x 0.429 m).
The V1 assembled tabletop bounds are 0.580 x 0.420 m, so tabletop scale is a
fixed physical constraint during calibration rather than a free camera-fit
parameter. The current contact-fit contract fixes table mass at 1.596 kg;
confirm it on a scale before treating that nominal value as an as-built fact.

`data/bitrobot_lerobot_subtask_datasets` copies the official source video
bytes into the subtask dataset without image transformation. The pinned raw
calibration therefore remains the camera-model hypothesis to test against
those RGB frames. V1 can set focal/aperture but cannot set the measured
principal point or lens distortion directly. The recorded-camera remap must be
applied to simulator policy images only after a raw-versus-rectified audit;
the existing data-augmentation remapper is not evidence that the evaluator is
already applying it.

The evaluator applies the recorded raw-image model to learned-policy camera inputs
without exposing calibration values as policy features. In particular, the
D405 pinhole source is the mean of the two pinned raw color calibrations:
`focal_length=24 mm`, `horizontal_aperture=35.310106 mm`, and
`vertical_aperture=26.482580 mm` (about `72.68 x 57.77 degrees`). A previous
`45.55 mm` horizontal aperture came from a generic product-sheet field of
view and was too wide for this dataset. D405 image remapping uses the
RealSense inverse-Brown model, not OpenCV's ordinary Brown model.

Camera-comparison reports must declare whether simulated PNGs were saved
before or after the recorded-camera remap. When they are saved after it,
estimate both real and simulated head images with the raw calibrated
intrinsics; interpreting those PNGs as simulator pinhole images would silently
invalidate a fit.

### Robot-occlusion-aware RGB comparison

`visual_alignment.py` scores an RGB-derived table silhouette, not a rendered
table mask. White G1/Dex1 links otherwise contaminate this score in both real
and simulated images. Use `export_head_robot_masks.py` to project only the
pinned G1 + Dex1-1 visual URDF from recorded/actual joint encoders and the
logged head-camera pose, then pass both masks to the comparison tool. This is
an offline self-occlusion exclusion: it must never use table pose, contact,
segmentation, or simulator object state, and must never enter a policy,
planner, reward, or inference-time branch.

The mask-IoU gate is still independent from CAD camera/table reprojection. A
poor RGB IoU can arise from material, exposure, reflection, or segmentation
differences even when camera geometry is good. Do not tune a camera mount to
one mask score; require the geometric gate and unused-episode evidence.

### Table-independent head-mount diagnostic

### FK-first camera and scene order

`fk_first_camera_calibration.py` is the release path for a source-only
calibration: first fit one head-left correction from static, arm-visible
`robot_q_current` poses; then fit the fixed table and its support plane with the accepted
head stereo; finally fit each D405 against that fixed scene.  It fixes raw
intrinsics, distortion, and the head stereo baseline.  It never changes a V1
camera default by itself.

Run the head and scene stages on the RTX host only:

```bash
PYTHONPATH="$PWD" conda run -n unitree python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.fk_first_camera_calibration head \
  --source-root <pinned_snapshot> --episodes 184 250 --urdf <g1_dex1_urdf> \
  --output-dir outputs/flip_table_real_to_sim/<run>/head
PYTHONPATH="$PWD" conda run -n unitree python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.fk_first_camera_calibration scene \
  --source-root <pinned_snapshot> --urdf <g1_dex1_urdf> \
  --head-report outputs/flip_table_real_to_sim/<run>/head/head_camera_calibration.json \
  --stereo-calibration <head_camera_params.yaml> \
  --output-dir outputs/flip_table_real_to_sim/<run>/scene
```

The head stage emits annotated source frames and
`manual_keypoints.template.json`.  Correct only low-confidence or visibly
wrong elbow/wrist/hand-base features, rerun with `--manual-keypoints`, and
retain both reports.  The scene stage refuses to run until the head hold-out
gate passes.  Before wrist fitting, regenerate D405 mask evidence from the
pinned RGB on the RTX host; never reuse a rejected mask merely because it is
available on disk:

```bash
FLIP_TABLE_HEAD_STEREO_CALIBRATION=<head_camera_params.yaml> \
evaluate/flip_table_simulation/real_to_sim_calibration/run_fk_first_wrist_evidence.sh \
  <pinned_snapshot> outputs/flip_table_real_to_sim/<run>/wrist_evidence 184 250
```

The wrist proposal remains rejected until both sides have a shared-offset
fit and an independent held-out real-image gate.  Do not replace missing D405
evidence with simulator images, scene truth, or a hand-tuned mount.

The table CAD identifies a support-plane normal and height, but not a complete
`root_from_workbench` pose when a known bench edge or fiducial is absent from
both head images. The scene manifest records that non-identifiability instead
of inventing an in-plane translation or yaw.

For a D405 time-offset sweep, generate one self-contained manifest per offset.
It retains the exact RGB bytes and changes only the FK row paired to each
image; it never substitutes a shifted video frame:

```bash
PYTHONPATH="$PWD" conda run -n unitree python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.wrist_time_offset_manifest \
  --input-manifest outputs/flip_table_real_to_sim/<run>/wrist_evidence/input/episode-184/manifest.json \
  --source-root <pinned_snapshot> --source-urdf <g1_dex1_urdf> \
  --q-current-offset-frames -1 \
  --output-dir outputs/flip_table_real_to_sim/<run>/wrist_offset/episode-184-qminus1
```

If table CAD/PnP residuals are too high to constrain a shared head mount, use
the early arm motion visible in real head-left RGB as an independent *offline*
diagnostic. `head_arm_motion_alignment.py` scores projected G1 arm-link motion
against inter-frame RGB motion support. It does not use table pose, simulator
state, policy features, rewards, or any inference-time branch.

Run it separately for at least two episodes. Its output is only an
episode-local candidate: do not copy the correction into the V1 camera setup
unless a conversion to the V1 stereo-rig convention, cross-episode consensus,
and unused-episode RGB gates all pass. This prevents table motion or lighting
changes from being mistaken for a measured camera mount.

```bash
PYTHONPATH="$PWD" conda run -n unitree python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.head_arm_motion_alignment \
  --source-root ~/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/10a6ec05f9993b8d59faad2957e47153b0f15f37 \
  --episode-index 250 --frames 0 10 20 30 40 50 60 \
  --urdf /path/to/g1_29dof_with_hand.urdf \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/head_arm_motion_0250
```

The runtime needs `pinocchio`, `scipy`, `pyarrow`, and `ffmpeg`. On a host
without `ffmpeg`, extract the requested RGB frames once as
`frame_XXXXXX.png` and pass their directory via `--image-root`; this preserves
the exact decoded pixels while avoiding an environment-specific video tool.

```bash
conda run -n tv python -m evaluate.flip_table_simulation.real_to_sim_calibration.prepare \
  --dataset-root ~/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/10a6ec05f9993b8d59faad2957e47153b0f15f37 \
  --output-dir outputs/flip_table_real_to_sim/<run_id>
```

The resulting `episodes/anchor.json` is directly compatible with the existing
`RecordedJointTargetPolicy` through its `recorded_upper_body_target_and_hand_cmd`
field after extraction to the policy's `actions` schema:

The default roles are selected deterministically from the complete numeric
audit and then frozen in `calibration_manifest.json`. If a nominal calibration
episode cannot yield three direct, stereo-consistent V1-CAD registrations, it
is unavailable as camera evidence even when its motion score is high. Create
a reviewed `team_ramen_flip_table_selection_override/v1` JSON with the
replacement role, the rejected and accepted CAD reports, and the reason, then
run the same full audit with:

```bash
conda run -n tv python -m evaluate.flip_table_simulation.real_to_sim_calibration.prepare \
  --dataset-root <pinned_hf_snapshot> \
  --output-dir outputs/flip_table_real_to_sim/<run_id> \
  --eef-fk-audit <all_episode_eef_fk_audit.json> \
  --selection-override <reviewed_visual_selection.json>
```

An override cannot add an unaudited, duplicate, or EEF/FK-ineligible episode.
The manifest records its digest and evidence beside the complete numeric audit;
it is selection provenance only, never a camera correction or a policy input.

```bash
evaluate/flip_table_simulation/real_to_sim_calibration/run_anchor_replay.sh \
  outputs/flip_table_real_to_sim/<run_id>/episodes/anchor.json
```

The runner bind-mounts only the generated replay JSON into the container and
emits `joint_tracking_report.json`. It never mount-writes the source dataset.
Recorded replays retain 50 Hz control/action-state traces and native requested
RGB evidence, while `FLIP_TABLE_REPLAY_REVIEW_VIDEO_HZ=10` limits only the
human-review MP4. The resulting trace records this video stride; it must never
be used for time alignment or actuator fitting.

For a deterministic free-space PD comparison, use
`run_actuator_probe.sh BUNDLE OUTPUT_DIR SOURCE_FRAME_END STIFFNESS_SCALE DAMPING_SCALE [SOURCE_FRAME_START [ARMATURE_SCALE]]`.
The two scales are equal-endpoint reset values relative to the immutable
persistent-worker actuator foundation; `ARMATURE_SCALE` defaults to `1`.
The probe preserves the real command
stream, source clock, table mass, and full control trace; it only bounds the
simulated prefix and samples the review MP4. Set armature and friction to
their baseline values unless armature is the explicit identification variable,
then rank candidates by the reported simulator-to-real encoder metric rather
than MP4 appearance.

Do not freeze a profile merely because it has the lowest replay RMSE.  Pass
two or more distinct calibration reports to `actuator_profile_selection.py`,
and attach a strict source-CAD static-precision report for each episode.  The
selector rejects a shared default when the real RGB evidence cannot establish
the required pre-contact precision; a `recommended_candidate: null` result is
the correct outcome in that case, not permission to pick the lowest-RMSE
candidate.  It remains an offline diagnostic and does not establish contact
forces or held-out performance.

To test that controller-identification boundary explicitly, create a separate
output directory and retain the report beside its trace. Do not use this result
to tune a policy or replace the WBC replay:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.replay materialize \
  --episode-bundle outputs/flip_table_real_to_sim/<run_id>/episodes/calibration_0250.json \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/full_body_diagnostic \
  --full-body-diagnostic

python -m evaluate.flip_table_simulation.real_to_sim_calibration.replay \
  analyze-full-body-trace \
  --trace outputs/flip_table_real_to_sim/<run_id>/full_body_diagnostic/eval/test_0/action_state_trace.jsonl \
  --output outputs/flip_table_real_to_sim/<run_id>/full_body_diagnostic/report.json
```

The full-body report separates lower-body, upper-body, and Dex1 tracking from
floating-base and table displacement. A root drop or drift is a failed
controller-model hypothesis, not a condition to hide with a lock, teleport,
attachment, or direct table manipulation.

### Dex1 contact-path diagnostic

Before fitting table friction or contact stiffness, verify that each Dex1
finger can generate and sustain contact against a 40 mm diagnostic fixture.
`Dex1ForceCalibrationPolicy` is intentionally available only through the
isolated runner: the fixture is authored before Isaac starts and is never part
of a replay or policy-evaluation scene. Its center is derived from measured
official Dex1 collision-STL bounds plus runtime link FK; V1's instance proxies
do not provide usable runtime USD bounds. The fixture remains static throughout
the diagnostic. The report records each finger's 20 N-limited drive effort,
contact-sensor force, opening, sustained contact duration, and its full
trajectory in the fixed fixture frame. The trajectory distinguishes a genuine
missing collision from a WBC/reset motion that carries a finger away from the
fixture. A failed report is a collider/sensor or fixture-geometry problem to
fix before contact-parameter fitting, not grounds for increasing friction or
relaxing a manipulation metric.

```bash
FLIP_TABLE_POLICY_NAME=Dex1ForceCalibrationPolicy \
FLIP_TABLE_EVAL_MODE=nominal \
FLIP_TABLE_TEST_NUM=1 \
FLIP_TABLE_TIME_OUT_LIMIT=20 \
FLIP_TABLE_SIM_OUTPUT_DIR="$PWD/outputs/flip_table_real_to_sim/<run_id>" \
evaluate/flip_table_simulation/run_eval.sh
```

This fixture is diagnostic-only: its collision forces, geometry, and pass/fail
result must never be exposed to a policy, planner, reward, or runtime branch.

### Persistent Isaac replay worker

For repeated fixed-scene replay calibration on one RTX machine, start one
worker and leave it running until an explicit stop. Isaac is initialized once;
each replay job still calls `env.reset(seed=...)`, constructs a new policy, and
uses a distinct output directory. It therefore saves only cold-start time and
does not share table/contact state between episodes.

The persistent root must be the parent of every replay output that will be
queued. Start this on the machine that owns Docker/Isaac (for example the
RTX5090 workstation), wait for `"state": "ready"`, then run the normal replay
command with that root exported:

```bash
cd /home/suzuki/GitHub/iros_2026_ramen
export FLIP_TABLE_PERSISTENT_EVAL_ROOT="$PWD/outputs/flip_table_real_to_sim"
evaluate/flip_table_simulation/persistent_eval.sh start
evaluate/flip_table_simulation/persistent_eval.sh status

evaluate/flip_table_simulation/real_to_sim_calibration/run_anchor_replay.sh \
  outputs/flip_table_real_to_sim/<run_id>/episodes/calibration_0250.json \
  outputs/flip_table_real_to_sim/<run_id>/calibration_0250_replay

# Stop only when no more jobs are needed; this closes Isaac and the container.
evaluate/flip_table_simulation/persistent_eval.sh stop
```

`start` waits until Isaac reports `"state": "ready"` (the first cold start
can take several minutes). Subsequent replays reuse that process and skip only
the Isaac startup; every queued evaluation still receives a new
`env.reset(seed=...)`. Use `restart` after changing the fixed USD, camera, or
foundation contact configuration. `force-stop` is reserved for an unresponsive
worker and discards any running job.

For a fixed-base replay, `replay.py materialize` saves diagnostic RGB after
the same recorded-camera remap used by policy camera tensors. This is required
for any raw-intrinsic real-vs-sim comparison; a raw pinhole PNG must never be
labelled as remapped evidence. `--initial-pose-only` stops at the terminal
warmup frame (step 119), before the first recorded action. It is the intended
fast path for ranking reset-only table/camera candidates, not a replay or
success evaluation.

The optional `$FLIP_TABLE_PERSISTENT_EVAL_ROOT/persistent_foundation.env`
contains only the four allowlisted reset-time arm-identification values. The
worker reads it before every cold start, records the values in `ready.json`,
and does not accept arbitrary shell code. This keeps a deliberate restart from
silently changing a calibration foundation. Per-job table/camera candidates
remain separate from this shared foundation.

`run_anchor_replay.sh` automatically selects the first local Python environment
that provides `numpy` and `pyarrow` (`tv`, Unitree, then xr-teleop). Override
this selection explicitly with `FLIP_TABLE_CALIBRATION_PYTHON=/path/to/python`
when using another environment.

The queue accepts only diagnostic replay/CV policy names and a strict allowlist
of reset/replay variables. It refuses unknown environment variables, absolute
or escaping paths, stale `test_0` output, and a missing worker. An offline
`FLIP_TABLE_CALIBRATION_TABLE_POSES_JSON` candidate is permitted per job and
is applied once during that job's reset; the worker clears it before the next
job. Calibration wrappers also request
`FLIP_TABLE_PERSISTENT_RECREATE_ENV=true`: this rebuilds the Gym/task
environment while retaining `SimulationApp`, so authored USD transforms and
task caches cannot leak from one candidate to the next without paying an Isaac
cold-start. It is not a policy server and does
not change the fixed USD, contact profile, or policy inputs while it is alive;
restart it after changing those foundation settings.

For a non-replay diagnostic policy, submit an explicit bounded job. The
worker resets the environment before it begins, and accepts only the policy
names listed in its `ready.json` plus allowlisted `--environment` values:

```bash
evaluate/flip_table_simulation/persistent_eval.sh submit \
  --policy-name CvRuleBasedPolicy \
  --time-out-limit 1800 \
  --seed 42 \
  --output-dir "$PWD/outputs/flip_table_real_to_sim/cv_rule_based_run" \
  --wait
```

Generate exact 640x480 focal/aperture candidates before camera fitting:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.camera \
  --calibration-dir outputs/flip_table_real_to_sim/<run_id>/calibration \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/camera_candidates
```

The two candidate files differ only in the unknown left/right D405 serial
assignment; selection must be made by held-out visual evidence.

## Calibration order

The order below is mandatory.  Do not compensate a camera/scene error by
changing contact parameters.

1. Pin and audit the dataset, then reserve the anchor, two calibration, and
   five held-out episodes before fitting any parameters.
2. Reset the simulator from the recorded `q_current` and `hand_state`, replay
   only recorded upper-body `q_desired`/hand commands, and compare simulator
   state to the recorded observed state.  Command tracking and observed-state
   matching are separate reports.  The replay trace also records the
   simulator-only white-table pose, velocity, and contact diagnostics when
   enabled.  Those values are offline evidence for distinguishing a physics
   mismatch from a camera mismatch; they must never enter policy inputs,
   actions, rewards, or inference-time control flow.
3. Fit head stereo, then left/right D405 link extrinsics, intrinsics, and
   timestamp offsets from RGB evidence, CAD, and FK.  The source stereo pair
   is available only for offline calibration; it is not policy input.
4. Fit fixed robot/table initial poses and visual assets under the accepted
   camera model.  Lighting/background/texture variation is added only after a
   fixed scene matches its reference.
5. Fit actuator, Dex1 collision, material friction/restitution, stiffness, and
   damping against table motion.  The 1.596 kg UTTER table mass is fixed.
6. Freeze shared parameters and evaluate the five unused episodes.  A failed
   gate is evidence of a failed calibration, not permission to tune a
   validation episode.

`actuator_identification.py` reports all 19 channels for auditability and
also reports `group_summaries.waist`, `.arms`, and `.dex1`.  An arm stiffness,
damping, armature, or friction candidate must be ranked only with
`group_summaries.arms` (14 shoulder/elbow/wrist channels).  The waist retains
the organizer WBC drive, while a Dex1 trace during table contact is not a
servo-only experiment.  Freeze a shared arm profile only after the two
reserved calibration episodes agree, then test it unchanged on the five
held-out episodes.

The RoboFinals evaluator binds a fixed host IPC port (`50000`).
`run_eval.sh` holds a non-blocking host lock for the full invocation so that a
second evaluator fails before creating an Isaac container.  Do not bypass this
lock: a port collision can attach a policy client to a stale environment and
invalidates a calibration trace.

The head stereo check derives metric scale from the two recorded RGB images and
the pinned 60.300 mm calibration baseline.  It is an offline diagnostic only:
neither its depth image nor its mask/point cloud may enter a policy, planner,
reward, or inference-time branch.

```bash
conda run -n tv python -m evaluate.flip_table_simulation.real_to_sim_calibration.stereo_geometry \
  --left outputs/flip_table_real_to_sim/<run_id>/real_rgb/anchor/frame_0000/head_left.png \
  --right outputs/flip_table_real_to_sim/<run_id>/real_rgb/anchor/frame_0000/head_right.png \
  --calibration outputs/flip_table_real_to_sim/<run_id>/calibration/head_camera_params.yaml \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/stereo_geometry/anchor_frame_0000
```

`stereo_diagnostics.json` contains the valid-depth fraction, table depth
quantiles, and calibration provenance.  It is accepted only as a scale check;
it does not establish robot-to-camera extrinsics by itself.

Fit a recorded initial scene directly from the dataset. Select three or more
pre-contact frames; partial and intermittently occluded rims are expected. The
fitter accepts a proposal only when CAD support is temporally stable and the
two head eyes agree. Its debug images show the actual registered CAD evidence.

The source fitter uses the exact V1 CAD rim and leg axes only. It deliberately
does not use the online policy's permissive quadrilateral/PnP fallback: an
inner brace can look like a plausible rectangle in one eye but is not valid
stereo calibration evidence. Since the recorded floating-base height is not a
known table-body height, source calibration searches one fixed, bounded body-Z
grid shared by every episode. This is an offline source-fit variable, not a
runtime camera or policy feature; a frame without a direct CAD wireframe fit is
reported unavailable rather than filled in with a monocular estimate.

```bash
conda run -n tv env PYTHONPATH=$PWD python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.source_cad_alignment \
  --source-root ~/.cache/huggingface/hub/datasets--Team-RAMEN--IROS2026_RAMEN_suzuki_flip_table_1/snapshots/10a6ec05f9993b8d59faad2957e47153b0f15f37 \
  --episode-index 250 --frames 0 10 20 30 40 50 \
  --urdf <g1_29dof_with_hand.urdf> \
  --stereo-calibration outputs/flip_table_real_to_sim/<run_id>/calibration/head_camera_params.yaml \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/source_cad_alignment_0250
```

Only a report with `accepted_for_fixed_scene_proposal: true` may seed a fixed
scene candidate. It remains insufficient to set friction, restitution, or
Dex1 contact stiffness; those require replay-motion evidence.

Convert an accepted source-CAD fit into a V1 workbench-local reset candidate
using a baseline V1 trace that records both the robot root and workbench pose.
The converter explicitly resolves the physical 180-degree yaw symmetry of the
assembled table by choosing the smallest equivalent yaw. It does not use
simulator state after reset and its output is forbidden from policy inputs.

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.source_scene_candidate \
  --source-alignment outputs/flip_table_real_to_sim/<run_id>/source_cad_alignment_0250/source_cad_alignment.json \
  --sim-trace outputs/flip_table_real_to_sim/<run_id>/baseline_replay/test_0/action_state_trace.jsonl \
  --output outputs/flip_table_real_to_sim/<run_id>/source_scene_candidate_0250.json
```

When probing the candidate, always source the same immutable replay runtime
environment before the probe environment. In particular, this preserves the
recorded initial `q_current`/Dex1 state; omitting it changes the head camera
pose and makes an image comparison meaningless. On a remote machine, override
only `FLIP_TABLE_SIM_OUTPUT_DIR` to its remote output path after sourcing both
files.

For initial-scene sweeps, run candidate resets in parallel rather than
restarting Isaac Sim for every candidate. Each candidate is a workbench-local
`[x, y, z]` offset plus tabletop yaw applied once at reset. A fixed-base replay
candidate also records the baseline V1 reset root position and yaw, so table
placement does not implicitly move G1 through the ordinary placement policy.
This is not a source-root estimate and is never updated after reset. A candidate may
also carry one shared `head_stereo_offset_local_m` and
`head_stereo_rotation_rpy_deg` applied identically to both head eyes. This is
an offline mount-identification probe, not domain randomization: the value is
fixed for the episode and never becomes a policy feature. The rendered PNGs
are diagnostic-only and are written under `frame_XXXX/env_NNN`:

The offset is the translation of the authored stereo-rig centre *after* the
same local rotation has been applied around that centre.  Derive it with
`source_head_mount_candidate.py`; do not substitute a left-eye translation,
because that has a different meaning whenever the mount rotation is nonzero.

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.parallel_scene_probe write-env \
  --candidates candidates.json \
  --replay-action-path replay_actions.json \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/parallel_probe
source outputs/flip_table_real_to_sim/<run_id>/replay_runtime.env
source outputs/flip_table_real_to_sim/<run_id>/parallel_probe/parallel_probe.env
evaluate/flip_table_simulation/run_eval.sh
python -m evaluate.flip_table_simulation.real_to_sim_calibration.parallel_scene_probe score \
  --real-image real_frame.png \
  --frame-dir outputs/flip_table_real_to_sim/<run_id>/parallel_probe/test_0/camera_frames/frame_0136 \
  --manifest outputs/flip_table_real_to_sim/<run_id>/parallel_probe/parallel_probe_manifest.json \
  --output outputs/flip_table_real_to_sim/<run_id>/parallel_probe/scores.json
```

Each probe also saves a diagnostic scene trace. Before ranking image scores,
run `source_projection_conformance.py` against that trace to prove that the
candidate reset actually realized the requested camera and table transforms.
This telemetry is offline-only and must never become a policy input.

If a source-derived reset has a systematic realized table/camera residual,
use `refine_scene_candidate_from_trace.py` once to produce a new candidate.
It corrects only the reset offset and planar yaw, checks workbench support,
and requires a fresh reset/render for validation; it is not a per-frame table
correction and cannot be used by a policy, planner, or reward.

This score only ranks a single RGB frame. Candidates whose PnP confidence or
reprojection error fail the same geometry gate are placed after reliable
candidates even when their corner overlap appears better; that pattern usually
means the detector selected an interior support rather than the tabletop rim.
It must not select a final calibration without multi-view fitting and held-out
episode validation.

For a batched candidate replay, score one environment explicitly. The report
aggregates both PnP and silhouette diagnostics over time, but remains
fail-closed when fewer than three real frames pass the PnP quality gate:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.compare_multiframe_head \
  --real-root outputs/flip_table_real_to_sim/<run_id>/real_rgb/calibration_0250 \
  --sim-root outputs/flip_table_real_to_sim/<run_id>/candidate_replay \
  --frame-map 0:119,10:136,449:868 \
  --environment-index 0 --sim-recorded-geometry \
  --output outputs/flip_table_real_to_sim/<run_id>/candidate_replay/head_alignment_env000.json
```

`parallel_scene_probe` also records a head-image white-table silhouette IoU
and symmetric edge distance.  These measurements are more robust than a
four-corner fit when the real tabletop rim is partially hidden by a hand, leg,
or highlight.  They are candidate-ranking diagnostics only; they do not
produce a table pose, modify a policy image, or relax the multi-frame and
held-out acceptance gates.

Extract synchronized, unmodified RGB references for all eight selected episodes:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.extract_visual_evidence \
  --manifest outputs/flip_table_real_to_sim/<run_id>/calibration_manifest.json \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/real_rgb
```

`replay materialize` writes the matching source-frame to simulator-step map in
`replay_actions.json` and exports all of those simulator frames. Use that map
for real/sim comparison; do not manually shift videos or select favorable
frames.

When a CAD/stereo fit needs frames beyond the standard six visual-evidence
samples, request those exact source indices before starting the replay. This
does not alter the target stream or the physical scene; it only exports the
matching diagnostic RGB frame at the deterministic 30 Hz to 50 Hz replay step:

```bash
FLIP_TABLE_REPLAY_CAMERA_SOURCE_FRAMES=20,30,40,50 \
  evaluate/flip_table_simulation/real_to_sim_calibration/run_anchor_replay.sh \
  outputs/flip_table_real_to_sim/<run_id>/episodes/calibration_0250.json \
  outputs/flip_table_real_to_sim/<run_id>/calibration_0250_replay
```

For table dynamics, use only source frames with accepted *paired* head-stereo
CAD fits. `table_motion_comparison.py` compares their table motion relative to
the first mutually visible pose with the trace's root-relative table pose.
This deliberately separates absolute reset error (camera/CAD reprojection)
from relative contact motion. A dynamic alignment is expected to fail the
*static reset-pose* temporal-spread gate; the motion comparator instead
requires its per-frame stereo gate and does not reuse it as a reset candidate.
It emits no table or phase metric if fewer than three pairs exist, or if the
source left/right pose disagreement exceeds 5 mm translation p95 or 0.75
degrees rotation p95. Those are observation-quality limits, not tuning
targets.

`temporal_cad_tracker.py` is a lighter offline observation tool for inspecting
which source frames retain usable table geometry. It uses the calibrated head
stereo RGB pair, RGB-derived left/right disparity consistency, encoder FK, and
the fixed table CAD only. A one-eye CAD fit is retained as debug evidence but
can never seed a reset pose; initial-scene fitting requires at least three
stereo-consistent frame pairs. Unobserved or internally inconsistent dynamic
frames remain explicitly unavailable for contact fitting.

```bash
conda run -n tv env PYTHONPATH=$PWD python -m \
  evaluate.flip_table_simulation.real_to_sim_calibration.temporal_cad_tracker \
  --source-root <pinned_hf_snapshot> --episode-index 250 \
  --start-frame 0 --end-frame 100 --stride 10 \
  --urdf <g1_29dof_with_hand.urdf> \
  --stereo-calibration outputs/flip_table_real_to_sim/<run_id>/calibration/head_camera_params.yaml \
  --initial-alignment outputs/flip_table_real_to_sim/<run_id>/source_cad_alignment_0250/source_cad_alignment.json \
  --cad-mesh data/flip_table_data_augmentation/outputs/source/v1-table-mesh/Table001_assembled_body_frame.obj \
  --output-dir outputs/flip_table_real_to_sim/<run_id>/temporal_stereo_cad_0250
```

Combine only independently accepted source head-mount proposals before a
fixed-scene probe. The consensus threshold is deliberately stricter than a
probe acceptance and its output is never a shared simulator default by itself:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.source_head_mount_consensus \
  --report outputs/flip_table_real_to_sim/<run_id>/source_head_mount_candidate_0250.json \
  --report outputs/flip_table_real_to_sim/<run_id>/source_head_mount_candidate_0509.json \
  --output outputs/flip_table_real_to_sim/<run_id>/source_head_mount_consensus.json
```

`source_head_mount_candidate.py` emits an **incremental** correction around
the reset candidate that produced its simulator trace. When a candidate
already contains a non-zero head offset or RPY correction, the tool composes
the increment instead of replacing those fields: the task-local translation
is added and the intrinsic-XYZ rotation is left-composed
(`R_increment @ R_candidate`). This prevents repeated offline calibration
passes from silently discarding an earlier mounted-camera correction. The
composition is recorded in every candidate report and remains reset-only
evidence, never a runtime camera correction.

The report carries this same candidate-relative value in both `correction`
(the stable consensus-tool contract) and `incremental_correction` (its
unambiguous semantic name). The consensus tool also reads older reports that
only contain `incremental_correction`; it never treats either field as an
absolute simulator camera setting.

Compose that shared head correction with one episode-specific table reset only
for a fixed-scene probe. This does not change the simulator default:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.fixed_scene_probe_candidate \
  --scene-candidate outputs/flip_table_real_to_sim/<run_id>/source_scene_candidate_0509.json \
  --head-mount-consensus outputs/flip_table_real_to_sim/<run_id>/source_head_mount_consensus.json \
  --output outputs/flip_table_real_to_sim/<run_id>/fixed_scene_probe_0509.json
```

For episodes where the tabletop is partly hidden during manipulation, the
offline FoundationPose pipeline may provide a stricter replacement source.
It uses only recorded RGB, RGB-derived stereo depth, robot FK, the fixed CAD,
and the three recorded camera views. The motion comparator accepts this
artifact only when its own tracker gate passed **and** the forward/reverse
residual is at most 5 mm p95 and 0.75 degrees p95. It then reads only
hash-verified, per-frame rendered-evidence observations; interpolated rows and
terminal forward-only predictions are excluded. This makes the artifact
eligible for offline contact identification, never for policy input or runtime
control. It is still not an external ground-truth measurement, so retained
residual risk must be reported with any fitted contact parameters.

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.table_motion_comparison \
  --source-alignment outputs/flip_table_real_to_sim/<run_id>/source_cad_alignment_dynamic.json \
  --replay-actions outputs/flip_table_real_to_sim/<run_id>/calibration_0250_replay/replay_actions.json \
  --sim-trace outputs/flip_table_real_to_sim/<run_id>/calibration_0250_replay/test_0/action_state_trace.jsonl \
  --output outputs/flip_table_real_to_sim/<run_id>/calibration_0250_replay/table_motion.json
```

For a held-out report, add the immutable source episode index to the head
comparison and pass both typed artifacts. The report accepts their metrics
only when their schema and episode index match the replay bundle; it does not
infer a pass from image similarity alone.

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.compare_multiframe_head \
  --real-root outputs/flip_table_real_to_sim/<run_id>/real_rgb/validation_0009 \
  --sim-root outputs/flip_table_real_to_sim/<run_id>/validation_0009 \
  --frame-map 0:119,10:137,266:563 \
  --source-episode-index 9 --sim-recorded-geometry \
  --output outputs/flip_table_real_to_sim/<run_id>/validation_0009/head_comparison.json
```

## Held-out acceptance

The five `validation_*` episodes in the immutable calibration manifest are the
only release gate.  Use one fixed shared-parameter digest across all five
episodes, and write one report per episode with measurements from recorded
real RGB/CAD geometry, the recorded joint replay, and the corresponding
simulator replay.  A missing value is a failure: this command never infers a
pass from an anchor, a calibration episode, or a source-only fit.

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.heldout_validation \
  --calibration-manifest outputs/flip_table_real_to_sim/<run_id>/calibration_manifest.json \
  --episode-report outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0009.json \
  --episode-report outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0066.json \
  --episode-report outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0074.json \
  --episode-report outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0308.json \
  --episode-report outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0338.json \
  --output outputs/flip_table_real_to_sim/<run_id>/heldout_acceptance.json
```

Each report must use schema
`team_ramen_flip_table_heldout_validation/v1` and include a
`shared_parameters_path` and the matching `shared_parameter_sha256`, plus all of: camera reprojection median/p95,
upper-body joint RMSE, table translation/rotation RMSE, phase timing error,
and mask IoU.  The output is calibration evidence only; it is forbidden from
policy, planner, reward, or inference-time inputs.

Generate the per-episode skeleton from a replay trace rather than typing the
joint metric.  This intentionally leaves every visual and table-motion metric
missing until a dedicated comparison artifact supplies it:

```bash
python -m evaluate.flip_table_simulation.real_to_sim_calibration.heldout_episode_report \
  --episode-bundle outputs/flip_table_real_to_sim/<run_id>/episodes/validation_0009.json \
  --trace outputs/flip_table_real_to_sim/<run_id>/validation_0009/test_0/action_state_trace.jsonl \
  --shared-parameters outputs/flip_table_real_to_sim/<run_id>/fixed_parameters.json \
  --head-comparison outputs/flip_table_real_to_sim/<run_id>/validation_0009/head_comparison.json \
  --table-motion-comparison outputs/flip_table_real_to_sim/<run_id>/validation_0009/table_motion.json \
  --output outputs/flip_table_real_to_sim/<run_id>/heldout/validation_0009.json
```
