# Real Table-Pose Calibration Capture

The flip-table demonstrations do not consistently expose all four physical
UTTER tabletop corners to the head stereo pair. Do not infer missing corners
from a leg, interior rib, shadow, or workbench edge.

Capture this calibration sequence before fitting table pose or contact physics:

1. Keep the G1 root and lower body stationary. Record `q_current` at 30 Hz and
   raw `cam_0`/`cam_1` head stereo at 640x480 with their original timestamps.
2. Place the 1.596 kg UTTER table on the workbench in each of three yaw poses.
   At every pose, keep hands outside the tabletop silhouette and expose all four
   physical outer tabletop corners to both head eyes for at least two seconds.
3. Repeat with each D405 observing a printed ChArUco or AprilTag board rigidly
   fixed to the tabletop. The board's transform to the CAD tabletop frame must
   be measured once with a ruler or jig.
4. For each static pose, save four manually reviewed stereo corner
   correspondences using `table_corner_annotations.template.json`. Reject any
   frame with occlusion, motion blur, or uncertain corner identity.
5. Fit camera-to-link and table-root transforms from the static captures first.
   Only then replay an assembly demonstration and fit contact parameters to the
   measured table trajectory. The capture is offline calibration evidence, not
   a policy input or a runtime sensor requirement.

The existing source demonstrations remain suitable for arm-response fitting.
They are not sufficient by themselves to prove a 20 mm / 3 degree table-pose
gate when the tabletop rim is occluded.
