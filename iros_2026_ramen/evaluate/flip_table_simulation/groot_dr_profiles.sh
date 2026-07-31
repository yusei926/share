#!/usr/bin/env bash

# Fixed simulator profiles used for model selection and the final held-out gate.
# The held-out profile keeps the same real-world limits, but reserves appearance
# categories and the difficult low-friction/high-restitution contact band.
groot_apply_dr_profile() {
  local profile="${1:?domain-randomization profile is required}"

  case "$profile" in
    nominal_v1)
      export FLIP_TABLE_GROOT_DR_PROFILE="nominal_v1"
      export FLIP_TABLE_ROOM_FLOOR_MATERIALS="oak_wood,rough_concrete,ceramic_tile,industrial_vinyl"
      export FLIP_TABLE_ROOM_WALL_MATERIALS="painted_plaster,rough_concrete,red_brick,oak_panels"
      export FLIP_TABLE_ROOM_FLOOR_PATTERNS="grid,checker,planks,border"
      export FLIP_TABLE_ROOM_WALL_PATTERNS="plain,baseboard,horizontal_stripes,vertical_panels,wainscot"
      export FLIP_TABLE_ROOM_PROP_ASSETS="Chair,Desk,Shelf,Cabinet,Crates,Plant"
      export FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE="0.65,0.95"
      export FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE="0.48,0.64"
      export FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE="0.02,0.08"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE="0.50,0.75"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE="0.35,0.46"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE="0.01,0.05"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE="0.60,0.90"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE="0.42,0.56"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE="0.02,0.08"
      export FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K="3800,6500"
      export FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K="5000,7000"
      export FLIP_TABLE_LIGHT_INTENSITY_RANGE="450,1200"
      export FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE="180,750"
      export FLIP_TABLE_LIGHT_EXPOSURE_RANGE="-0.35,0.35"
      ;;
    validation_v1)
      export FLIP_TABLE_GROOT_DR_PROFILE="validation_v1"
      export FLIP_TABLE_ROOM_FLOOR_MATERIALS="oak_wood,rough_concrete,ceramic_tile"
      export FLIP_TABLE_ROOM_WALL_MATERIALS="painted_plaster,rough_concrete,oak_panels"
      export FLIP_TABLE_ROOM_FLOOR_PATTERNS="grid,checker,planks"
      export FLIP_TABLE_ROOM_WALL_PATTERNS="plain,baseboard,horizontal_stripes,vertical_panels"
      export FLIP_TABLE_ROOM_PROP_ASSETS="Chair,Desk,Shelf,Cabinet,Crates"
      export FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE="0.70,0.90"
      export FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE="0.52,0.60"
      export FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE="0.03,0.07"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE="0.55,0.70"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE="0.38,0.44"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE="0.02,0.04"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE="0.65,0.85"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE="0.45,0.53"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE="0.03,0.07"
      export FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K="4300,6500"
      export FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K="5000,6500"
      export FLIP_TABLE_LIGHT_INTENSITY_RANGE="600,1200"
      export FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE="300,750"
      export FLIP_TABLE_LIGHT_EXPOSURE_RANGE="-0.20,0.35"
      export FLIP_TABLE_RL_CAMERA_POSITION_JITTER_M="0.002"
      export FLIP_TABLE_RL_CAMERA_ROTATION_JITTER_DEG="0.7"
      export FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS="1"
      export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS="1"
      export FLIP_TABLE_JOINT_NOISE_RAD="0.015"
      export FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE="0.4"
      ;;
    held_out_v1)
      export FLIP_TABLE_GROOT_DR_PROFILE="held_out_v1"
      export FLIP_TABLE_ROOM_FLOOR_MATERIALS="industrial_vinyl"
      export FLIP_TABLE_ROOM_WALL_MATERIALS="red_brick"
      export FLIP_TABLE_ROOM_FLOOR_PATTERNS="border"
      export FLIP_TABLE_ROOM_WALL_PATTERNS="wainscot"
      export FLIP_TABLE_ROOM_PROP_ASSETS="Plant"
      export FLIP_TABLE_CONTACT_HAND_WHITE_STATIC_RANGE="0.65,0.70"
      export FLIP_TABLE_CONTACT_HAND_WHITE_DYNAMIC_RANGE="0.48,0.52"
      export FLIP_TABLE_CONTACT_HAND_WHITE_RESTITUTION_RANGE="0.07,0.08"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_STATIC_RANGE="0.50,0.55"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_DYNAMIC_RANGE="0.35,0.38"
      export FLIP_TABLE_CONTACT_WHITE_WORKBENCH_RESTITUTION_RANGE="0.04,0.05"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_STATIC_RANGE="0.60,0.65"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_DYNAMIC_RANGE="0.42,0.45"
      export FLIP_TABLE_CONTACT_WORKBENCH_HAND_RESTITUTION_RANGE="0.07,0.08"
      export FLIP_TABLE_INDOOR_LIGHT_TEMPERATURE_K="3800,4300"
      export FLIP_TABLE_SUN_LIGHT_TEMPERATURE_K="6500,7000"
      export FLIP_TABLE_LIGHT_INTENSITY_RANGE="450,600"
      export FLIP_TABLE_SUN_LIGHT_INTENSITY_RANGE="180,300"
      export FLIP_TABLE_LIGHT_EXPOSURE_RANGE="-0.35,-0.20"
      export FLIP_TABLE_RL_CAMERA_POSITION_JITTER_M="0.003"
      export FLIP_TABLE_RL_CAMERA_ROTATION_JITTER_DEG="1.0"
      export FLIP_TABLE_RL_CAMERA_LATENCY_MAX_STEPS="2"
      export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS="2"
      export FLIP_TABLE_JOINT_NOISE_RAD="0.02"
      export FLIP_TABLE_UPPER_BODY_POSE_RANGE_SCALE="0.5"
      ;;
    *)
      echo "ERROR: unknown GR00T DR profile: $profile" >&2
      return 2
      ;;
  esac
}
