#!/usr/bin/env bash
# Orin上で4視点（head左右 + D405左右）用のCompressedImageを無劣化記録する。
# このscriptはcontainer内で実行する。G1の制御commandは一切送らない。
# ROS 2 setup.bash は未定義のAMENT_TRACE_SETUP_FILESを参照することがあるため、
# sourceより前にnounsetを有効化しない。
set -eo pipefail

recordings_root="${1:-/recordings}"
session_name="${2:-yolo_fourcam_$(date +%Y%m%d_%H%M%S)}"
output_dir="${recordings_root}/${session_name}"

if [[ -e "${output_dir}" ]]; then
  echo "ERROR: output already exists: ${output_dir}" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
set -u

# recorder containerをhost userで実行しても、image内にそのUIDのHOMEがない場合がある。
# ROS log directoryを/tmpに固定して、記録開始前のlogger初期化失敗を防ぐ。
export HOME="${HOME:-/tmp}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${HOME}/.ros/log}"
mkdir -p "${ROS_LOG_DIR}"

echo "[recording] output=${output_dir}"
echo "[recording] head packed stereo + wrist_left/right RGB; no robot command is sent"
echo "[recording] Ctrl-C to finalize rosbag"

# sqlite3 profile=none はこのROS2 versionで最大throughputを優先する設定。
# 512MiB cacheは3 streamの短期I/O jitterを吸収する。host bind mountへ書くため、
# docker overlayへの書込みでcamera messageを落とさない。
exec ros2 bag record \
  --storage-preset-profile none \
  --max-cache-size 536870912 \
  -o "${output_dir}" \
  /head/camera/color/image_raw/compressed \
  /wrist_left/camera/color/image_raw/compressed \
  /wrist_right/camera/color/image_raw/compressed
