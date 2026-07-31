#!/usr/bin/env bash
# Disable PCsensor FootSwitch factory keyboard/mouse mappings outside AVP teleop.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE="$ROOT_DIR/inference/desktop/xr/udev/99-pcsensor-footswitch-teleop.rules"
TARGET="/etc/udev/rules.d/99-pcsensor-footswitch-teleop.rules"

[[ -f "$SOURCE" ]] || { echo "ERROR: missing rule: $SOURCE" >&2; exit 1; }
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx "plugdev"; then
  echo "ERROR: $USER is not in the plugdev group required for this pedal." >&2
  echo "Ask an administrator to run: sudo usermod -aG plugdev $USER" >&2
  echo "Then log out and back in before retrying." >&2
  exit 1
fi
sudo install -m 0644 "$SOURCE" "$TARGET"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=input
echo "Installed $TARGET. Unplug and reconnect the PCsensor FootSwitch once."
