#!/usr/bin/env bash
set -euo pipefail

# Build the official Dex1-1 serial-to-DDS service with the repository's narrow
# thread-safety/health patch.  The script never calibrates or commands a hand.
# Restart is an explicit opt-in because even a service-only change belongs in
# a supervised robot maintenance window.

OFFICIAL_REPOSITORY="https://github.com/unitreerobotics/dex1_1_service.git"
OFFICIAL_REVISION="cdd9fc5a78d51521eb262a56e0c5c19770700932"
SOURCE_DIR="/home/unitree/dex1_1_service"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PATCH_PATH="$REPOSITORY_ROOT/inference/desktop/xr/patches/dex1_1_service_thread_safety.patch"
SERVICE_PATH="$REPOSITORY_ROOT/inference/desktop/xr/systemd/dex1_1_gripper_server.service"
RESTART=false

usage() {
  cat <<EOF
Usage: $(basename "$0") [--restart]

Build and install the pinned official Dex1-1 service with the Team RAMEN
thread-safety and health patch. Without --restart, no running process changes.

EOF
}

while (($#)); do
  case "$1" in
    --restart) RESTART=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

[[ -r "$PATCH_PATH" ]] || { echo "ERROR: missing patch: $PATCH_PATH" >&2; exit 1; }
[[ -r "$SERVICE_PATH" ]] || { echo "ERROR: missing service unit: $SERVICE_PATH" >&2; exit 1; }

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone "$OFFICIAL_REPOSITORY" "$SOURCE_DIR"
fi

git -C "$SOURCE_DIR" fetch origin main
git -C "$SOURCE_DIR" cat-file -e "$OFFICIAL_REVISION^{commit}"

if git -C "$SOURCE_DIR" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  CURRENT_BASE="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
  [[ "$CURRENT_BASE" == "$OFFICIAL_REVISION" ]] || {
    echo "ERROR: already-patched checkout has unexpected HEAD: $CURRENT_BASE" >&2
    exit 1
  }
  echo "Dex1 hardening patch is already applied."
else
  # The official service writes a runtime log in its source directory on some
  # Orin images.  Ignore untracked runtime artifacts, but never overwrite a
  # tracked source/configuration change.
  [[ -z "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]] || {
    echo "ERROR: refusing to patch a dirty Dex1 service checkout: $SOURCE_DIR" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 1
  }
  git -C "$SOURCE_DIR" checkout --detach "$OFFICIAL_REVISION"
  git -C "$SOURCE_DIR" apply --check "$PATCH_PATH"
  git -C "$SOURCE_DIR" apply "$PATCH_PATH"
fi

cmake -S "$SOURCE_DIR" -B "$SOURCE_DIR/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$SOURCE_DIR/build" --parallel "$(nproc)"
test -x "$SOURCE_DIR/bin/dex1_1_gripper_server"

sudo install -m 0644 "$SERVICE_PATH" /etc/systemd/system/dex1_1_gripper_server.service
sudo systemctl daemon-reload

echo "Built hardened Dex1 service from $OFFICIAL_REVISION."
echo "Patch SHA-256: $(sha256sum "$PATCH_PATH" | awk '{print $1}')"
if [[ "$RESTART" == true ]]; then
  sudo systemctl restart dex1_1_gripper_server.service
  sudo systemctl --no-pager --full status dex1_1_gripper_server.service
else
  echo "Service was not restarted. Re-run with --restart during a supervised maintenance window."
fi
