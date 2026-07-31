#!/usr/bin/env bash
# Install the camera-only AVP server as a recoverable Orin systemd service.
# This installer never starts/restarts the service unless --restart is given.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TEMPLATE="$REPOSITORY_ROOT/inference/desktop/xr/systemd/avp_teleimager.service.in"
UNIT_NAME="avp_teleimager.service"
SERVICE_USER="${AVP_TELEIMAGER_SERVICE_USER:-$(id -un)}"
TELEIMAGER_PYTHON="${XR_TELEIMAGER_PYTHON:-/home/unitree/miniconda3/envs/teleimager/bin/python}"
RESTART=0

case "${1:-}" in
  "") ;;
  --restart) RESTART=1 ;;
  -h|--help)
    echo "Usage: $(basename "$0") [--restart]"
    exit 0
    ;;
  *)
    echo "Usage: $(basename "$0") [--restart]" >&2
    exit 2
    ;;
esac
[[ $# -le 1 ]] || { echo "Usage: $(basename "$0") [--restart]" >&2; exit 2; }
[[ -f "$TEMPLATE" ]] || { echo "ERROR: missing $TEMPLATE" >&2; exit 1; }
id "$SERVICE_USER" >/dev/null
[[ -x "$TELEIMAGER_PYTHON" ]] || {
  echo "ERROR: TeleImager Python is missing: $TELEIMAGER_PYTHON" >&2
  exit 1
}
if ! "$TELEIMAGER_PYTHON" -c \
    'from importlib.metadata import version; assert version("mcap") == "1.3.0"' \
    >/dev/null 2>&1; then
  "$TELEIMAGER_PYTHON" -m pip install \
    --disable-pip-version-check "mcap==1.3.0"
fi

temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT
sed \
  -e "s|@REPOSITORY_ROOT@|$REPOSITORY_ROOT|g" \
  -e "s|@SERVICE_USER@|$SERVICE_USER|g" \
  "$TEMPLATE" >"$temporary"
sudo install -m 0644 "$temporary" "/etc/systemd/system/$UNIT_NAME"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT_NAME"

echo "Installed and enabled $UNIT_NAME for user $SERVICE_USER."
echo "No running camera process was changed."
if [[ "$RESTART" == 1 ]]; then
  echo "Restarting the camera-only service; no Unitree DDS publisher is involved."
  sudo systemctl restart "$UNIT_NAME"
  sudo systemctl --no-pager --full status "$UNIT_NAME"
fi
