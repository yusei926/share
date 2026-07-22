#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
IMAGE="${ROBOFINALS_IMAGE:-paperc/robofinals:RoboFinals-IKEA-V1}"
RUNTIME_ROOT="${FLIP_TABLE_OBJECT_POSE_RUNTIME:-$HOME/.cache/team-ramen/flip-table-object-pose}"
# Keep the compiler matched to V1's torch 2.10.0+cu128 without changing the
# workstation-wide CUDA alternative. The digest is the linux/amd64 manifest
# for NVIDIA's official Ubuntu 22.04 CUDA 12.8.1 devel image.
CUDA_HOME_HOST="${CUDA_HOME_HOST:-$RUNTIME_ROOT/cuda-12.8}"
CUDA_TOOLKIT_IMAGE="${CUDA_TOOLKIT_IMAGE:-nvidia/cuda@sha256:6617a625f4090c76c545a0e7d63f2e441718ef9af7f4efe7dd1242a29e289fd7}"
# A configured corporate or regional mirror is unnecessary here: every Python
# wheel is hash-pinned, and the first download should come from the canonical
# index used for the preceding metadata audit.
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
RUNTIME_MODE="${FLIP_TABLE_AUG_RUNTIME_MODE:-auto}"
ROOTFS="${ROBOFINALS_ROOTFS:-/workspace/robofinals_rootfs_v1}"
OCI_LAYOUT="${ROBOFINALS_OCI_LAYOUT:-/workspace/robofinals_oci_v1}"
EXPECTED_IMAGE_DIGEST="$(python3 - "$ROOT_DIR/data/flip_table_data_augmentation/configs/pipeline_v1.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["container_digest"])
PY
)"

if [[ "${INSTALL_VERIFIED:-0}" != "1" ]]; then
  echo "ERROR: dependency installation requires the audited INSTALL_VERIFIED=1 acknowledgement" >&2
  exit 2
fi
prepare_cuda_toolkit() {
  if [[ -x "$CUDA_HOME_HOST/bin/nvcc" ]]; then
    return
  fi
  command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || {
    echo "ERROR: CUDA 12.8 compiler is missing and Docker is unavailable: $CUDA_HOME_HOST/bin/nvcc" >&2
    exit 1
  }
  case "$CUDA_HOME_HOST" in
    "$RUNTIME_ROOT"/*) ;;
    *)
      echo "ERROR: automatic CUDA extraction only permits a runtime-local CUDA_HOME_HOST" >&2
      exit 1
      ;;
  esac
  mkdir -p "$RUNTIME_ROOT"
  temporary="$(mktemp -d "${RUNTIME_ROOT}.cuda.XXXXXX")"
  container=""
  cleanup_cuda_toolkit() {
    [[ -z "$container" ]] || docker rm -f "$container" >/dev/null 2>&1 || true
    rm -rf "$temporary"
  }
  trap cleanup_cuda_toolkit EXIT
  docker pull "$CUDA_TOOLKIT_IMAGE" >&2
  platform="$(docker image inspect "$CUDA_TOOLKIT_IMAGE" --format '{{.Os}}/{{.Architecture}}')"
  [[ "$platform" == "linux/amd64" ]] || {
    echo "ERROR: pinned CUDA image has unexpected platform: $platform" >&2
    exit 1
  }
  container="$(docker create "$CUDA_TOOLKIT_IMAGE")"
  docker cp "$container:/usr/local/cuda-12.8/." "$temporary"
  [[ -x "$temporary/bin/nvcc" ]] || {
    echo "ERROR: pinned CUDA image lacks /usr/local/cuda-12.8/bin/nvcc" >&2
    exit 1
  }
  version="$($temporary/bin/nvcc --version | grep -F -o 'release 12.8' || true)"
  [[ "$version" == "release 12.8" ]] || {
    echo "ERROR: extracted CUDA compiler is not release 12.8" >&2
    exit 1
  }
  mv "$temporary" "$CUDA_HOME_HOST"
  temporary=""
  trap - EXIT
}

prepare_cuda_toolkit
if [[ "$RUNTIME_MODE" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    RUNTIME_MODE="docker"
  else
    RUNTIME_MODE="direct"
  fi
fi
if [[ "$RUNTIME_MODE" != "docker" && "$RUNTIME_MODE" != "direct" ]]; then
  echo "ERROR: FLIP_TABLE_AUG_RUNTIME_MODE must be auto, docker, or direct" >&2
  exit 2
fi
if [[ "$RUNTIME_MODE" == "docker" ]]; then
  OBSERVED_IMAGE_DIGEST="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
  PREPARE_PYTHON="${PREPARE_PYTHON:-$ROOT_DIR/model/subtask_policy_training/.venv/bin/python}"
else
  for path in "$OCI_LAYOUT/manifest.json" "$ROOTFS/opt/conda/envs/robofinals/bin/python"; do
    [[ -e "$path" ]] || {
      echo "ERROR: direct V1 runtime is incomplete; run setup_vast.sh first: $path" >&2
      exit 1
    }
  done
  OBSERVED_IMAGE_DIGEST="sha256:$(sha256sum "$OCI_LAYOUT/manifest.json" | awk '{print $1}')"
  PREPARE_PYTHON="$ROOTFS/opt/conda/envs/robofinals/bin/python"
fi
if [[ "$OBSERVED_IMAGE_DIGEST" != "$EXPECTED_IMAGE_DIGEST" ]]; then
  echo "ERROR: V1 image digest mismatch: $OBSERVED_IMAGE_DIGEST" >&2
  exit 1
fi
[[ -x "$PREPARE_PYTHON" ]] || {
  echo "ERROR: object-pose preparation Python is missing: $PREPARE_PYTHON" >&2
  exit 1
}

PYTHONPATH="$ROOT_DIR" \
  "$PREPARE_PYTHON" \
  -m data.flip_table_data_augmentation.scripts.prepare_object_pose_runtime \
  --runtime-root "$RUNTIME_ROOT"

if [[ "$RUNTIME_MODE" == "docker" ]]; then
  exec docker run --rm --gpus all --network host --ipc host \
  -e INSTALL_VERIFIED=1 \
  -e PIP_INDEX_URL="$PIP_INDEX_URL" \
  -e ROBOFINALS_IMAGE_DIGEST="$OBSERVED_IMAGE_DIGEST" \
  -e CUDA_HOME=/usr/local/cuda-12.8 \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;12.0+PTX}" \
  -v "$ROOT_DIR":/repo:ro \
  -v "$RUNTIME_ROOT":/runtime \
  -v "$CUDA_HOME_HOST":/usr/local/cuda-12.8:ro \
  "$IMAGE" -lc '
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate robofinals
export PATH="$CUDA_HOME/bin:$PATH"

if [[ ! -x /runtime/venv/bin/python ]]; then
  python -m venv --system-site-packages /runtime/venv
fi
source /runtime/venv/bin/activate
python -m pip install --require-hashes --no-deps --no-build-isolation \
  -r /repo/data/flip_table_data_augmentation/object_pose/runtime-requirements.txt
if ! python -c "import torch, pytorch3d, pytorch3d._C" >/dev/null 2>&1; then
  python -m pip install --no-build-isolation --no-deps /runtime/pytorch3d
fi
if ! python -c "import nvdiffrast" >/dev/null 2>&1; then
  python -m pip install --no-build-isolation --no-deps /runtime/nvdiffrast
fi

cmake -S /runtime/FoundationPose/mycpp -B /runtime/FoundationPose/mycpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_ROOT_DIR=/runtime/venv \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build /runtime/FoundationPose/mycpp/build -j"$(nproc)"

PYTHONPATH=/runtime/FoundationPose:/runtime/FoundationPose/mycpp/build:/repo \
python /repo/data/flip_table_data_augmentation/scripts/verify_object_pose_runtime.py \
  --config /repo/data/flip_table_data_augmentation/configs/pipeline_v1.json \
  --runtime-root /runtime \
  --observed-image-digest "$ROBOFINALS_IMAGE_DIGEST"
'
fi

export INSTALL_VERIFIED=1
export PIP_INDEX_URL
export ROBOFINALS_IMAGE_DIGEST="$OBSERVED_IMAGE_DIGEST"
export CUDA_HOME="$CUDA_HOME_HOST"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;12.0+PTX}"
export PATH="$CUDA_HOME/bin:$ROOTFS/opt/conda/envs/robofinals/bin:$PATH"
export LD_LIBRARY_PATH="$ROOTFS/opt/conda/envs/robofinals/lib:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ ! -x "$RUNTIME_ROOT/venv/bin/python" ]]; then
  "$PREPARE_PYTHON" -m venv --system-site-packages "$RUNTIME_ROOT/venv"
fi
VENV_PYTHON="$RUNTIME_ROOT/venv/bin/python"
"$VENV_PYTHON" -m pip install --require-hashes --no-deps --no-build-isolation \
  -r "$ROOT_DIR/data/flip_table_data_augmentation/object_pose/runtime-requirements.txt"
if ! "$VENV_PYTHON" -c "import torch, pytorch3d, pytorch3d._C" >/dev/null 2>&1; then
  "$VENV_PYTHON" -m pip install --no-build-isolation --no-deps "$RUNTIME_ROOT/pytorch3d"
fi
if ! "$VENV_PYTHON" -c "import nvdiffrast" >/dev/null 2>&1; then
  "$VENV_PYTHON" -m pip install --no-build-isolation --no-deps "$RUNTIME_ROOT/nvdiffrast"
fi

cmake -S "$RUNTIME_ROOT/FoundationPose/mycpp" \
  -B "$RUNTIME_ROOT/FoundationPose/mycpp/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_ROOT_DIR="$RUNTIME_ROOT/venv" \
  -Dpybind11_DIR="$("$VENV_PYTHON" -m pybind11 --cmakedir)"
cmake --build "$RUNTIME_ROOT/FoundationPose/mycpp/build" -j"$(nproc)"

PYTHONPATH="$RUNTIME_ROOT/FoundationPose:$RUNTIME_ROOT/FoundationPose/mycpp/build:$ROOT_DIR" \
  "$VENV_PYTHON" \
  "$ROOT_DIR/data/flip_table_data_augmentation/scripts/verify_object_pose_runtime.py" \
  --config "$ROOT_DIR/data/flip_table_data_augmentation/configs/pipeline_v1.json" \
  --runtime-root "$RUNTIME_ROOT" \
  --observed-image-digest "$ROBOFINALS_IMAGE_DIGEST"
