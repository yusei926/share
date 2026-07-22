#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${1:-${FLIP_TABLE_GROOT_RUNTIME_DIR:-/workspace/flip_table_groot_runtime}}"
BASE_PYTHON="${FLIP_TABLE_GROOT_BASE_PYTHON:-${CONDA_PREFIX:-/opt/conda/envs/robofinals}/bin/python}"
UV_BIN="${FLIP_TABLE_GROOT_UV_BIN:-$(command -v uv || true)}"

if [[ ! -x "$BASE_PYTHON" ]]; then
  echo "ERROR: base Python is not executable: $BASE_PYTHON" >&2
  exit 1
fi
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  echo "ERROR: uv is required to prepare the isolated GR00T runtime" >&2
  exit 1
fi

mkdir -p "$RUNTIME_DIR"
"$UV_BIN" venv \
  --allow-existing \
  --python "$BASE_PYTHON" \
  "$RUNTIME_DIR/.venv"
"$UV_BIN" pip install \
  --python "$RUNTIME_DIR/.venv/bin/python" \
  --torch-backend cu128 \
  --requirements "$SOURCE_DIR/requirements.txt"

"$RUNTIME_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata

import numpy
import torch

from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.processor_groot import (
    GrootN17ActionDecodeStep,
    GrootN17PackInputsStep,
    make_groot_pre_post_processors_from_pretrained,
)

version = importlib.metadata.version("lerobot")
if version != "0.6.0":
    raise SystemExit(f"expected lerobot==0.6.0, found {version}")
print(
    "GR00T evaluation runtime ready: "
    f"lerobot={version}, torch={torch.__version__}, numpy={numpy.__version__}, "
    f"cuda={torch.cuda.is_available()}"
)
PY
