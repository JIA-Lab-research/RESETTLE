#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the directory containing the four LeRobot suites}"
: "${VJEPA_PRETRAIN:?Set VJEPA_PRETRAIN to the V-JEPA2-AC checkpoint}"
: "${RESETTLE_OUTPUT:?Set RESETTLE_OUTPUT to the checkpoint output directory}"
PYTHON="${PYTHON:-python}"
NUM_GPUS="${NUM_GPUS:-8}"
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be positive" >&2; exit 1; }
CONFIG="$(mktemp --suffix=.yaml)"
trap 'rm -f "$CONFIG"' EXIT

"$PYTHON" - "$ROOT/configs/libero.yaml" "$CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle)
config["folder"] = os.environ["RESETTLE_OUTPUT"]
config["meta"]["pretrain_checkpoint"] = os.environ["VJEPA_PRETRAIN"]
root = Path(os.environ["LIBERO_DATA_ROOT"])
config["data"]["datasets"] = [
    str(root / f"libero_{suite}_no_noops_1.0.0_lerobot")
    for suite in ("object", "goal", "spatial", "10")
]
with open(sys.argv[2], "w") as handle:
    yaml.safe_dump(config, handle)
PY

devices=()
for ((index = 0; index < NUM_GPUS; index++)); do
    devices+=("cuda:${index}")
done
cd "$ROOT"
"$PYTHON" -m app.main --fname "$CONFIG" --devices "${devices[@]}"