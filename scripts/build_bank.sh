#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RESETTLE_CHECKPOINT:?Set RESETTLE_CHECKPOINT to the guarded K=1 latest.pt}"
: "${LIBERO_DATA_ROOT:?Set LIBERO_DATA_ROOT to the four-suite LeRobot parent directory}"
: "${RESETTLE_BANK:?Set RESETTLE_BANK to the goal-bank output directory}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BANK_BATCH_SIZE:-16}"
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BANK_BATCH_SIZE must be positive" >&2; exit 1; }
[[ -f "$RESETTLE_CHECKPOINT" ]] || { echo "Missing checkpoint: $RESETTLE_CHECKPOINT" >&2; exit 1; }
params="$(dirname "$RESETTLE_CHECKPOINT")/params-pretrain.yaml"
[[ -f "$params" ]] || { echo "Missing training config: $params" >&2; exit 1; }
"$PYTHON" - "$params" <<'PY'
import sys
import yaml

with open(sys.argv[1]) as handle:
    intent = yaml.safe_load(handle)["intent_action"]
expected = {
    "action_parameterization": "state_servo_residual",
    "state_servo_prior": "state_goal_delta_guarded_single_tick_v7",
    "condition_on_state_servo_prior": True,
    "residual_execution_mode": "guarded_single_tick",
    "residual_representation": "executed_correction",
    "chunk_size": 1,
    "goal_frame_index": 4,
    "guarded_native_steps": 5,
}
for key, value in expected.items():
    if intent.get(key) != value:
        raise SystemExit(f"Incompatible guarded-v7 training config: {key}")
PY

suites=(goal spatial object 10)
for suite in "${suites[@]}"; do
    [[ -d "$LIBERO_DATA_ROOT/libero_${suite}_no_noops_1.0.0_lerobot" ]] || {
        echo "Missing LeRobot suite: libero_${suite}" >&2; exit 1;
    }
done
gpu_count="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
(( gpu_count >= 4 )) || { echo "Four visible GPUs are required; found $gpu_count" >&2; exit 1; }
mkdir -p "$RESETTLE_BANK"
cd "$ROOT"

pids=()
for rank in 0 1 2 3; do
    suite="${suites[rank]}"
    output="$RESETTLE_BANK/libero_${suite}"
    (
        CUDA_VISIBLE_DEVICES="$rank" "$PYTHON" -u -m app.build_bank \
            --data-root "$LIBERO_DATA_ROOT" --suite "$suite" \
            --checkpoint "$RESETTLE_CHECKPOINT" --output-dir "$output" \
            --device cuda:0 --batch-size "$BATCH_SIZE"
    ) >"$RESETTLE_BANK/worker-${rank}-libero_${suite}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
(( status == 0 )) || { echo "Bank encoding failed; inspect $RESETTLE_BANK/worker-*.log" >&2; exit 1; }
"$PYTHON" -m app.build_bank --output-dir "$RESETTLE_BANK" --checkpoint "$RESETTLE_CHECKPOINT" --merge