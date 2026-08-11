#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="${CAESAR_HOURLY_EVAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_pilot_eval}"
CKPT_DIR="${CAESAR_HOURLY_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_pilot}"
SELECTION_DIR="${CAESAR_HOURLY_SELECTION_DIR:-$ROOT/unified_results/caesar_era5_hourly_selection}"
SELECTED_CKPT_DIR="${CAESAR_HOURLY_SELECTED_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_selected}"
OBJECTIVE_V_DIR="${CAESAR_HOURLY_OBJECTIVE_V_DIR:-$ROOT/unified_results/objective_v1_era5_hourly_tuned_v}"
OBJECTIVE_D_DIR="${CAESAR_HOURLY_OBJECTIVE_D_DIR:-$ROOT/unified_results/objective_v1_era5_hourly_tuned_d}"
OBJECTIVE_ORIGINAL_DIR="${CAESAR_HOURLY_OBJECTIVE_ORIGINAL_DIR:-$ROOT/unified_results/objective_v1_era5_hourly_original}"
FINAL_DIR="${CAESAR_HOURLY_FINAL_DIR:-$ROOT/unified_results/caesar_era5_hourly_final}"
LOG_DIR="${CAESAR_HOURLY_OBJECTIVE_LOG_DIR:-$ROOT/logs/caesar_era5_hourly_objective}"
INPUT_ROOT="$ROOT/unified_results/objective_v1"

mkdir -p "$LOG_DIR"

python "$ROOT/scripts/select_caesar_era5_hourly_checkpoint.py" \
  --eval-dir "$EVAL_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  --original-dir "$ROOT/checkpoints/caesar" \
  --output-dir "$SELECTION_DIR" \
  --selected-checkpoint-dir "$SELECTED_CKPT_DIR" \
  | tee "$LOG_DIR/selection.log"

readarray -t selected_names < <(python - "$SELECTION_DIR/selection.json" <<'PY'
import json
import sys
selection = json.load(open(sys.argv[1]))
print(selection["models"]["V"]["selected"]["name"])
print(selection["models"]["D"]["selected"]["name"])
print(selection["models"]["V"]["selected"]["norm_type"])
print(selection["models"]["D"]["selected"]["norm_type"])
PY
)
V_NORM_TYPE="${selected_names[2]}"
D_NORM_TYPE="${selected_names[3]}"

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu 0 \
  --output-root "$OBJECTIVE_V_DIR" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-V \
  --caesar-checkpoint-root "$SELECTED_CKPT_DIR" \
  --caesar-variant hourly_tuned \
  --caesar-norm-type "$V_NORM_TYPE" \
  --warmups 2 \
  --repeats 5 \
  --force \
  >"$LOG_DIR/objective_v.log" 2>&1 &
v_pid=$!

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu 1 \
  --output-root "$OBJECTIVE_D_DIR" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-D \
  --caesar-checkpoint-root "$SELECTED_CKPT_DIR" \
  --caesar-variant hourly_tuned \
  --caesar-norm-type "$D_NORM_TYPE" \
  --warmups 2 \
  --repeats 5 \
  --force \
  >"$LOG_DIR/objective_d.log" 2>&1 &
d_pid=$!

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu 2 \
  --output-root "$OBJECTIVE_ORIGINAL_DIR" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-V CAESAR-D \
  --caesar-checkpoint-root "$ROOT/checkpoints/caesar" \
  --caesar-variant original \
  --caesar-norm-type mean_range \
  --warmups 2 \
  --repeats 5 \
  --force \
  >"$LOG_DIR/objective_original.log" 2>&1 &
original_pid=$!

failed=0
wait "$v_pid" || failed=1
wait "$d_pid" || failed=1
wait "$original_pid" || failed=1
if [[ "$failed" -ne 0 ]]; then
  echo "Objective evaluation failed; inspect $LOG_DIR" >&2
  exit 1
fi

python "$ROOT/scripts/summarize_caesar_era5_stable_tuning.py" \
  --baseline "$OBJECTIVE_ORIGINAL_DIR/era5_npy/summary.json" \
  --tuned-v "$OBJECTIVE_V_DIR/era5_npy/summary.json" \
  --tuned-d "$OBJECTIVE_D_DIR/era5_npy/summary.json" \
  --selection-json "$SELECTION_DIR/selection.json" \
  --require-improvement \
  --output-dir "$FINAL_DIR" \
  | tee "$LOG_DIR/final_summary.log"
