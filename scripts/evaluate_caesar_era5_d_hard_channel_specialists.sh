#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${CAESAR_D_SPECIALIST_OUTPUT_DIR:-$ROOT/checkpoints/caesar_era5_d_hard_channel_specialists}"
OUTPUT_DIR="${CAESAR_D_SPECIALIST_EVAL_DIR:-$ROOT/unified_results/diagnostic_caesar_d_hard_channel_specialists}"
LOG_DIR="${CAESAR_D_SPECIALIST_EVAL_LOG_DIR:-$ROOT/logs/diagnostic_caesar_d_hard_channel_specialists}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

run_one() {
  local gpu="$1"
  local checkpoint_name="$2"
  local variable_start="$3"
  local variable_count="$4"
  local output_name="$5"
  local batch_size=16
  if (( variable_count < batch_size )); then
    batch_size="$variable_count"
  fi
  local checkpoint="$CHECKPOINT_DIR/$checkpoint_name.pt"
  local output="$OUTPUT_DIR/$output_name.json"
  test -f "$checkpoint"
  if [[ -s "$output" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    "$ROOT/scripts/diagnose_caesar_d_temporal_reconstruction.py" \
    --checkpoint "$checkpoint" \
    --output "$output" \
    --mode official \
    --variable-start "$variable_start" \
    --max-variables "$variable_count" \
    --batch-size "$batch_size" \
    --device cuda:0 \
    >"$LOG_DIR/$output_name.log" 2>&1
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_one 3 specific_humidity_lr1em4 37 37 candidate_specific_humidity_lr1em4 & pids+=("$!")
run_one 4 relative_humidity_lr1em4 185 37 candidate_relative_humidity_lr1em4 & pids+=("$!")
run_one 5 single_level_lr1em4 259 9 candidate_single_level_lr1em4 & pids+=("$!")
run_one 6 specific_humidity_lr3em5 37 37 candidate_specific_humidity_lr3em5 & pids+=("$!")
run_one 7 relative_humidity_lr3em5 185 37 candidate_relative_humidity_lr3em5 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
trap - INT TERM EXIT
if (( failed != 0 )); then
  exit "$failed"
fi
touch "$OUTPUT_DIR/specialist_evaluation.complete"
