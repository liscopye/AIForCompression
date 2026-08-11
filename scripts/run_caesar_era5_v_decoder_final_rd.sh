#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
CHECKPOINT="${CAESAR_V_DECODER_FINAL_CKPT:-$ROOT/checkpoints/caesar_era5_v_decoder_quality_100k/from_lowrate_lr3em4.pt}"
VARIANT="${CAESAR_V_DECODER_FINAL_VARIANT:-decoder_quality_100k_lr3em4}"
OUTPUT_ROOT="${CAESAR_V_DECODER_FINAL_RD_DIR:-$ROOT/unified_results/objective_era5_caesar_v_decoder_final_rd}"
LOG_ROOT="${CAESAR_V_DECODER_FINAL_RD_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_decoder_final_rd}"
GPU="${CAESAR_V_DECODER_FINAL_GPU:-2}"
EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 0.003 0.001 0.0001 0.000003 0.000000001)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
test -f "$CHECKPOINT"

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu "$GPU" \
  --output-root "$OUTPUT_ROOT" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-V \
  --caesar-checkpoint-root "$CHECKPOINT" \
  --caesar-variant "$VARIANT" \
  --caesar-norm-type mean_range \
  --caesar-eb "${EBS[@]}" \
  --warmups 0 \
  --repeats 1 \
  --no-lpips \
  >"$LOG_ROOT/$VARIANT.log" 2>&1

touch "$OUTPUT_ROOT/complete"
