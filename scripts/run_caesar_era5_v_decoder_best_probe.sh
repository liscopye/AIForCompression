#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
CHECKPOINT="$ROOT/checkpoints/caesar_era5_v_decoder_quality_highlr_10k/decoder_lr3em4_update10000.pt"
OUTPUT_ROOT="${CAESAR_V_DECODER_BEST_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_v_decoder_best_probe}"
LOG_ROOT="${CAESAR_V_DECODER_BEST_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_decoder_best_probe}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
test -f "$CHECKPOINT"

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu 2 \
  --output-root "$OUTPUT_ROOT" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-V \
  --caesar-checkpoint-root "$CHECKPOINT" \
  --caesar-variant decoder_lr3em4_u10k \
  --caesar-norm-type mean_range \
  --caesar-eb 0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 \
  --warmups 0 \
  --repeats 1 \
  --no-lpips \
  >"$LOG_ROOT/probe.log" 2>&1

touch "$OUTPUT_ROOT/complete"
