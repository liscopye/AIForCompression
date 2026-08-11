#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
OUTPUT_ROOT="${CAESAR_ERA5_100K_OBJECTIVE_DIR:-$ROOT/unified_results/objective_era5_caesar_v_finetuned_100k_rd}"
CHECKPOINT="${CAESAR_ERA5_100K_CHECKPOINT:-$ROOT/checkpoints/caesar_era5_daily_v_full_100k/rd_lr3em5_lam3em5_update100000.pt}"
LOG_ROOT="${CAESAR_ERA5_100K_OBJECTIVE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_finetuned_100k_rd}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT/shards" "$LOG_ROOT"
test -f "$CHECKPOINT"
test -f "$INPUT_ROOT/era5_npy/samples.json"
test -f "$INPUT_ROOT/era5_npy/normalization.json"

run_shard() {
  local gpu="$1"
  shift
  local shard="$OUTPUT_ROOT/shards/gpu$gpu"
  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$shard" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-V \
    --caesar-checkpoint-root "$CHECKPOINT" \
    --caesar-variant finetuned_100k \
    --caesar-norm-type mean_range \
    --caesar-eb "$@" \
    --warmups 0 \
    --repeats 1 \
    >"$LOG_ROOT/gpu$gpu.log" 2>&1
}

pids=()
run_shard 2 0.1 1e-9 & pids+=("$!")
run_shard 3 0.01 & pids+=("$!")
run_shard 4 0.003 & pids+=("$!")
run_shard 5 0.001 & pids+=("$!")
run_shard 6 0.0001 & pids+=("$!")
run_shard 7 3e-6 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != 0 ]]; then
  exit "$failed"
fi

python "$ROOT/scripts/build_era5_finetuned_all_models.py" \
  --baseline "$INPUT_ROOT/era5_npy/summary.json" \
  --shard-root "$OUTPUT_ROOT/shards" \
  --output "$OUTPUT_ROOT"
touch "$OUTPUT_ROOT/complete"
