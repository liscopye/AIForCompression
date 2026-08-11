#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
CKPT_ROOT="$ROOT/checkpoints/caesar_era5_v_decoder_quality_extreme_5k"
OUTPUT_ROOT="${CAESAR_V_DECODER_EXTREME_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_v_decoder_extreme_probe}"
LOG_ROOT="${CAESAR_V_DECODER_EXTREME_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_decoder_extreme_probe}"
EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT/shards" "$LOG_ROOT"

run_one() {
  local gpu="$1"
  local variant="$2"
  local checkpoint="$3"
  local shard="$OUTPUT_ROOT/shards/$variant"
  test -f "$checkpoint"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$shard" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-V \
    --caesar-checkpoint-root "$checkpoint" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-eb "${EBS[@]}" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$LOG_ROOT/$variant.log" 2>&1
  touch "$shard/complete"
}

pids=()
run_one 2 sr_lr3em3_best "$CKPT_ROOT/sr_lr3em3.pt" & pids+=("$!")
run_one 3 sr_lr1em3_best "$CKPT_ROOT/sr_lr1em3.pt" & pids+=("$!")
run_one 4 decoder_lr1em3_best "$CKPT_ROOT/decoder_lr1em3.pt" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" == 0 ]]; then
  touch "$OUTPUT_ROOT/complete"
fi
exit "$failed"
