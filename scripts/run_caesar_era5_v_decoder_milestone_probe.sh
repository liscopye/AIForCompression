#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
CKPT_ROOT="$ROOT/checkpoints/caesar_era5_v_decoder_quality_100k"
MILESTONE="${CAESAR_V_DECODER_PROBE_MILESTONE:-25000}"
OUTPUT_ROOT="${CAESAR_V_DECODER_MILESTONE_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_v_decoder_${MILESTONE}_probe}"
LOG_ROOT="${CAESAR_V_DECODER_MILESTONE_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_decoder_${MILESTONE}_probe}"
EBS=(0.3 0.025 0.02 0.01)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT/shards" "$LOG_ROOT"

run_one() {
  local gpu="$1"
  local stem="$2"
  local checkpoint="$CKPT_ROOT/${stem}_update${MILESTONE}.pt"
  local variant="${stem}_u${MILESTONE}"
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
run_one 2 from_lowrate_lr3em4 & pids+=("$!")
run_one 3 from_decoder10k_lr1em4 & pids+=("$!")
run_one 4 from_decoder10k_lr3em5 & pids+=("$!")

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
