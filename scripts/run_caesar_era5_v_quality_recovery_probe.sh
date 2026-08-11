#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
RECOVERY_ROOT="$ROOT/checkpoints/caesar_era5_v_lowrate_quality_recovery_10k"
LOWRATE_SOURCE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt"
OUTPUT_ROOT="${CAESAR_V_RECOVERY_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_v_quality_recovery_probe}"
LOG_ROOT="${CAESAR_V_RECOVERY_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_v_quality_recovery_probe}"
EBS=(0.3 0.2 0.1 0.05 0.03 0.01)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT/shards" "$LOG_ROOT"
test -f "$INPUT_ROOT/era5_npy/samples.json"
test -f "$INPUT_ROOT/era5_npy/normalization.json"

run_variant() {
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
names=()

launch() {
  local gpu="$1"
  local variant="$2"
  local checkpoint="$3"
  run_variant "$gpu" "$variant" "$checkpoint" &
  pids+=("$!")
  names+=("$variant")
}

launch 2 lowrate_base_100k "$LOWRATE_SOURCE"
launch 3 normalized_recovery_u2k "$RECOVERY_ROOT/normalized_lam1em4_lr3em6_update2000.pt"
launch 4 normalized_recovery_u5k "$RECOVERY_ROOT/normalized_lam1em4_lr3em6_update5000.pt"
launch 5 normalized_recovery_u10k "$RECOVERY_ROOT/normalized_lam1em4_lr3em6_update10000.pt"
launch 6 source_lam3em2_recovery_u2k "$RECOVERY_ROOT/source_lam3em2_lr3em6_update2000.pt"
launch 7 source_lam3em2_recovery_u10k "$RECOVERY_ROOT/source_lam3em2_lr3em6_update10000.pt"

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "finished ${names[$index]}"
  else
    echo "failed ${names[$index]}" >&2
    failed=1
  fi
done

if [[ "$failed" == 0 ]]; then
  touch "$OUTPUT_ROOT/complete"
fi
exit "$failed"
