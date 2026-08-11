#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="$ROOT/unified_results/objective_all_to_all_v1"
CKPT_ROOT="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k"
PACKAGED_D="$CKPT_ROOT/packaged_d"
OUTPUT_ROOT="${CAESAR_VD_LOWRATE_100K_RD_DIR:-$ROOT/unified_results/objective_era5_caesar_vd_lowrate_100k_rd}"
LOG_ROOT="${CAESAR_VD_LOWRATE_100K_RD_LOG_DIR:-$ROOT/logs/objective_era5_caesar_vd_lowrate_100k_rd}"
EBS=(0.1 0.01 0.003 0.001 0.0001 3e-6 1e-9)

source /workspace/ai4cp/bin/activate
mkdir -p "$PACKAGED_D" "$OUTPUT_ROOT/shards" "$LOG_ROOT"
test -f "$INPUT_ROOT/era5_npy/samples.json"
test -f "$INPUT_ROOT/era5_npy/normalization.json"

package_d() {
  local tag="$1"
  local vae="$CKPT_ROOT/d_s1_lr1em5_lam${tag}_full100k_update100000.pt"
  local output="$PACKAGED_D/caesar_d_lam${tag}_stage1_100k_with_original_diffusion.pt"
  test -f "$vae"
  python "$ROOT/scripts/package_caesar_d_stage1.py" \
    --vae "$vae" \
    --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
    --output "$output"
}

package_d 1em4
package_d 3em4
package_d 1em3

run_variant() {
  local gpu="$1"
  local model="$2"
  local variant="$3"
  local checkpoint="$4"
  local shard="$OUTPUT_ROOT/shards/$variant"
  test -f "$checkpoint"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$shard" \
    --input-root "$INPUT_ROOT" \
    --models "$model" \
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
  local model="$2"
  local variant="$3"
  local checkpoint="$4"
  run_variant "$gpu" "$model" "$variant" "$checkpoint" &
  pids+=("$!")
  names+=("$variant")
}

launch 2 CAESAR-V v_lowrate_lam1em4_100k \
  "$CKPT_ROOT/v_lr1em5_lam1em4_full100k_update100000.pt"
launch 3 CAESAR-V v_lowrate_lam3em4_100k \
  "$CKPT_ROOT/v_lr1em5_lam3em4_full100k_update100000.pt"
launch 4 CAESAR-V v_lowrate_lam1em3_100k \
  "$CKPT_ROOT/v_lr1em5_lam1em3_full100k_update100000.pt"
launch 5 CAESAR-D d_stage1_lam1em4_100k_diagnostic \
  "$PACKAGED_D/caesar_d_lam1em4_stage1_100k_with_original_diffusion.pt"
launch 6 CAESAR-D d_stage1_lam3em4_100k_diagnostic \
  "$PACKAGED_D/caesar_d_lam3em4_stage1_100k_with_original_diffusion.pt"
launch 7 CAESAR-D d_stage1_lam1em3_100k_diagnostic \
  "$PACKAGED_D/caesar_d_lam1em3_stage1_100k_with_original_diffusion.pt"

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
