#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
CURRENT_CHECKPOINT="${CAESAR_D_ENSEMBLE_CURRENT:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_overlap5000.pt}"
LOWRATE_CHECKPOINT="${CAESAR_D_ENSEMBLE_LOWRATE:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam1em3_decoder_best_original_stage2.pt}"
CURRENT_OUTPUT="${CAESAR_D_ENSEMBLE_CURRENT_OUTPUT:-$ROOT/unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_ensemble4_rd}"
LOWRATE_OUTPUT="${CAESAR_D_ENSEMBLE_LOWRATE_OUTPUT:-$ROOT/unified_results/objective_era5_caesar_d_lam1em3_original_stage2_ensemble4_rd}"
LOG_DIR="${CAESAR_D_ENSEMBLE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_ensemble4_full_curves}"
POLL_SECONDS="${CAESAR_D_ENSEMBLE_POLL_SECONDS:-600}"
EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 0.003 0.001 0.0001 0.000003 1e-9)

source /workspace/ai4cp/bin/activate
mkdir -p "$LOG_DIR" "$CURRENT_OUTPUT" "$LOWRATE_OUTPUT"
test -f "$CURRENT_CHECKPOINT"
test -f "$LOWRATE_CHECKPOINT"

exec 9>"$ROOT/.caesar_gpu_benchmark.lock"
flock 9

while ! CUDA_VISIBLE_DEVICES=3 python -c \
  'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
  >/dev/null 2>&1; do
  date -u '+cuda_unavailable_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG_DIR/watcher.log"
  sleep "$POLL_SECONDS"
done
date -u '+cuda_available_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG_DIR/watcher.log"

run_curve() {
  local gpu="$1"
  local checkpoint="$2"
  local variant="$3"
  local output="$4"
  local log="$5"
  if [[ -f "$output/complete" ]]; then
    return
  fi
  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$checkpoint" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-interpo-rate 3 \
    --caesar-diffusion-ensemble-size 4 \
    --caesar-batch-size 16 \
    --caesar-eb "${EBS[@]}" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    --force \
    >"$log" 2>&1
}

{
  printf 'role=complete_caesar_d_diffusion_ensemble4_rd_audit\n'
  printf 'ebs=%s\n' "${EBS[*]}"
  printf 'ensemble_size=4\ninterpo_rate=3\nbatch_size=16\n'
  sha256sum "$CURRENT_CHECKPOINT" "$LOWRATE_CHECKPOINT"
} >"$LOG_DIR/source_manifest.txt"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_curve \
  3 "$CURRENT_CHECKPOINT" d_decoder100k_stage2_overlap5k_ensemble4 \
  "$CURRENT_OUTPUT" "$LOG_DIR/current.log" & pids+=("$!")
run_curve \
  4 "$LOWRATE_CHECKPOINT" d_lam1em3_original_stage2_ensemble4 \
  "$LOWRATE_OUTPUT" "$LOG_DIR/lowrate.log" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
trap - INT TERM EXIT
if (( failed != 0 )); then
  exit "$failed"
fi

python "$ROOT/scripts/build_caesar_era5_d_complete_5k_compare.py" \
  --baseline "$ROOT/unified_results/objective_all_to_all_v1/era5_npy/summary.json" \
  --original "$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd/era5_npy/summary.json" \
  --candidate "$CURRENT_OUTPUT/era5_npy/summary.json" \
  --candidate-variant d_decoder100k_stage2_overlap5k_ensemble4 \
  --candidate-label "CAESAR-D 5k ensemble-4" \
  --status "Complete 13-point diffusion ensemble-4 audit" \
  --output "$ROOT/unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_ensemble4_compare"

python "$ROOT/scripts/build_caesar_era5_d_complete_5k_compare.py" \
  --baseline "$ROOT/unified_results/objective_all_to_all_v1/era5_npy/summary.json" \
  --original "$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd/era5_npy/summary.json" \
  --candidate "$LOWRATE_OUTPUT/era5_npy/summary.json" \
  --candidate-variant d_lam1em3_original_stage2_ensemble4 \
  --candidate-label "CAESAR-D low-rate ensemble-4" \
  --status "Complete 13-point low-rate diffusion ensemble-4 audit" \
  --output "$ROOT/unified_results/objective_era5_caesar_d_lam1em3_original_stage2_ensemble4_compare"

python "$ROOT/scripts/build_caesar_era5_vd_complete_compare.py" \
  --baseline "$ROOT/unified_results/objective_all_to_all_v1/combined_summary.json" \
  --v-final "$ROOT/unified_results/objective_era5_caesar_v_decoder_final_rd/era5_npy/summary.json" \
  --v-variant decoder_quality_100k_lr3em4 \
  --d-original "$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd/era5_npy/summary.json" \
  --d-final "$ROOT/unified_results/objective_era5_caesar_d_decoder100k_stage2_overlap5k_rd/era5_npy/summary.json" \
  --d-variant d_lam3em4_decoder100k_stage2_overlap5000 \
  --d-keyframe "$ROOT/unified_results/objective_era5_caesar_d_lam1em3_keyframe_only_rd/era5_npy/summary.json" \
  --d-keyframe-variant d_lam1em3_keyframe_only_ablation \
  --d-ensemble "$CURRENT_OUTPUT/era5_npy/summary.json" \
  --d-ensemble-variant d_decoder100k_stage2_overlap5k_ensemble4 \
  --d-lowrate-ensemble "$LOWRATE_OUTPUT/era5_npy/summary.json" \
  --d-lowrate-ensemble-variant d_lam1em3_original_stage2_ensemble4 \
  --output "$ROOT/unified_results/objective_era5_caesar_vd_complete_compare"

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG_DIR/source_manifest.txt"
touch "$LOG_DIR/complete"
