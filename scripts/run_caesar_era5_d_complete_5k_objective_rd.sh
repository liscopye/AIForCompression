#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
DECODER="$ROOT/checkpoints/caesar_era5_d_decoder_quality_100k/lam3em4_from_lowrate_lr3em4.pt"
STAGE2="$ROOT/checkpoints/caesar_era5_d_stage2_overlap_5k/lam3em4_stage2_lr1em4_update5000.pt"
ORIGINAL="$ROOT/checkpoints/caesar/caesar_d.pt"
PACKAGED_DIR="$ROOT/checkpoints/caesar_era5_d_complete_candidates"
CANDIDATE="$PACKAGED_DIR/lam3em4_decoder_current_stage2_5k.pt"
CANDIDATE_OUT="${CAESAR_D_5K_OBJECTIVE_DIR:-$ROOT/unified_results/objective_era5_caesar_d_complete_5k_rd}"
ORIGINAL_OUT="${CAESAR_D_ORIGINAL_13PT_DIR:-$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd}"
LOG_DIR="${CAESAR_D_5K_OBJECTIVE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_complete_5k_rd}"
EBS=(0.3 0.1 0.05 0.03 0.025 0.02 0.015 0.01 0.003 0.001 0.0001 0.000003 0.000000001)

source /workspace/ai4cp/bin/activate
mkdir -p "$PACKAGED_DIR" "$CANDIDATE_OUT" "$ORIGINAL_OUT" "$LOG_DIR"
test -f "$DECODER"
test -f "$STAGE2"
test -f "$ORIGINAL"

python "$ROOT/scripts/package_caesar_d_stage1.py" \
  --vae "$DECODER" \
  --base "$STAGE2" \
  --output "$CANDIDATE"

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'role=independent objective-v1 probe while 100k/200k training continues\n'
  printf 'candidate=%s\n' "$CANDIDATE"
  sha256sum "$DECODER" "$STAGE2" "$CANDIDATE" "$ORIGINAL"
} >"$CANDIDATE_OUT/source_manifest.txt"

run_curve() {
  local gpu="$1"
  local checkpoint="$2"
  local variant="$3"
  local output="$4"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$gpu" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$checkpoint" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-eb "${EBS[@]}" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$LOG_DIR/$variant.log" 2>&1
}

run_curve 3 "$CANDIDATE" d_lam3em4_decoder_current_stage2_5k "$CANDIDATE_OUT" &
p0=$!
run_curve 4 "$ORIGINAL" original "$ORIGINAL_OUT" &
p1=$!

failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1

if [[ "$failed" == 0 ]]; then
  touch "$CANDIDATE_OUT/complete" "$ORIGINAL_OUT/complete"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$CANDIDATE_OUT/source_manifest.txt"
fi
exit "$failed"
