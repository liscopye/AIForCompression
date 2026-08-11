#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE_ROOT="${CAESAR_D_STAGE2_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_d_stage2_milestone_probes_decoder100k}"
PACKAGED_ROOT="${CAESAR_D_PROBE_PACKAGED_ROOT:-$ROOT/checkpoints/caesar_era5_d_stage2_objective_probes_decoder100k}"
BASELINE="${CAESAR_D_AUDIT_BASELINE:-$ROOT/unified_results/objective_all_to_all_v1/era5_npy/summary.json}"
ORIGINAL="${CAESAR_D_AUDIT_ORIGINAL:-$ROOT/unified_results/objective_era5_caesar_d_original_13pt_rd/era5_npy/summary.json}"
OUTPUT_ROOT="${CAESAR_D_FULL_AUDIT_ROOT:-$ROOT/unified_results}"
LOG_ROOT="${CAESAR_D_FULL_AUDIT_LOG_ROOT:-$ROOT/logs/caesar_era5_d_decoder100k_stage2_full_curve_audits}"
POLL_SECONDS="${CAESAR_D_FULL_AUDIT_POLL_SECONDS:-600}"
GPU="${CAESAR_D_13PT_GPU:-3}"
MILESTONES=(50000 75000 100000)

source /workspace/ai4cp/bin/activate
mkdir -p "$LOG_ROOT"
test -f "$BASELINE"
test -f "$ORIGINAL"
exec 9>"$ROOT/.caesar_gpu_benchmark.lock"

wait_for_cuda() {
  local log="$LOG_ROOT/cuda_watcher.log"
  while ! CUDA_VISIBLE_DEVICES="$GPU" python -c \
    'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1; do
    date -u '+cuda_unavailable_utc=%Y-%m-%dT%H:%M:%SZ' >>"$log"
    sleep "$POLL_SECONDS"
  done
  date -u '+cuda_available_utc=%Y-%m-%dT%H:%M:%SZ' >>"$log"
}

for milestone in "${MILESTONES[@]}"; do
  probe_complete="$PROBE_ROOT/update${milestone}/complete"
  checkpoint="$PACKAGED_ROOT/fixed_decoder_stage2_update${milestone}.pt"
  variant="d_decoder100k_stage2_update${milestone}"
  rd_output="$OUTPUT_ROOT/objective_era5_caesar_d_decoder100k_stage2_${milestone}_rd"
  compare_output="$OUTPUT_ROOT/objective_era5_caesar_d_decoder100k_stage2_${milestone}_compare"

  while [[ ! -f "$probe_complete" || ! -s "$checkpoint" ]]; do
    sleep "$POLL_SECONDS"
  done

  if [[ ! -f "$rd_output/complete" ]]; then
    flock 9
    wait_for_cuda
    CAESAR_D_13PT_CHECKPOINT="$checkpoint" \
    CAESAR_D_13PT_VARIANT="$variant" \
    CAESAR_D_13PT_INTERPO_RATE=3 \
    CAESAR_D_DECODER100K_13PT_DIR="$rd_output" \
    CAESAR_D_DECODER100K_13PT_LOG_DIR="$LOG_ROOT/update${milestone}" \
      bash "$ROOT/scripts/run_caesar_era5_d_decoder100k_stage2_5k_13pt.sh"
    flock -u 9
  fi

  python "$ROOT/scripts/build_caesar_era5_d_complete_5k_compare.py" \
    --baseline "$BASELINE" \
    --original "$ORIGINAL" \
    --candidate "$rd_output/era5_npy/summary.json" \
    --candidate-variant "$variant" \
    --candidate-label "CAESAR-D Stage2 ${milestone}" \
    --status "Stage2 ${milestone} full 13-point audit" \
    --output "$compare_output" \
    >"$LOG_ROOT/update${milestone}_compare.log" 2>&1
done

touch "$LOG_ROOT/complete"
