#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${CAESAR_D_CPU_AUDIT_OUTPUT:-$ROOT/unified_results/diagnostic_caesar_d_stage2_cpu_full268}"
LOG_ROOT="${CAESAR_D_CPU_AUDIT_LOG_ROOT:-$ROOT/logs/diagnostic_caesar_d_stage2_cpu_full268}"
THREADS="${CAESAR_D_CPU_AUDIT_THREADS_PER_JOB:-24}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

names=(stage2_5k bestval35k update75000 update100000)
checkpoints=(
  "$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_overlap5000.pt"
  "$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_bestval35k.pt"
  "$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_update75000.pt"
  "$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_update100000.pt"
)

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'role=full268_cpu_sampling_screen_not_formal_eb_rd\n'
  printf 'threads_per_job=%s\n' "$THREADS"
  sha256sum "${checkpoints[@]}"
} >"$OUTPUT_ROOT/source_manifest.txt"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

for index in "${!names[@]}"; do
  name="${names[$index]}"
  checkpoint="${checkpoints[$index]}"
  test -s "$checkpoint"
  if [[ -s "$OUTPUT_ROOT/$name.json" ]]; then
    continue
  fi
  OMP_NUM_THREADS="$THREADS" \
  MKL_NUM_THREADS="$THREADS" \
  OPENBLAS_NUM_THREADS="$THREADS" \
    python "$ROOT/scripts/diagnose_caesar_d_temporal_reconstruction.py" \
      --checkpoint "$checkpoint" \
      --output "$OUTPUT_ROOT/$name.json" \
      --mode diffusion_ensemble \
      --ensemble-size 1 \
      --max-variables 268 \
      --batch-size 16 \
      --device cpu \
      >"$LOG_ROOT/$name.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
trap - INT TERM EXIT

if (( failed != 0 )); then
  exit "$failed"
fi

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_ROOT/source_manifest.txt"
touch "$OUTPUT_ROOT/complete"
