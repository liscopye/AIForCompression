#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
STAGE2_ROOT="${CAESAR_D_STAGE2_ROOT:-$ROOT/checkpoints/caesar_era5_d_stage2_full_200k}"
FIXED_STAGE1="${CAESAR_D_PROBE_STAGE1:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_overlap5000.pt}"
PACKAGED_ROOT="${CAESAR_D_PROBE_PACKAGED_ROOT:-$ROOT/checkpoints/caesar_era5_d_stage2_objective_probes_decoder100k}"
OUTPUT_ROOT="${CAESAR_D_STAGE2_PROBE_DIR:-$ROOT/unified_results/objective_era5_caesar_d_stage2_milestone_probes_decoder100k}"
LOG_ROOT="${CAESAR_D_STAGE2_PROBE_LOG_DIR:-$ROOT/logs/objective_era5_caesar_d_stage2_milestone_probes_decoder100k}"
GPU="${CAESAR_D_STAGE2_PROBE_GPU:-3}"
POLL_SECONDS="${CAESAR_D_STAGE2_PROBE_POLL_SECONDS:-600}"
MILESTONES=(50000 75000 100000)
EBS=(0.3 0.03 0.015 0.003)

source /workspace/ai4cp/bin/activate
mkdir -p "$PACKAGED_ROOT" "$OUTPUT_ROOT" "$LOG_ROOT"
test -f "$FIXED_STAGE1"
exec 9>"$ROOT/.caesar_gpu_benchmark.lock"

wait_for_stable_checkpoint() {
  local checkpoint="$1"
  local size_before size_after
  while true; do
    if [[ -s "$checkpoint" ]]; then
      size_before="$(stat -c '%s' "$checkpoint")"
      sleep 5
      size_after="$(stat -c '%s' "$checkpoint")"
      if [[ "$size_before" == "$size_after" ]]; then
        return
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

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
  stage2="$STAGE2_ROOT/lam3em4_stage2_from_original_lr1em4_200k_update${milestone}.pt"
  packaged="$PACKAGED_ROOT/fixed_decoder_stage2_update${milestone}.pt"
  variant="d_decoder100k_stage2_update${milestone}"
  output="$OUTPUT_ROOT/update${milestone}"
  log="$LOG_ROOT/update${milestone}.log"

  if [[ -f "$output/complete" ]]; then
    continue
  fi
  wait_for_stable_checkpoint "$stage2"
  flock 9
  wait_for_cuda

  python "$ROOT/scripts/package_caesar_d_stage1.py" \
    --vae "$FIXED_STAGE1" \
    --base "$stage2" \
    --output "$packaged"

  {
    date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
    printf 'role=objective-v1 Stage2 milestone screening with a fixed decoder\n'
    printf 'milestone=%s\n' "$milestone"
    printf 'ebs=%s\n' "${EBS[*]}"
    sha256sum "$FIXED_STAGE1" "$stage2" "$packaged"
  } >"$output.source_manifest.txt"

  python -u "$ROOT/scripts/run_objective_benchmark.py" \
    --dataset era5_npy \
    --gpu "$GPU" \
    --output-root "$output" \
    --input-root "$INPUT_ROOT" \
    --models CAESAR-D \
    --caesar-checkpoint-root "$packaged" \
    --caesar-variant "$variant" \
    --caesar-norm-type mean_range \
    --caesar-eb "${EBS[@]}" \
    --warmups 0 \
    --repeats 1 \
    --no-lpips \
    >"$log" 2>&1

  touch "$output/complete"
  date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$output.source_manifest.txt"
  flock -u 9
done

touch "$OUTPUT_ROOT/complete"
