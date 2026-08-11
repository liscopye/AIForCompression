#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${CAESAR_D_BESTVAL35K_CHECKPOINT:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam3em4_decoder100k_stage2_bestval35k.pt}"
INPUT_ROOT="${OBJECTIVE_INPUT_ROOT:-$ROOT/unified_results/objective_all_to_all_v1}"
OUTPUT_ROOT="${CAESAR_D_BESTVAL35K_OUTPUT:-$ROOT/unified_results/objective_era5_caesar_d_stage2_bestval35k_probe}"
LOG_ROOT="${CAESAR_D_BESTVAL35K_LOG_ROOT:-$ROOT/logs/objective_era5_caesar_d_stage2_bestval35k_probe}"
GPU="${CAESAR_D_BESTVAL35K_GPU:-7}"
POLL_SECONDS="${CAESAR_D_BESTVAL35K_POLL_SECONDS:-600}"
EBS=(0.3 0.03 0.015 0.003)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
test -s "$CHECKPOINT"

if [[ -f "$OUTPUT_ROOT/complete" ]]; then
  exit 0
fi

exec 9>"$ROOT/.caesar_gpu_benchmark.lock"
flock 9

while ! CUDA_VISIBLE_DEVICES="$GPU" python -c \
  'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' \
  >/dev/null 2>&1; do
  date -u '+cuda_unavailable_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG_ROOT/watcher.log"
  sleep "$POLL_SECONDS"
done
date -u '+cuda_available_utc=%Y-%m-%dT%H:%M:%SZ' >>"$LOG_ROOT/watcher.log"

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'role=objective-v1 Stage2 validation-best 35k four-point screening\n'
  printf 'variant=d_fixed_decoder_stage2_bestval35k\n'
  printf 'ebs=%s\n' "${EBS[*]}"
  sha256sum "$CHECKPOINT"
} >"$OUTPUT_ROOT/source_manifest.txt"

python -u "$ROOT/scripts/run_objective_benchmark.py" \
  --dataset era5_npy \
  --gpu "$GPU" \
  --output-root "$OUTPUT_ROOT" \
  --input-root "$INPUT_ROOT" \
  --models CAESAR-D \
  --caesar-checkpoint-root "$CHECKPOINT" \
  --caesar-variant d_fixed_decoder_stage2_bestval35k \
  --caesar-norm-type mean_range \
  --caesar-interpo-rate 3 \
  --caesar-batch-size 16 \
  --caesar-eb "${EBS[@]}" \
  --warmups 0 \
  --repeats 1 \
  --no-lpips \
  >"$LOG_ROOT/run.log" 2>&1

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_ROOT/source_manifest.txt"
touch "$OUTPUT_ROOT/complete"
