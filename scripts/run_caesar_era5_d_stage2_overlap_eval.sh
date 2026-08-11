#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${ERA5_DAILY_PROBE:-/workspace/Data/ERA5/daily00_center512_validation_probe.npy}"
CKPT_DIR="${CAESAR_D_STAGE2_OVERLAP_DIR:-$ROOT/checkpoints/caesar_era5_d_stage2_overlap_5k}"
OUTPUT_DIR="${CAESAR_D_STAGE2_OVERLAP_EVAL_DIR:-$ROOT/unified_results/caesar_era5_d_stage2_overlap_eval}"
LOG_DIR="${CAESAR_D_STAGE2_OVERLAP_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_d_stage2_overlap_eval}"
STEPS=(50 100 250 500 1000 2000 5000)
EBS=(0.3 0.1 0.03 0.01 0.003)

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$PROBE"

run_eval() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  local mode="$4"
  local output="$OUTPUT_DIR/${name}_${mode}"
  local log="$LOG_DIR/${name}_${mode}.log"
  local mode_args=()
  local ebs=("${EBS[@]}")
  if [[ "$mode" == "model_only" ]]; then
    mode_args+=(--caesar_no_pca)
    ebs=(0.3)
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$PROBE" \
    --output_dir "$output" \
    --models caesar_d \
    --max_samples 16 \
    --max_channels 30 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_norm_type mean_range \
    --caesar_eb "${ebs[@]}" \
    --caesar_num_windows 1 \
    --batch_size 64 \
    --no_lpips \
    "${mode_args[@]}" \
    >"$log" 2>&1
}

run_checkpoint() {
  local gpu="$1"
  local name="$2"
  local checkpoint="$3"
  test -f "$checkpoint"
  run_eval "$gpu" "$name" "$checkpoint" model_only
  run_eval "$gpu" "$name" "$checkpoint" pca
}

queue() {
  local gpu="$1"
  local tag="$2"
  if [[ "$tag" == "lam1em4" ]]; then
    run_checkpoint "$gpu" original_d "$ROOT/checkpoints/caesar/caesar_d.pt"
  fi
  for step in "${STEPS[@]}"; do
    local name="${tag}_stage2_lr1em4_update${step}"
    run_checkpoint "$gpu" "$name" "$CKPT_DIR/$name.pt"
  done
}

pids=()
queue 4 lam1em4 & pids+=("$!")
queue 7 lam3em4 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
