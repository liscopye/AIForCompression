#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${CAESAR_DIRECT_EVAL_DATA:-/workspace/Data/ERA5/finetune_processed/era5_test.npy}"
SOURCE_DIR="${CAESAR_SOURCE_CKPT_DIR:-$ROOT/checkpoints/caesar_era5_hourly_source_sweep}"
PACKAGED_DIR="$SOURCE_DIR/packaged_d"
OUTPUT_DIR="${CAESAR_SOURCE_DIRECT_EVAL_DIR:-$ROOT/unified_results/caesar_era5_source_direct_reconstruction}"
LOG_DIR="${CAESAR_SOURCE_DIRECT_EVAL_LOG_DIR:-$ROOT/logs/caesar_era5_source_direct_reconstruction}"

mkdir -p "$PACKAGED_DIR" "$OUTPUT_DIR" "$LOG_DIR"
test -f "$DATA"

for tag in lr1e4_lam1e3 lr1e4_lam3e3 lr1e4_lam1e2; do
  python "$ROOT/scripts/package_caesar_d_stage1.py" \
    --vae "$SOURCE_DIR/ds_${tag}.pt" \
    --base "$ROOT/checkpoints/caesar/caesar_d.pt" \
    --output "$PACKAGED_DIR/ds_${tag}.pt" \
    >/dev/null
done

run_eval() {
  local gpu="$1"
  local model="$2"
  local name="$3"
  local checkpoint="$4"
  local windows=1
  if [[ "$model" == "caesar_v" ]]; then
    windows=2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/run_dataset_compression.py" \
    --dataset era5_npy \
    --data_root "$DATA" \
    --output_dir "$OUTPUT_DIR/$name" \
    --models "$model" \
    --max_samples 16 \
    --max_channels 268 \
    --resolution 240 240 \
    --caesar_ckpt_dir "$checkpoint" \
    --caesar_norm_type mean_range \
    --caesar_eb 0.001 \
    --caesar_num_windows "$windows" \
    --batch_size 64 \
    --caesar_no_pca \
    --no_lpips \
    >"$LOG_DIR/$name.log" 2>&1
}

run_eval 2 caesar_v vs_lr1e4_lam1e3 "$SOURCE_DIR/vs_lr1e4_lam1e3.pt" &
pids=("$!")
run_eval 3 caesar_v vs_lr1e4_lam3e3 "$SOURCE_DIR/vs_lr1e4_lam3e3.pt" &
pids+=("$!")
run_eval 4 caesar_v vs_lr1e4_lam1e2 "$SOURCE_DIR/vs_lr1e4_lam1e2.pt" &
pids+=("$!")
run_eval 5 caesar_d ds_lr1e4_lam1e3 "$PACKAGED_DIR/ds_lr1e4_lam1e3.pt" &
pids+=("$!")
run_eval 6 caesar_d ds_lr1e4_lam3e3 "$PACKAGED_DIR/ds_lr1e4_lam3e3.pt" &
pids+=("$!")
run_eval 7 caesar_d ds_lr1e4_lam1e2 "$PACKAGED_DIR/ds_lr1e4_lam1e2.pt" &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
