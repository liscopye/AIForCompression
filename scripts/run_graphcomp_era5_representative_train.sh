#!/usr/bin/env bash
set -euo pipefail

cd /workspace/AIForCompression

DATA=/workspace/Data/ERA5/finetune_processed_time_split_t45_v16/era5_train.npy
OUT_ROOT=/workspace/AIForCompression/unified_results/graphcomp_era5_representative_fullres_s100_m20_e300
LOG_ROOT=/workspace/AIForCompression/logs/graphcomp/era5_representative_fullres_s100_m20_e300
mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_one() {
  local gpu="$1"
  local channel="$2"
  local name="$3"
  local out="$OUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  echo "[$(date -Is)] start $name channel=$channel gpu=$gpu" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/graphcomp_redsea_readme_repro.py \
    --input_kind era5 \
    --era5_npy "$DATA" \
    --era5_channel "$channel" \
    --frames 45 \
    --y_stride 1 \
    --x_stride 1 \
    --scale 100 \
    --sigma 1 \
    --min_size 20 \
    --epochs 300 \
    --cnn_epochs 300 \
    --lr 1e-3 \
    --cnn_lr 1e-3 \
    --cnn_batch_size 16 \
    --device cuda:0 \
    --skip_residual \
    --skip_reconstruction_save \
    --wandb \
    --wandb_project graphcomp-era5 \
    --wandb_name "graphcomp_era5_${name}_fullres_s100_m20_e300" \
    --output_dir "$out" >> "$log" 2>&1
  echo "[$(date -Is)] done $name" | tee -a "$log"
}

# Representative ERA5 variables:
# z: geopotential at 1000/850/500 hPa
# t: temperature at 1000/850/500 hPa
# t2m: 2m temperature single-level field
run_one 2 0   z_1000 &
run_one 3 6   z_850 &
run_one 4 15  z_500 &
run_one 5 148 t_1000 &
run_one 6 154 t_850 &
run_one 7 163 t_500 &
wait

run_one 2 263 t2m

echo "[$(date -Is)] all representative GraphComp ERA5 training jobs completed"
