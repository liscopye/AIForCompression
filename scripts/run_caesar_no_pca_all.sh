#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
PYTHON=${PYTHON:-python}
RUNNER="$ROOT/scripts/run_dataset_compression.py"

cd "$ROOT"

run_one() {
  local gpu="$1"
  shift
  echo "=== GPU ${gpu}: $* ==="
  "$PYTHON" -u "$RUNNER" "$@" \
    --models caesar_v caesar_d \
    --caesar_no_pca \
    --caesar_eb 0 \
    --no_lpips \
    --gpu "$gpu"
}

run_one 1 \
  --dataset e3sm_npz \
  --data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz \
  --output_dir unified_results/e3sm_npz_caesar_no_pca \
  --max_samples 64 \
  --turb_rot_image_group_mode variables \
  --npz_image_channels 3 &

run_one 2 \
  --dataset turb_rot_npz \
  --data_root /workspace/Turb_Rot_testset.npz \
  --output_dir unified_results/turb_rot_npz_caesar_no_pca \
  --max_samples 64 \
  --turb_rot_image_group_mode sections \
  --npz_image_channels 3 &

run_one 3 \
  --dataset era5_npy \
  --data_root /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --output_dir unified_results/era5_npy_caesar_no_pca \
  --max_samples 16 \
  --max_channels 3 \
  --resolution 240 240 &

run_one 4 \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/kodak_caesar_no_pca \
  --max_samples 24 \
  --resolution 512 512 &

run_one 5 \
  --dataset s2c \
  --data_root /workspace/Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c_caesar_no_pca \
  --max_samples 16 \
  --tile_size 1024 &

run_one 6 \
  --dataset tomo \
  --data_root /workspace/Data/tomo_00001.h5 \
  --output_dir unified_results/tomo_caesar_no_pca \
  --max_samples 16 &

run_one 7 \
  --dataset hurricane \
  --data_root /workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500 \
  --output_dir unified_results/hurricane_caesar_no_pca \
  --max_samples 16 &

wait

run_one 1 \
  --dataset nyx \
  --data_root /workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_caesar_no_pca \
  --max_samples 16 &

run_one 2 \
  --dataset lysozyme \
  --data_root /workspace/Data/lysozyme_processed \
  --output_dir unified_results/lysozyme_caesar_no_pca \
  --max_samples 16 &

wait
