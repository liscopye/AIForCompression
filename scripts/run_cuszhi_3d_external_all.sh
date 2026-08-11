#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
PYTHON=${PYTHON:-python}
RUNNER="$ROOT/scripts/run_external_scientific_codecs.py"

cd "$ROOT"

run_cuszhi_3d() {
  echo "=== $* ==="
  "$PYTHON" -u "$RUNNER" "$@" --models cuSZ-Hi-3D --no_lpips
}

run_cuszhi_3d \
  --dataset e3sm_npz \
  --data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz \
  --output_dir unified_results/e3sm_npz_external_models_n64/cuszhi_3d \
  --max_samples 64 \
  --npz_image_mode variables \
  --npz_image_channels 3 \
  --eb 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001

run_cuszhi_3d \
  --dataset turb_rot_npz \
  --data_root /workspace/Turb_Rot_testset.npz \
  --output_dir unified_results/turb_rot_npz_external_models_n64/cuszhi_3d \
  --max_samples 64 \
  --npz_image_mode sections \
  --npz_image_channels 3 \
  --eb 0.1 0.05 0.02 0.01 0.005 0.002 0.001

run_cuszhi_3d \
  --dataset era5_npy \
  --data_root /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --output_dir unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d \
  --max_samples 16 \
  --resolution 240 240 \
  --era5_max_channels 3 \
  --eb 0.1 0.05 0.01 0.005 0.001 0.0005 0.0001

run_cuszhi_3d \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/kodak_external_models/cuszhi_3d \
  --max_samples -1 \
  --eb 0.075 0.1 0.15 0.2 0.3 0.5 0.75

run_cuszhi_3d \
  --dataset s2c \
  --data_root /workspace/Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c_external_models_n16/cuszhi_3d \
  --max_samples 16 \
  --tile_size 1024 \
  --eb 0.075 0.1 0.15 0.2 0.3 0.5 0.75

run_cuszhi_3d \
  --dataset tomo \
  --data_root /workspace/Data/tomo_00001.h5 \
  --output_dir unified_results/tomo_external_models_n16/cuszhi_3d \
  --max_samples 16 \
  --tomo_group_frames 3 \
  --eb 0.08 0.1 0.15 0.2 0.25 0.3 0.35

run_cuszhi_3d \
  --dataset hurricane \
  --data_root /workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500 \
  --output_dir unified_results/hurricane_external_models_n16/cuszhi_3d \
  --max_samples 16 \
  --eb 0.5 0.55 0.6 0.65 0.7 0.8 0.85

run_cuszhi_3d \
  --dataset nyx \
  --data_root /workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_external_models_n16/cuszhi_3d \
  --max_samples 16 \
  --eb 0.5 0.55 0.6 0.65 0.7 0.8 0.85

run_cuszhi_3d \
  --dataset lysozyme \
  --data_root /workspace/Data/lysozyme_processed \
  --output_dir unified_results/lysozyme_external_models_n16/cuszhi_3d \
  --max_samples 16 \
  --eb 0.05 0.08 0.1 0.15 0.2 0.35 0.5
