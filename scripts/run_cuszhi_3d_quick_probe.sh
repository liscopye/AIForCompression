#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
PYTHON=${PYTHON:-python}
RUNNER="$ROOT/scripts/run_external_scientific_codecs.py"
MAX_N=${MAX_N:-4}
MAX_KODAK=${MAX_KODAK:-6}

cd "$ROOT"

run_one() {
  local gpu=$1
  shift
  echo "=== GPU ${gpu}: $* ==="
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$RUNNER" "$@" --models cuSZ-Hi-3D --no_lpips
}

pids=()

run_one 0 \
  --dataset e3sm_npz \
  --data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz \
  --output_dir unified_results/e3sm_npz_external_models_n64/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --npz_image_mode variables \
  --npz_image_channels 3 \
  --eb 0.5 0.25 0.1 0.05 0.02 0.01 0.005 &
pids+=($!)

run_one 1 \
  --dataset turb_rot_npz \
  --data_root /workspace/Turb_Rot_testset.npz \
  --output_dir unified_results/turb_rot_npz_external_models_n64/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --npz_image_mode sections \
  --npz_image_channels 3 \
  --eb 0.5 0.25 0.1 0.05 0.02 0.01 0.005 &
pids+=($!)

run_one 2 \
  --dataset era5_npy \
  --data_root /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --output_dir unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --resolution 240 240 \
  --era5_max_channels 3 \
  --eb 0.5 0.2 0.1 0.05 0.01 0.005 0.001 &
pids+=($!)

run_one 3 \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/kodak_external_models/cuszhi_3d_quick_probe \
  --max_samples "$MAX_KODAK" \
  --eb 0.75 0.5 0.3 0.2 0.1 0.075 &
pids+=($!)

run_one 4 \
  --dataset tomo \
  --data_root /workspace/Data/tomo_00001.h5 \
  --output_dir unified_results/tomo_external_models_n16/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --tomo_group_frames 3 \
  --eb 0.5 0.2 0.1 0.05 0.02 0.01 0.005 &
pids+=($!)

run_one 5 \
  --dataset hurricane \
  --data_root /workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500 \
  --output_dir unified_results/hurricane_external_models_n16/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --eb 1.5 1.0 0.7 0.5 0.3 0.1 0.05 &
pids+=($!)

run_one 6 \
  --dataset nyx \
  --data_root /workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_external_models_n16/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --eb 1.5 1.0 0.7 0.5 0.1 0.05 0.01 &
pids+=($!)

run_one 7 \
  --dataset lysozyme \
  --data_root /workspace/Data/lysozyme_processed \
  --output_dir unified_results/lysozyme_external_models_n16/cuszhi_3d_quick_probe \
  --max_samples "$MAX_N" \
  --eb 0.5 0.2 0.1 0.05 0.02 0.01 0.005 &
pids+=($!)

for pid in "${pids[@]}"; do
  wait "$pid"
done

run_one 0 \
  --dataset s2c \
  --data_root /workspace/Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c_external_models_n16/cuszhi_3d_quick_probe \
  --max_samples 1 \
  --tile_size 1024 \
  --eb 0.75 0.5 0.3 0.2 0.15 0.1 0.075
