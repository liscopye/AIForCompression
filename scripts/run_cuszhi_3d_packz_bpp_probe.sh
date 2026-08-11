#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
PYTHON=${PYTHON:-python}
RUNNER="$ROOT/scripts/run_external_scientific_codecs.py"
LOG_DIR="$ROOT/logs/cuszhi_3d_packz_bpp_probe"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_probe() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[$(date -Is)] start $name on GPU $gpu" | tee "$LOG_DIR/${name}.status"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$RUNNER" \
    "$@" \
    --models cuSZ-Hi-3D \
    --cuszhi_sample_mode whole3d \
    --cuszhi_pack_z \
    --cuszhi_min_abs_eb 1e-12 \
    --max_samples 1 \
    --no_lpips \
    > "$LOG_DIR/${name}.log" 2>&1
  echo "[$(date -Is)] done $name" | tee -a "$LOG_DIR/${name}.status"
}

COMMON_EB=(2 1 0.5 0.25 0.1 0.05 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001)
NYX_EB=(10 2 0.5 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001 0.0000000001)
LYSO_EB=(2 0.5 0.005 0.001 0.0002 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001)

run_probe 0 e3sm_npz \
  --dataset e3sm_npz \
  --data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz \
  --output_dir unified_results/e3sm_npz_external_models_n64/cuszhi_3d_packz_bpp_probe_n1 \
  --npz_variable_index 0 \
  --section_index 0 \
  --eb "${COMMON_EB[@]}" &

run_probe 1 turb_rot_npz \
  --dataset turb_rot_npz \
  --data_root /workspace/Turb_Rot_testset.npz \
  --output_dir unified_results/turb_rot_npz_external_models_n64/cuszhi_3d_packz_bpp_probe_n1 \
  --npz_variable_index 0 \
  --section_index 0 \
  --eb "${COMMON_EB[@]}" &

run_probe 2 era5_npy \
  --dataset era5_npy \
  --data_root /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --output_dir unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d_packz_bpp_probe_n1 \
  --era5_time_start 0 \
  --eb "${COMMON_EB[@]}" &

run_probe 3 kodak \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/kodak_external_models/cuszhi_3d_packz_bpp_probe_n1 \
  --kodak_stack_images 24 \
  --eb "${COMMON_EB[@]}" &

run_probe 4 s2c \
  --dataset s2c \
  --data_root /workspace/Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c_external_models_n16/cuszhi_3d_packz_bpp_probe_n1 \
  --tile_size 1024 \
  --s2c_bands B02 B03 B04 B08 \
  --eb "${COMMON_EB[@]}" &

run_probe 5 tomo \
  --dataset tomo \
  --data_root /workspace/Data/tomo_00001.h5 \
  --output_dir unified_results/tomo_external_models_n16/cuszhi_3d_packz_bpp_probe_n1 \
  --resolution 512 512 \
  --cuszhi_z_depth 512 \
  --eb "${COMMON_EB[@]}" &

run_probe 6 hurricane \
  --dataset hurricane \
  --data_root /workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500 \
  --output_dir unified_results/hurricane_external_models_n16/cuszhi_3d_packz_bpp_probe_n1 \
  --eb "${COMMON_EB[@]}" &

run_probe 7 nyx \
  --dataset nyx \
  --data_root /workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_external_models_n16/cuszhi_3d_packz_bpp_probe_n1 \
  --eb "${NYX_EB[@]}" &

wait

run_probe 0 lysozyme \
  --dataset lysozyme \
  --data_root /workspace/Data/lysozyme_processed \
  --output_dir unified_results/lysozyme_external_models_n16/cuszhi_3d_packz_bpp_probe_n1 \
  --cuszhi_z_depth 500 \
  --eb "${LYSO_EB[@]}"

echo "[$(date -Is)] all done"
