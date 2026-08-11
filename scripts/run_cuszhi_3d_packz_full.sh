#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
PYTHON=${PYTHON:-python}
RUNNER="$ROOT/scripts/run_external_scientific_codecs.py"
LOG_DIR="$ROOT/logs/cuszhi_3d_packz_full"

mkdir -p "$LOG_DIR"
cd "$ROOT"

run_cuszhi_packz() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "[$(date -Is)] start $name on GPU $gpu" | tee "$LOG_DIR/${name}.status"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u "$RUNNER" \
    "$@" \
    --models cuSZ-Hi-3D \
    --cuszhi_sample_mode whole3d \
    --cuszhi_pack_z \
    --max_samples 1 \
    > "$LOG_DIR/${name}.log" 2>&1
  echo "[$(date -Is)] done $name" | tee -a "$LOG_DIR/${name}.status"
}

# Pack-z policy:
# - E3SM/Turb_Rot: fixed variable+section, all available time as z.
# - ERA5: fixed time, all 268 variables as z.
# - Hurricane: all 100 timesteps as z.
# - NYX/Tomo: spatial slices as z; Tomo is cropped to keep the volume practical.
# - Lysozyme: chunk/frame flattened, z=500.
# - S2C/Kodak: pseudo-3D references, multi-band or image/RGB stacked as z.

run_cuszhi_packz 0 e3sm_npz \
  --dataset e3sm_npz \
  --data_root /workspace/Data/E3SM/caesar_processed/e3sm_5vars_paperlike240_first800_caesar.npz \
  --output_dir unified_results/e3sm_npz_external_models_n64/cuszhi_3d_packz_full_n1 \
  --npz_variable_index 0 \
  --section_index 0 \
  --eb 0.5 0.25 0.1 0.05 0.02 0.01 0.005 &

run_cuszhi_packz 1 turb_rot_npz \
  --dataset turb_rot_npz \
  --data_root /workspace/Data/Turb_Rot_testset.npz \
  --output_dir unified_results/turb_rot_npz_external_models_n64/cuszhi_3d_packz_full_n1 \
  --npz_variable_index 0 \
  --section_index 0 \
  --eb 0.5 0.25 0.1 0.05 0.02 0.01 0.005 &

run_cuszhi_packz 2 era5_npy \
  --dataset era5_npy \
  --data_root /workspace/Data/ERA5/finetune_processed/era5_test.npy \
  --output_dir unified_results/era5_npy_external_models_c3_t16_240/cuszhi_3d_packz_full_n1 \
  --era5_time_start 0 \
  --eb 0.2 0.1 0.05 0.01 0.005 0.001 0.0005 &

run_cuszhi_packz 3 kodak \
  --dataset kodak \
  --data_root /workspace/Data/Kodac \
  --output_dir unified_results/kodak_external_models/cuszhi_3d_packz_full_n1 \
  --kodak_stack_images 24 \
  --eb 0.5 0.3 0.2 0.1 0.05 0.02 0.01 &

run_cuszhi_packz 4 s2c \
  --dataset s2c \
  --data_root /workspace/Data/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911/S2C_MSIL2A_20260509T022531_N0512_R046_T51RUQ_20260509T055911.SAFE \
  --output_dir unified_results/s2c_external_models_n16/cuszhi_3d_packz_full_n1 \
  --tile_size 1024 \
  --s2c_bands B02 B03 B04 B08 \
  --eb 0.5 0.3 0.2 0.15 0.1 0.075 0.05 &

run_cuszhi_packz 5 tomo \
  --dataset tomo \
  --data_root /workspace/Data/tomo_00001.h5 \
  --output_dir unified_results/tomo_external_models_n16/cuszhi_3d_packz_full_n1 \
  --resolution 512 512 \
  --cuszhi_z_depth 512 \
  --eb 0.2 0.1 0.05 0.02 0.01 0.005 0.001 &

run_cuszhi_packz 6 hurricane \
  --dataset hurricane \
  --data_root /workspace/Data/SDRBENCH-Hurricane-ISABEL-100x500x500/100x500x500 \
  --output_dir unified_results/hurricane_external_models_n16/cuszhi_3d_packz_full_n1 \
  --eb 0.7 0.5 0.3 0.2 0.1 0.05 0.02 &

run_cuszhi_packz 7 nyx \
  --dataset nyx \
  --data_root /workspace/Data/SDRBENCH-EXASKY-NYX-512x512x512/SDRBENCH-EXASKY-NYX-512x512x512 \
  --output_dir unified_results/nyx_external_models_n16/cuszhi_3d_packz_full_n1 \
  --eb 0.5 0.2 0.1 0.05 0.02 0.01 0.005 &

wait

run_cuszhi_packz 0 lysozyme \
  --dataset lysozyme \
  --data_root /workspace/Data/lysozyme_processed \
  --output_dir unified_results/lysozyme_external_models_n16/cuszhi_3d_packz_full_n1 \
  --cuszhi_z_depth 500 \
  --eb 0.5 0.005 0.002 0.001 0.0005 0.0002 0.0001

echo "[$(date -Is)] all done"
