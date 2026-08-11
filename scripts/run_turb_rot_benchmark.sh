#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
DATA=${TURB_ROT_DATA:-/workspace/Data/Turb_Rot_testset.npz}
OUT=${TURB_ROT_OUT:-$ROOT/unified_results/turb_rot_npz}
MAX_IMAGE_SAMPLES=${MAX_IMAGE_SAMPLES:-16}
MAX_CAESAR_SAMPLES=${MAX_CAESAR_SAMPLES:-16}
SECTION_INDEX=${TURB_ROT_SECTION_INDEX:-0}
SECTION_START=${TURB_ROT_SECTION_START:-0}
CAESAR_EB=${CAESAR_EB:-"1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2"}
MAX_MODEL_JOBS=${MAX_MODEL_JOBS:-all}
VENV=${AI4C_VENV:-/workspace/ai4cp}

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p "$OUT" "$ROOT/logs"

model_job_args=()
if [[ "$MAX_MODEL_JOBS" != "all" ]]; then
  model_job_args=(--max_model_jobs "$MAX_MODEL_JOBS")
fi

read -r -a caesar_eb_args <<< "$CAESAR_EB"

python -u scripts/run_dataset_compression.py \
  --dataset turb_rot_npz \
  --data_root "$DATA" \
  --output_dir "$OUT/image_models" \
  --models DCAE LIC-HPCM DCMVC DCVC-RT \
  --max_samples "$MAX_IMAGE_SAMPLES" \
  --turb_rot_section_start "$SECTION_START" \
  "${model_job_args[@]}"

python -u scripts/run_dataset_compression.py \
  --dataset turb_rot_npz \
  --data_root "$DATA" \
  --output_dir "$OUT/caesar_original" \
  --models caesar_v caesar_d \
  --max_samples "$MAX_CAESAR_SAMPLES" \
  --turb_rot_section_index "$SECTION_INDEX" \
  --caesar_ckpt_dir "$ROOT/checkpoints/caesar" \
  --caesar_eb "${caesar_eb_args[@]}"

python -u scripts/run_dataset_compression.py \
  --dataset turb_rot_npz \
  --data_root "$DATA" \
  --output_dir "$OUT/caesar_tuned" \
  --models caesar_v caesar_d \
  --max_samples "$MAX_CAESAR_SAMPLES" \
  --turb_rot_section_index "$SECTION_INDEX" \
  --caesar_ckpt_dir "$ROOT/checkpoints/caesar_tuned" \
  --caesar_eb "${caesar_eb_args[@]}"

python -u utils/plot_turb_rot_results.py \
  --image "$OUT/image_models/summary.json" \
  --caesar_orig "$OUT/caesar_original/summary.json" \
  --caesar_tuned "$OUT/caesar_tuned/summary.json" \
  --output_dir "$OUT/plots"
