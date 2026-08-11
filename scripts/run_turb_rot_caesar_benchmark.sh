#!/usr/bin/env bash
set -euo pipefail

ROOT=${AI4C_ROOT:-/workspace/AIForCompression}
DATA=${TURB_ROT_DATA:-/workspace/Turb_Rot_testset.npz}
OUT=${TURB_ROT_OUT:-$ROOT/unified_results/turb_rot_npz}
MAX_CAESAR_SAMPLES=${MAX_CAESAR_SAMPLES:-16}
SECTION_INDEX=${TURB_ROT_SECTION_INDEX:-0}
CAESAR_EB=${CAESAR_EB:-"1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2"}
VENV=${AI4C_VENV:-/workspace/ai4cp}

source "$VENV/bin/activate"
cd "$ROOT"
mkdir -p "$OUT"

read -r -a caesar_eb_args <<< "$CAESAR_EB"

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
