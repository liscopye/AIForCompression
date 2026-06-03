#!/bin/bash
#SBATCH --job-name=caesar_eb
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_eb_sweep_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_eb_sweep_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA=/data/run01/scxj523/zsh/project/Data/ERA5/2024
LOG_ROOT="$ROOT/logs/results"

mkdir -p "$ROOT/logs"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd "$ROOT"

EB_VALUES=(5e-3 1e-2 3e-2 5e-2 8e-2 1e-1 2e-1)

for EB in "${EB_VALUES[@]}"; do
  EB_LABEL="${EB//./p}"
  EB_LABEL="${EB_LABEL//-/m}"
  OUT_DIR="$LOG_ROOT/CAESAR_eb_${EB_LABEL}"
  mkdir -p "$OUT_DIR"

  echo "=== [$(date)] Running eb=${EB} -> ${OUT_DIR} ==="
  python -u scripts/test_caesar_era5.py \
    --data_root "$DATA" \
    --output_dir "$OUT_DIR" \
    --ckpt_dir "$ROOT/checkpoints/caesar" \
    --model both \
    --batch_size 1 \
    --max_channels 32 \
    --eb "$EB"
  echo "=== [$(date)] Done eb=${EB} ==="
done

echo "=== [$(date)] All done ==="
