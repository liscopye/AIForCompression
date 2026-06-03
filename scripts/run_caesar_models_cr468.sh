#!/bin/bash
#SBATCH --job-name=caesar_cr468
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_cr468_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/caesar_cr468_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA=/data/run01/scxj523/zsh/project/Data/ERA5/2024
LOG_ROOT="$ROOT/logs/results"

mkdir -p "$ROOT/logs" "$LOG_ROOT/CAESAR_cr468"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u

cd "$ROOT"
python -u scripts/test_caesar_era5.py \
  --data_root "$DATA" \
  --output_dir "$LOG_ROOT/CAESAR_cr468" \
  --ckpt_dir "$ROOT/checkpoints/caesar" \
  --model both \
  --batch_size 1 \
  --max_channels 32 \
  --eb 1e-3
