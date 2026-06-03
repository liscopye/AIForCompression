#!/bin/bash
#SBATCH --job-name=video_models_268
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/video_models_268_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/video_models_268_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA=/data/run01/scxj523/zsh/project/Data/ERA5/2024
LOG_ROOT="$ROOT/logs/results"

mkdir -p "$ROOT/logs" "$LOG_ROOT"

export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"

cd "$ROOT"
python -u scripts/test_video_intra_era5.py \
  --data_root "$DATA" \
  --output_dir "$LOG_ROOT/DCVC-FM" \
  --model DCVC_FM \
  --max_samples 1
