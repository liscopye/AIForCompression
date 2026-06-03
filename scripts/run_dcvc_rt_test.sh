#!/bin/bash
#SBATCH --job-name=dcvc_rt_test
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_rt_test_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/dcvc_rt_test_%j.log

set -euo pipefail

export PATH="/data/run01/scxj523/zsh/envs/zsh/bin:$PATH"
export CONDA_PREFIX=/data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression

echo "=== Running DCVC-RT test ==="
python -u scripts/test_video_intra_era5.py \
  --data_root "/data/run01/scxj523/zsh/project/Data/ERA5/2024" \
  --output_dir "/data/run01/scxj523/zsh/project/AIForCompression/logs/results/DCVC-RT" \
  --model DCVC_RT \
  --max_samples 1
