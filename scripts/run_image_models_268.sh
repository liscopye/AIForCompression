#!/bin/bash
#SBATCH --job-name=image_models_268
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/image_models_268_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/image_models_268_%j.log

set -eo pipefail

ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA=/data/run01/scxj523/zsh/project/Data/ERA5/2024
LOG_ROOT="$ROOT/logs/results"

mkdir -p "$ROOT/logs" "$LOG_ROOT"

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh
set -u
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

cd "$ROOT/models/DCAE"
python -u test_era5.py \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/dcae" \
  --output_dir "$LOG_ROOT/DCAE" \
  --compress \
  --max_samples 1

cd "$ROOT/models/LIC_TCM"
python -u test_era5.py \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/lictcm" \
  --output_dir "$LOG_ROOT/LIC_TCM" \
  --compress \
  --max_samples 1

cd "$ROOT/models/WeConvene"
python -u test_era5.py \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/weconvene" \
  --output_dir "$LOG_ROOT/WeConvene" \
  --compress \
  --max_samples 1

cd "$ROOT"
python -u scripts/test_extra_image_era5.py \
  --model LIC-HPCM-base \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/lic-hpcm/hpcm-base/mse" \
  --output_dir "$LOG_ROOT/LIC-HPCM-base" \
  --max_samples 1

python -u scripts/test_extra_image_era5.py \
  --model LIC-HPCM-large \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/lic-hpcm/hpcm-large/mse" \
  --output_dir "$LOG_ROOT/LIC-HPCM-large" \
  --max_samples 1

python -u scripts/test_extra_image_era5.py \
  --model RwkvCompress \
  --data_root "$DATA" \
  --ckpt_dir "$ROOT/checkpoints/rwkvcompress/mse" \
  --output_dir "$LOG_ROOT/RwkvCompress" \
  --max_samples 1
