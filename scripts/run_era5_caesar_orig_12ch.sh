#!/bin/bash
#SBATCH --job-name=era5_caesar_orig12ch
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/era5_caesar_orig12ch_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/era5_caesar_orig12ch_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA_FILE=/data/run01/scxj523/zsh/project/Data/ERA5/2024
OUTPUT_DIR=$PROJECT_ROOT/unified_results/era5_caesar_12ch

cd $PROJECT_ROOT

python scripts/run_dataset_compression.py \
  --dataset era5 \
  --data_root $DATA_FILE \
  --output_dir $OUTPUT_DIR \
  --models caesar_v caesar_d \
  --max_samples -1 \
  --caesar_ckpt_dir $PROJECT_ROOT/checkpoints/caesar \
  --max_channels 12 \
  --caesar_eb 1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3
