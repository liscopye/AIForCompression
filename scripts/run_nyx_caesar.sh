#!/bin/bash
#SBATCH --job-name=nyx_caesar
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/nyx_caesar_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/nyx_caesar_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA_FILE=/data/run01/scxj523/zsh/project/Data/nyx/SDRBENCH-EXASKY-NYX-512x512x512
OUTPUT_DIR=$PROJECT_ROOT/unified_results/nyx_caesar

cd $PROJECT_ROOT

python scripts/run_dataset_compression.py \
  --dataset nyx \
  --data_root $DATA_FILE \
  --output_dir $OUTPUT_DIR \
  --models caesar_v caesar_d \
  --max_samples 16 \
  --caesar_eb 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 8e-3 1.2e-2 2e-2 5e-2 1e-1 5e-1
