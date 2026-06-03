#!/bin/bash
#SBATCH --job-name=cra5_tomo
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_tomo_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_tomo_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA_FILE=/data/run01/scxj523/zsh/project/Data/tomo/tomo_00083.h5
OUTPUT_DIR=$PROJECT_ROOT/unified_results/cra5_tomo

cd $PROJECT_ROOT

python scripts/run_dataset_compression.py \
  --dataset tomo \
  --data_root $DATA_FILE \
  --output_dir $OUTPUT_DIR \
  --models CRA5 \
  --max_samples 10 \
  --image_eval_mode real \
  --allow_cra5_adapted
