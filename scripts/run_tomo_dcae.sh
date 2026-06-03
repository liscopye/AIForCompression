#!/bin/bash
#SBATCH --job-name=tomo_dcae
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_dcae_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_dcae_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA_FILE=/data/run01/scxj523/zsh/project/Data/tomo/tomo_00001_rec/recon.h5
OUTPUT_DIR=$PROJECT_ROOT/unified_results/tomo_recon3ch

cd $PROJECT_ROOT

python scripts/run_dataset_compression.py \
  --dataset tomo \
  --data_root $DATA_FILE \
  --output_dir $OUTPUT_DIR \
  --models DCAE \
  --max_samples 90 \
  --tomo_group_frames 3 \
  --image_eval_mode real
