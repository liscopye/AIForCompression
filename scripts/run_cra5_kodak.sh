#!/bin/bash
#SBATCH --job-name=cra5_kodak
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_kodak_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/cra5_kodak_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
KODAK_ROOT=/data/run01/scxj523/zsh/project/Data/Kodac
OUTPUT_DIR=$PROJECT_ROOT/unified_results/cra5_kodak

cd $PROJECT_ROOT

python scripts/run_dataset_compression.py \
  --dataset kodak \
  --data_root $KODAK_ROOT \
  --output_dir $OUTPUT_DIR \
  --models CRA5 \
  --max_samples -1 \
  --image_eval_mode real \
  --allow_cra5_adapted
