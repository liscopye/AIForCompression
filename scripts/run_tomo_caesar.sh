#!/bin/bash
#SBATCH --job-name=tomo_caesar
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_caesar_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_caesar_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

PROJECT_ROOT=/data/run01/scxj523/zsh/project/AIForCompression
DATA_FILE=/data/run01/scxj523/zsh/project/Data/tomo/tomo_00001_rec/recon.h5
OUTPUT_DIR=$PROJECT_ROOT/unified_results/tomo_recon_caesar

cd $PROJECT_ROOT

# caesar_v needs 8 frames, caesar_d needs 16 frames — run both in one pass (same sequence, same summary.json)
# eb sweep with dense transition points between 1e-3 and 3e-3:
python scripts/run_dataset_compression.py \
  --dataset tomo \
  --data_root $DATA_FILE \
  --output_dir $OUTPUT_DIR \
  --models caesar_v caesar_d \
  --max_samples 16 \
  --caesar_eb 1e-3 1.2e-3 1.5e-3 1.8e-3 2e-3 2.5e-3 3e-3 5e-3 8e-3 1.2e-2 2e-2
