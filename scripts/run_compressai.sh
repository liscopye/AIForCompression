#!/bin/bash
#SBATCH --job-name=compressai_era5
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=compressai_era5_%j.log
#SBATCH --error=compressai_era5_%j.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression/CRA5

python run_all_compressai.py \
    --data_root /data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5 \
    --output_dir /data/run01/scxj523/zsh/project/AIForCompression/CRA5/results/all_compressai \
    --max_samples 1 \
    --metrics mse
