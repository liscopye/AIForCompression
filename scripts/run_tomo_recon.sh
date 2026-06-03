#!/bin/bash
#SBATCH --job-name=tomo_recon
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8

#SBATCH --output=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_recon_%j.log
#SBATCH --error=/data/run01/scxj523/zsh/project/AIForCompression/logs/tomo_recon_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

export OMP_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=64

python /data/run01/scxj523/zsh/project/AIForCompression/scripts/recon_tomo.py
