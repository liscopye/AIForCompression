#!/bin/bash
#SBATCH --job-name=LICTCM_sel
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=LICTCM_sel_%j.log
#SBATCH --error=LICTCM_sel_%j.log

eval "$(/data/home/scxj523/run/miniconda3/bin/conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

cd /data/run01/scxj523/zsh/project/AIForCompression
python test_selected_channels.py --models LIC_TCM
