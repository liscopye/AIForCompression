#!/bin/bash
#SBATCH --job-name=test_selected_ch
#SBATCH --partition=gpu_5090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --output=test_selected_ch_%j.log
#SBATCH --error=test_selected_ch_%j.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate /data/run01/scxj523/zsh/envs/zsh

echo "Hostname: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no gpu')"
echo "Python: $(which python)"

cd /data/run01/scxj523/zsh/project/AIForCompression

python -u test_selected_channels.py \
    --data_root /data/run01/scxj523/zsh/project/AIForCompression/Data/ERA5/2024 \
    --max_samples 1
