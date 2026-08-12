#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN="/workspace/Data/lysozyme_processed/mmap/lysozyme_train_nf16.npy"
VAL="/workspace/Data/lysozyme_processed/mmap/lysozyme_val_nf16.npy"
OUT="$ROOT/checkpoints/caesar_lysozyme"
LOG="$ROOT/logs/caesar_lysozyme_retrain"
TRAINER="$ROOT/scripts/finetune_caesar_fixed.py"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUT" "$LOG"
test -f "$TRAIN"
test -f "$VAL"
test -f "$ROOT/checkpoints/caesar/caesar_v.pt"
test -f "$ROOT/checkpoints/caesar/caesar_d.pt"

upload_artifact() {
  local path="$1"
  local name="$2"
  wandb artifact put \
    --name "caesar-finetune/$name" \
    --type model \
    --alias best \
    --description "Lysozyme retrain with independent validation split; seed 2025" \
    "$path"
}

run_v() {
  CUDA_VISIBLE_DEVICES=0 python -u "$TRAINER" \
    --model_type V --stage 1 --device cuda:0 \
    --data_path "$TRAIN" --val_data_path "$VAL" \
    --iterations 100000 --batch_size 32 \
    --lr 1e-4 --lambda_rate 1e-5 --rate_mode bits \
    --val_interval 2000 --save_interval 10000 \
    --num_workers 4 --prefetch_factor 2 --seed 2025 \
    --output_ckpt "$OUT/caesar_v_tuning_lysozyme.pt" \
    --wandb_project caesar-finetune \
    2>&1 | tee "$LOG/caesar_v.log"
  upload_artifact "$OUT/caesar_v_tuning_lysozyme.pt" caesar-v-lysozyme-retrain
}

run_d() {
  CUDA_VISIBLE_DEVICES=1 python -u "$TRAINER" \
    --model_type D --stage 1 --device cuda:0 \
    --data_path "$TRAIN" --val_data_path "$VAL" \
    --iterations 100000 --batch_size 32 \
    --lr 1e-4 --lambda_rate 1e-5 --rate_mode bits \
    --val_interval 2000 --save_interval 10000 \
    --num_workers 4 --prefetch_factor 2 --seed 2025 \
    --output_ckpt "$OUT/caesar_d_tuning_lysozyme_vae.pt" \
    --wandb_project caesar-finetune \
    2>&1 | tee "$LOG/caesar_d_stage1.log"
  upload_artifact "$OUT/caesar_d_tuning_lysozyme_vae.pt" caesar-d-lysozyme-stage1-retrain

  CUDA_VISIBLE_DEVICES=1 python -u "$TRAINER" \
    --model_type D --stage 2 --device cuda:0 \
    --data_path "$TRAIN" --val_data_path "$VAL" \
    --vae_ckpt_path "$OUT/caesar_d_tuning_lysozyme_vae.pt" \
    --ckpt_path "$ROOT/checkpoints/caesar/caesar_d.pt" \
    --iterations 200000 --batch_size 32 --gradient_accumulation_steps 2 \
    --lr 1e-4 --lambda_rate 1e-5 --rate_mode bits \
    --val_interval 2000 --save_interval 10000 \
    --num_workers 4 --prefetch_factor 2 --seed 2025 \
    --output_ckpt "$OUT/caesar_d_tuning_lysozyme.pt" \
    --wandb_project caesar-finetune \
    2>&1 | tee "$LOG/caesar_d_stage2.log"
  upload_artifact "$OUT/caesar_d_tuning_lysozyme.pt" caesar-d-lysozyme-retrain
}

run_v &
pid_v=$!
run_d &
pid_d=$!

status=0
wait "$pid_v" || status=1
wait "$pid_d" || status=1
exit "$status"
