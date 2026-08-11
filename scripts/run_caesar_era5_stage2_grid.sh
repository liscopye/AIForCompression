#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/AIForCompression
DATA=/workspace/Data/ERA5/finetune_processed_time_split_t45_v16
VAE="$ROOT/checkpoints/caesar_era5_stability_20260723/d_mr_lam1e4_anchor0_update100.pt"
BASE="$ROOT/checkpoints/caesar/caesar_d.pt"
OUT="$ROOT/checkpoints/caesar_era5_stage2_20260723"
LOG="$ROOT/logs/caesar_era5_stage2_20260723"
PROJECT=caesar-era5-stable-tuning
GROUP=stage2-500-grid-20260723

mkdir -p "$OUT" "$LOG"
pids=()
names=()

cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

launch() {
  local gpu=$1
  local lr=$2
  local anchor=$3
  local name=$4
  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/finetune_caesar_era5.py" \
    --model_type D \
    --stage 2 \
    --device cuda:0 \
    --data_dir "$DATA" \
    --ckpt_path "$BASE" \
    --vae_ckpt_path "$VAE" \
    --output_ckpt "$OUT/$name.pt" \
    --iterations 500 \
    --batch_size 32 \
    --val_batch_size 64 \
    --num_workers 2 \
    --prefetch_factor 2 \
    --train_size 256 \
    --norm_type mean_range \
    --lr "$lr" \
    --warmup_updates 100 \
    --anchor_weight "$anchor" \
    --max_grad_norm 1.0 \
    --log_interval 25 \
    --val_interval 500 \
    --save_interval 500 \
    --milestone_steps 50 100 250 500 \
    --seed 20260723 \
    --wandb_project "$PROJECT" \
    --wandb_group "$GROUP" \
    --wandb_run_name "$name" \
    --wandb_tags era5 stability-grid stage2 \
    --require_wandb \
    >"$LOG/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  printf 'started gpu=%s pid=%s name=%s\n' "$gpu" "$!" "$name"
}

launch 4 1e-6 0 d_s2_lr1e6_anchor0
launch 5 3e-6 0 d_s2_lr3e6_anchor0
launch 6 1e-5 0 d_s2_lr1e5_anchor0
launch 7 3e-6 1 d_s2_lr3e6_anchor1

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'finished name=%s\n' "${names[$index]}"
  else
    printf 'failed name=%s\n' "${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
