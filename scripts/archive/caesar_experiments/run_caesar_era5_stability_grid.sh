#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/AIForCompression
DATA=/workspace/Data/ERA5/finetune_processed_time_split_t45_v16
OUT="$ROOT/checkpoints/caesar_era5_stability_20260723"
LOG="$ROOT/logs/caesar_era5_stability_20260723"
PROJECT=caesar-era5-stable-tuning
GROUP=stage1-2k-grid-20260723

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
  local model=$2
  local norm=$3
  local lambda_rate=$4
  local anchor_weight=$5
  local name=$6
  local source_ckpt
  if [[ "$model" == V ]]; then
    source_ckpt="$ROOT/checkpoints/caesar/caesar_v.pt"
  else
    source_ckpt="$ROOT/checkpoints/caesar/caesar_d.pt"
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python "$ROOT/scripts/finetune_caesar_era5.py" \
    --model_type "$model" \
    --stage 1 \
    --device cuda:0 \
    --data_dir "$DATA" \
    --ckpt_path "$source_ckpt" \
    --output_ckpt "$OUT/$name.pt" \
    --iterations 2000 \
    --batch_size 32 \
    --val_batch_size 64 \
    --num_workers 2 \
    --prefetch_factor 2 \
    --train_size 256 \
    --norm_type "$norm" \
    --rate_mode bpp \
    --lambda_rate "$lambda_rate" \
    --lr 1e-5 \
    --warmup_updates 500 \
    --anchor_weight "$anchor_weight" \
    --max_grad_norm 1.0 \
    --log_interval 100 \
    --val_interval 2000 \
    --save_interval 2000 \
    --milestone_steps 100 250 500 1000 2000 \
    --seed 20260723 \
    --wandb_project "$PROJECT" \
    --wandb_group "$GROUP" \
    --wandb_run_name "$name" \
    --wandb_tags era5 stability-grid stage1 \
    --require_wandb \
    >"$LOG/$name.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  printf 'started gpu=%s pid=%s name=%s\n' "$gpu" "$!" "$name"
}

launch 0 V mean_range    1e-4 0 v_mr_lam1e4_anchor0
launch 1 V mean_range_hw 1e-4 0 v_hw_lam1e4_anchor0
launch 2 V mean_range_hw 1e-4 1 v_hw_lam1e4_anchor1
launch 3 V mean_range_hw 3e-4 1 v_hw_lam3e4_anchor1
launch 4 D mean_range    1e-4 0 d_mr_lam1e4_anchor0
launch 5 D mean_range_hw 1e-4 0 d_hw_lam1e4_anchor0
launch 6 D mean_range_hw 1e-4 1 d_hw_lam1e4_anchor1
launch 7 D mean_range_hw 3e-4 1 d_hw_lam3e4_anchor1

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    printf 'finished name=%s\n' "${names[$index]}"
  else
    printf 'failed name=%s log=%s/%s.log\n' "${names[$index]}" "$LOG" "${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
