#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
VAE="${CAESAR_D_SPECIALIST_VAE:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam1em3_decoder_best_original_stage2.pt}"
BASE="${CAESAR_D_SPECIALIST_BASE:-$ROOT/checkpoints/caesar/caesar_d.pt}"
OUTPUT_DIR="${CAESAR_D_SPECIALIST_OUTPUT_DIR:-$ROOT/checkpoints/caesar_era5_d_hard_channel_specialists}"
LOG_DIR="${CAESAR_D_SPECIALIST_LOG_DIR:-$ROOT/logs/caesar_era5_d_hard_channel_specialists}"
UPDATES="${CAESAR_D_SPECIALIST_UPDATES:-1000}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -d "$DATA_DIR"
test -f "$VAE"
test -f "$BASE"
wandb login --verify >/dev/null

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'objective=screen_stage2_specialization_on_hard_era5_variable_groups\n'
  printf 'updates=%s\n' "$UPDATES"
  printf 'groups=specific_humidity[37,74),relative_humidity[185,222),single_level[259,268)\n'
  printf 'external_protocol=unchanged_daily_era5_mean_range\n'
  sha256sum "$VAE" "$BASE"
} >"$OUTPUT_DIR/source_manifest.txt"

common_args=(
  --model_type D
  --stage 2
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --interpo_rate 3
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --diffusion_steps 32
  --diffusion_objective noise
  --num_workers 4
  --prefetch_factor 2
  --norm_type mean_range
  --ckpt_path "$BASE"
  --vae_ckpt_path "$VAE"
  --iterations "$UPDATES"
  --warmup_updates 100
  --log_interval 25
  --val_interval 250
  --save_interval "$UPDATES"
  --milestone_steps 50 100 250 500 1000
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group d-stage2-hard-channel-specialists
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local channel_start="$2"
  local channel_end="$3"
  local lr="$4"
  local name="$5"
  local channel_count=$((channel_end - channel_start))
  local micro_batch=32
  local accumulation=2
  local val_stride=4
  if (( channel_count < micro_batch )); then
    micro_batch="$channel_count"
    accumulation=$(((64 + micro_batch / 2) / micro_batch))
  fi
  if (( channel_end - channel_start < 16 )); then
    val_stride=1
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common_args[@]}" \
    --netcdf_val_channel_stride "$val_stride" \
    --batch_size "$micro_batch" \
    --val_batch_size "$micro_batch" \
    --gradient_accumulation_steps "$accumulation" \
    --train_channel_start "$channel_start" \
    --train_channel_end "$channel_end" \
    --lr "$lr" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    --wandb_tags era5 caesar-d stage2 specialist pilot \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

run_one 3 37 74 1e-4 specific_humidity_lr1em4 & pids+=("$!")
run_one 4 185 222 1e-4 relative_humidity_lr1em4 & pids+=("$!")
run_one 5 259 268 1e-4 single_level_lr1em4 & pids+=("$!")
run_one 6 37 74 3e-5 specific_humidity_lr3em5 & pids+=("$!")
run_one 7 185 222 3e-5 relative_humidity_lr3em5 & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
trap - INT TERM EXIT
exit "$failed"
