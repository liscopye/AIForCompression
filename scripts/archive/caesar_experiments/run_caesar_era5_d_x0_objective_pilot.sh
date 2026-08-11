#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
VAE="${CAESAR_D_X0_VAE:-$ROOT/checkpoints/caesar_era5_d_complete_candidates/lam1em3_decoder_best_original_stage2.pt}"
BASE="${CAESAR_D_X0_BASE:-$ROOT/checkpoints/caesar/caesar_d.pt}"
OUTPUT_DIR="${CAESAR_D_X0_OUTPUT_DIR:-$ROOT/checkpoints/caesar_era5_d_x0_objective_pilot}"
LOG_DIR="${CAESAR_D_X0_LOG_DIR:-$ROOT/logs/caesar_era5_d_x0_objective_pilot}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$VAE"
test -f "$BASE"
wandb login --verify >/dev/null

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'objective=screen_sampling_aligned_stage2_losses\n'
  printf 'updates=500\nlearning_rate=3e-5\n'
  printf 'external_protocol=unchanged_daily_era5_mean_range\n'
  sha256sum "$VAE" "$BASE"
} >"$OUTPUT_DIR/source_manifest.txt"

data_args=(
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
  --netcdf_val_channel_stride 16
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --gradient_accumulation_steps 2
  --diffusion_steps 32
  --num_workers 4
  --prefetch_factor 2
  --norm_type mean_range
  --ckpt_path "$BASE"
  --vae_ckpt_path "$VAE"
  --iterations 500
  --lr 3e-5
  --warmup_updates 100
  --log_interval 10
  --val_interval 100
  --save_interval 500
  --milestone_steps 50 100 250 500
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group d-stage2-x0-objective-pilot
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local objective="$2"
  local weight="$3"
  local name="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${data_args[@]}" \
    --diffusion_objective "$objective" \
    --diffusion_x0_weight "$weight" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    --wandb_tags era5 caesar-d stage2 x0-objective pilot \
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

run_one 3 hybrid 0.01 hybrid_w001 & pids+=("$!")
run_one 4 hybrid 0.1 hybrid_w01 & pids+=("$!")
run_one 5 hybrid 1.0 hybrid_w1 & pids+=("$!")
run_one 6 x0 1.0 x0_only & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
pids=()
date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
trap - INT TERM EXIT
exit "$failed"
