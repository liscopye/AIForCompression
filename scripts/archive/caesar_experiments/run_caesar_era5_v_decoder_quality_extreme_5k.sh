#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
SOURCE="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k/v_lr1em5_lam1em3_full100k_update100000.pt"
OUTPUT_DIR="${CAESAR_V_DECODER_EXTREME_DIR:-$ROOT/checkpoints/caesar_era5_v_decoder_quality_extreme_5k}"
LOG_DIR="${CAESAR_V_DECODER_EXTREME_LOG_DIR:-$ROOT/logs/caesar_era5_v_decoder_quality_extreme_5k}"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
test -f "$SOURCE"
wandb login --verify >/dev/null

common=(
  --model_type V
  --stage 1
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 8
  --frame_step 24
  --temporal_stride 8
  --netcdf_val_channel_stride 4
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 5000
  --rate_mode bpp
  --distortion_domain normalized
  --lambda_rate 0
  --warmup_updates 100
  --log_interval 50
  --val_interval 500
  --save_interval 5000
  --milestone_steps 250 500 1000 2000 5000
  --norm_type mean_range
  --ckpt_path "$SOURCE"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group v-lowrate-frozen-encoder-quality-extreme-5k
  --wandb_tags era5 low-rate frozen-encoder quality-recovery extreme-lr
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local scope="$2"
  local lr="$3"
  local name="${scope}_lr${lr//-/m}"

  echo "GPU $gpu: $name scope=$scope lr=$lr"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --trainable_scope "$scope" \
    --lr "$lr" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
names=()

launch() {
  local gpu="$1"
  local scope="$2"
  local lr="$3"
  local name="${scope}_lr${lr//-/m}"
  run_one "$gpu" "$scope" "$lr" &
  pids+=("$!")
  names+=("$name")
}

launch 3 decoder 1e-3
launch 4 decoder 3e-3
launch 5 decoder 1e-2
launch 6 sr 1e-3
launch 7 sr 3e-3

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "finished ${names[$index]}"
  else
    echo "failed ${names[$index]}" >&2
    failed=1
  fi
done
exit "$failed"
