#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ERA5_SHARD_DIR:-/workspace/Data/ERA5/hourly_center512_shards_20240301_90d}"
SOURCE_DIR="$ROOT/checkpoints/caesar_era5_vd_lowrate_100k"
OUTPUT_DIR="${CAESAR_D_STAGE2_OVERLAP_DIR:-$ROOT/checkpoints/caesar_era5_d_stage2_overlap_5k}"
LOG_DIR="${CAESAR_D_STAGE2_OVERLAP_LOG_DIR:-$ROOT/logs/caesar_era5_d_stage2_overlap_5k}"
BASE="$ROOT/checkpoints/caesar/caesar_d.pt"

source /workspace/ai4cp/bin/activate
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
wandb login --verify >/dev/null

source_lam1em4="$SOURCE_DIR/d_s1_lr1em5_lam1em4_full100k_update100000.pt"
source_lam3em4="$SOURCE_DIR/d_s1_lr1em5_lam3em4_full100k_update100000.pt"
test -f "$BASE"
test -f "$source_lam1em4"
test -f "$source_lam3em4"

{
  date -u '+started_utc=%Y-%m-%dT%H:%M:%SZ'
  printf 'data_dir=%s\n' "$DATA_DIR"
  printf 'objective=matching_diffusion_for_frozen_rate_stage1_latents\n'
  printf 'note=decoder-only Stage-1 updates can be merged later because inference_qlatent is decoder-independent\n'
  printf 'iterations=5000\n'
  printf 'diffusion_steps=32\n'
  printf 'effective_batch=64\n'
  printf 'lr=1e-4\n'
  sha256sum "$BASE" "$source_lam1em4" "$source_lam3em4"
} >"$OUTPUT_DIR/source_manifest.txt"

common=(
  --model_type D
  --stage 2
  --data_backend npy_shards
  --data_dir "$DATA_DIR"
  --train_timesteps 1776
  --val_timesteps 384
  --n_frame 16
  --frame_step 24
  --temporal_stride 1
  --netcdf_val_channel_stride 16
  --netcdf_max_open_file_pairs 8
  --train_size 256
  --batch_size 32
  --gradient_accumulation_steps 2
  --val_batch_size 32
  --num_workers 4
  --prefetch_factor 2
  --iterations 5000
  --diffusion_steps 32
  --lr 1e-4
  --warmup_updates 250
  --log_interval 25
  --val_interval 500
  --save_interval 5000
  --milestone_steps 50 100 250 500 1000 2000 5000
  --norm_type mean_range
  --ckpt_path "$BASE"
  --wandb_project caesar-era5-hourly-tuning
  --wandb_group d-matching-stage2-overlap-5k
  --wandb_tags era5 caesar-d stage2 diffusion matching-latent paper-lr
  --require_wandb
  --device cuda:0
)

run_one() {
  local gpu="$1"
  local tag="$2"
  local vae="$3"
  local name="${tag}_stage2_lr1em4"

  echo "GPU $gpu: $name vae=$(basename "$vae")"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$ROOT/scripts/finetune_caesar_era5.py" \
    "${common[@]}" \
    --vae_ckpt_path "$vae" \
    --output_ckpt "$OUTPUT_DIR/$name.pt" \
    --wandb_run_name "$name" \
    >"$LOG_DIR/$name.log" 2>&1
  touch "$OUTPUT_DIR/$name.done"
}

pids=()
run_one 4 lam1em4 "$source_lam1em4" & pids+=("$!")
run_one 7 lam3em4 "$source_lam3em4" & pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

date -u '+finished_utc=%Y-%m-%dT%H:%M:%SZ' >>"$OUTPUT_DIR/source_manifest.txt"
exit "$failed"
