#!/bin/bash
# Launch CAESAR fine-tuning with clean GPU isolation.
# Usage:
#   bash scripts/run_finetune.sh V         # CAESAR-V on GPU 0
#   bash scripts/run_finetune.sh D1        # CAESAR-D Stage 1 on GPU 1
#   bash scripts/run_finetune.sh D2        # CAESAR-D Stage 2 on GPU 2

set -e

# ── per-task config ──
case "$1" in
  V)
    MODEL_TYPE="V"
    STAGE="1"
    GPU="0"
    ITERS=100000
    BATCH=24
    ;;
  D1)
    MODEL_TYPE="D"
    STAGE="1"
    GPU="1"
    ITERS=100000
    BATCH=16
    ;;
  D2)
    MODEL_TYPE="D"
    STAGE="2"
    GPU="2"
    ITERS=200000
    BATCH=64
    VAE_CKPT="/workspace/AIForCompression/checkpoints/caesar/caesar_d_tuning_lysozyme_vae.pt"
    ;;
  *)
    echo "Usage: bash scripts/run_finetune.sh {V|D1|D2}"
    exit 1
    ;;
esac

# ── clean env ──
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=$GPU

# ── activate ──
source /workspace/ai4cp/bin/activate

echo "=== GPU $GPU | Model $MODEL_TYPE Stage $STAGE | Batch $BATCH ==="
python -c "import torch; print(f'CUDA devices visible: {torch.cuda.device_count()}, using GPU: {torch.cuda.get_device_name(0)}')"

# ── run ──
if [ "$STAGE" = "2" ]; then
  python /workspace/AIForCompression/scripts/finetune_caesar.py \
    --model_type "$MODEL_TYPE" --stage "$STAGE" \
    --iterations "$ITERS" --batch_size "$BATCH" \
    --device cuda --val_interval 2000 --save_interval 10000 \
    --vae_ckpt_path "$VAE_CKPT"
else
  python /workspace/AIForCompression/scripts/finetune_caesar.py \
    --model_type "$MODEL_TYPE" --stage "$STAGE" \
    --iterations "$ITERS" --batch_size "$BATCH" \
    --device cuda --val_interval 2000 --save_interval 10000
fi
