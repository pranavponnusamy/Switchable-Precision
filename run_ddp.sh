#!/usr/bin/env bash
set -euo pipefail

# W&B
export WANDB_API_KEY="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"
export WANDB_MODE="${WANDB_MODE:-online}"

# Infer GPU count
NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

# Fallback to single process if 0 or 1 GPU
if [ "${NUM_GPUS}" -le 1 ]; then
  echo "Running single-process training (${NUM_GPUS} GPU(s) detected)"
  exec python train.py
else
  echo "Running DDP training with ${NUM_GPUS} GPUs"
  exec torchrun --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    train.py
fi
