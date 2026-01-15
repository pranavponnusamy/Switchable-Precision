#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export WANDB_API_KEY=""
export WANDB_MODE="${WANDB_MODE:-online}"

NUM_GPUS="${NUM_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

if [ "${NUM_GPUS}" -le 1 ]; then
  exec python train/train.py
else
  exec torchrun --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    train/train.py
fi
