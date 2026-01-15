# Efficient LLMs: Switchable & Dynamic Quantization

## Scripts in `train/`
- `train.py`: Main training script supporting switchable precision.
- `train_adv.py`: Script for full finetuning using LORA. 
- `opt_train.py`: Training with cyclic precision schedules (CPT).
- `opt_train_lora.py`: Training with cyclic precision and LoRA adapters.

## Notebooks
- `notebooks/adversarial.ipynb`: Adversarial robustness exploration.

## How to Run
1. Set your `WANDB_API_KEY` in `run.sh`.
2. Run the script:
   ```bash
   bash run.sh
   ```
