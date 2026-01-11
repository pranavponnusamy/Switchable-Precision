#!/bin/bash

#SBATCH --job-name=train-lora
#SBATCH --account=paceship-efficient_geneol
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --mem-per-gpu=200G
#SBATCH --time=10:00:00
#SBATCH --output=./Nlogs/train-lora/train-lora.out
#SBATCH --error=./Nlogs/train-lora/train-lora.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$USER@gatech.edu

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate eic

export PYTHONNOUSERSITE=1
export HF_HOME="/storage/home/hcoda1/4/pponnusamy7/ps-efficient_geneol-0/EIC-Coding-Test/hf"

python train.py

