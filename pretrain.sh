#!/bin/bash -eux
#SBATCH --job-name=adni_transformer_pretraining
#SBATCH --output=./logs/pretraining%j.out
#SBATCH --partition=gpua100 # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH  --gpus=1 # -G
#SBATCH --mem=32G # -m
#SBATCH --time=24:00:00 

# Run python script
eval "$(conda shell.bash hook)"
conda activate mml_env
python ~/adni_transformer/src/train.py
