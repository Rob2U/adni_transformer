#!/bin/bash -eux
#SBATCH --job-name=adni_transformer_pretraining
#SBATCH --output=./logs/pretraining%j.out
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH  --gpus=1 # -G
#SBATCH --mem=32gb # -m
#SBATCH --time=08:00:00 

# Run python script
eval "$(conda shell.bash hook)"
conda activate mml_env
python ~/adni_transformer/src/train.py --dataset=PretrainADNI --model_name=MaskedAutoencoder --test_fraction=0.0 --validation_fraction=0.2 --max_epochs=1
