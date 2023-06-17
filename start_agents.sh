#!/bin/bash -eux
#SBATCH --job-name=adni_experiments
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=16 # -c
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH #SBATCH --output=./logs/adni_transformer%A_%a_%j.out # A - array id, a - job array id, j - job id
#SBATCH --array=0-9%2

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate mml_env
wandb agent --count 1 "$sweep_id"