#!/bin/bash -eux
#SBATCH --job-name=adni_experiments
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=4 # -c
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/dhc/home/oliver.zimmermann/logs/adni_transformer%A_%a_%j.log # A - array id, a - job array id, j - job id
#SBATCH --array=0-12%4

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate mml_env
wandb agent --count 1 "allsparks/test_sweeps/$sweep_id"