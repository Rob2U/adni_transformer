#!/bin/bash -eux
#SBATCH --job-name=adni_benchmarks
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/dhc/home/oliver.zimmermann/logs/adni_transformer%A_%a_%j.log # A - array id, a - job array id, j - job id
#SBATCH --array=0-2%3

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate mml_env
wandb agent --count 1 "allsparks/benchmarks/$sweep_id"