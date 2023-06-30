#!/bin/bash -eux
#SBATCH --job-name=shufflenetv2_vit_sweep # -J
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/dhc/home/oliver.zimmermann/logs/%A_%a_%j.log # A - array id, a - job array id, j - job id
#SBATCH --array=0-24%2

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate mml_env
wandb agent --count 1 "allsparks/model-comparison/$sweep_id"
