#!/bin/bash -eux
#SBATCH --job-name=test_basic_params # -J
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/dhc/home/robert.weeke/logs/%A_%a_%j.log # A - array id, a - job array id, j - job id
#SBATCH --array=0-24%2

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate mml_env
wandb agent --count 1 "allsparks/test_all_basic/$sweep_id"
