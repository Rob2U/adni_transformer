#!/bin/bash -eux
#SBATCH --job-name=pretraining_bb_simclr_sweep-1 # -J
#SBATCH --partition=gpupro # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=32G
#SBATCH --gpus=2
#SBATCH --time=08:00:00
#SBATCH --output=/dhc/home/robert.weeke/logs/%A_%a_%j.log # A - array id, a - job array id, j - job id
#SBATCH --array=0-6%2

sweep_id=$(<sweep_id.txt)

eval "$(conda shell.bash hook)"
conda activate adni
wandb agent --count 1 "allsparks/ShuffleNetV2_ADNI/$sweep_id"
