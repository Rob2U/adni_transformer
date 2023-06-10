#!/bin/bash -eux
#SBATCH --job-name=adni_transformer_job1
#SBATCH --output=./logs/adni_transformer%j.out
#SBATCH --partition=gpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH  --gpus=1 # -G
#SBATCH --mem=16gb # -m
#SBATCH --time=02:00:00 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robert.weeke@student.hpi.uni-potsdam.de

# Run python script
eval "$(conda shell.bash hook)"
conda activate mml_env
python ~/repos/adni_transformer/src/train.py

# execute this file while being in the same directory as this file (output lands then in docs)
# i did not investigate global paths yet