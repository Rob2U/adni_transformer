#!/bin/bash -eux
#SBATCH --job-name=adni_transformer
#SBATCH --output=logs/adni_transformer%j.out
#SBATCH --partition=vcpu # -p
#SBATCH --cpus-per-task=8 # -c
#SBATCH --mem=8gb
#SBATCH --time=00:00:10 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robert.weeke@student.hpi.uni-potsdam.de

# Run python script
eval "$(conda shell.bash hook)"
conda activate mml_env
python train.py