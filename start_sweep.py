import wandb
from sweep_config import SWEEP_CONFIG

sweep_id = wandb.sweep(SWEEP_CONFIG)

with open("sweep_id.txt", "w") as file:
    file.write(sweep_id)