import wandb
from sweep_config import SWEEP_CONFIG

sweep_id = wandb.sweep(SWEEP_CONFIG)

with open("sweep_id.txt", "w") as file:
    file.write(sweep_id)


# use this to start a sweep
# you can than start an agent with wandb agent count=... allsparks/test_sweeps/sweep_id