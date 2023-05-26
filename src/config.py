"""This is the configuration file for the project."""

# Model related constants
# ...

# Training related constants
LEARNING_RATE = 3e-5
BATCH_SIZE = 128
MIN_EPOCHS = 1
MAX_EPOCHS = 1

# Dataset related constants
DATA_DIR = "/dhc/groups/adni_transformer/adni_128_int/"
NUM_WORKERS = (
    1  # I had a problem with this because it was EXTREMLY slow, so I set it to 1
)

# Compute related constants
ACCELERATOR = "gpu"
DEVICES = 1

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/dhc/groups/adni_transformer/checkpoints/3D_Conv"
