"""This is the configuration file for the project."""

# Model related constants
# ...


# Dataset to use
DATASET = "ADNI"

# Training related constants
LEARNING_RATE = 3e-5
BATCH_SIZE = 8
MIN_EPOCHS = 1
MAX_EPOCHS = 2

# dataset  fractions
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.2

# Dataset related constants
DATA_DIR = "/dhc/groups/adni_transformer/adni_128_int/"
META_FILE_PATH = "/dhc/groups/adni_transformer/adni_metadata/df_procAlex_MMSE.csv"
NUM_WORKERS = (
    1  # I had a problem with this because it was EXTREMLY slow, so I set it to 1
)

# Compute related constants
ACCELERATOR = "gpu"
DEVICES = 1

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/dhc/groups/adni_transformer/checkpoints/3D_Conv"
