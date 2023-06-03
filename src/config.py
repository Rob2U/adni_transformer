"""This is the configuration file for the project."""

# Model related constants
# ADNIShuffleNetV2
WIDTH_MULT = 1.0

# Dataset to use
DATASET = "ADNI"

# Training related constants
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
MIN_EPOCHS = 1
MAX_EPOCHS = 15

# dataset  fractions
TRAIN_FRACTION = 0.7
VALIDATION_FRACTION = 0.1
TEST_FRACTION = 0.2

# Dataset related constants
DATA_DIR = "/dhc/groups/adni_transformer/adni_128_int/"
META_FILE_PATH = "/dhc/groups/adni_transformer/adni_metadata/df_procAlex_MMSE.csv"
NUM_WORKERS = (
    4  # I had a problem with this because it was EXTREMLY slow, so I set it to 1
)
NUM_CLASSES = 2

# Compute related constants
ACCELERATOR = "gpu"
DEVICES = 1

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/dhc/groups/adni_transformer/checkpoints/3D_Conv"
