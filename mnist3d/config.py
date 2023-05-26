"""This is the configuration file for the mnist3d project."""

INPUT_DIM = 28 * 28 * 28  # because we flatten the input
OUTPUT_DIM = 10
HIDDEN_DIM = 1024
DROPOUT = 0.2

LEARNING_RATE = 3e-5
BATCH_SIZE = 128
MIN_EPOCHS = 1
MAX_EPOCHS = 1

# Dataset
DATA_DIR = "./MNIST"
NUM_WORKERS = (
    1  # I had a problem with this because it was EXTREMLY slow, so I set it to 1
)

# Compute related
ACCELERATOR = "cpu"
DEVICES = 1

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/"
