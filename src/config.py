"""This is the configuration file for the project."""

# all of those are combined in the model-args
#---------------------------------------------
SHUFFLENETV2_CONFIG = {
    "width_mult": 1.0,
}
#---------------------------------------------

OPTIMIZER_CONFIG = {
    "learning_rate": 5e-2,
}

DATA_CONFIG = {
    "dataset": "ADNI",
    "data_dir": "/dhc/groups/adni_transformer/adni_128_int/",
    "meta_file_path": "/dhc/groups/adni_transformer/adni_metadata/df_procAlex_MMSE.csv",
    "train_fraction": 0.7,
    "validation_fraction": 0.1,
    "test_fraction": 0.2,
    "batch_size": 4,
    "num_workers": 4,
}

TRAINER_CONFIG = {
    "accelerator": "cuda",
    "devices": 1,
    "min_epochs": 1,
    "max_epochs": 15,
}


WANDB_CONFIG = {
    "wandb_project": "ADNI_parsing",
}

CHECKPOINT_CONFIG = {
    "checkpoint_path_without_model_name": "/dhc/groups/adni_transformer/checkpoints/",
    "pretrained_path": None,
    # "PRETRAINED_PATH" = "/dhc/groups/adni_transformer/checkpoints/benchmarks/LitADNIShuffleNetV2/2023-06-14 09:23:42-epoch=01-val_loss=0.63.ckpt"
}

