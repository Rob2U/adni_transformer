"""This is the configuration file for the project."""
"""When adding new hyperparameters, make sure to add them to the parser as well."""

DEFAULTS = dict(

    HYPERPARAMETERS = dict(
        model_name="SimCLR",
        backbone="ShuffleNetV2", # this will be overwritten by the parser / sweep config
        optimizer="Adam",
        learning_rate=1e-3,
        batch_size=64,
        train_fraction=1.0,
        validation_fraction=0.0,
        test_fraction=0.0,
        hidden_dim_proj_head=2048,
        output_dim_proj_head=512,
    ),

    DATALOADING = dict(
        dataset="ADNIPretraining",
        data_dir="/dhc/groups/adni_transformer/adni_128_int/",
        meta_file_path="/dhc/groups/adni_transformer/adni_metadata/df_procAlex_MMSE.csv",
        num_workers=4,
    ),

    COMPUTATION = dict(
        accelerator="cuda",
        devices=1,
        max_epochs=2,
        precision="32",
        compile=False,
    ),

    WANDB = dict(
        wandb_project="pretrainingOnADNI",
        log_model=True,
        sweep=False,
        benchmark=False,
    ),

    CHECKPOINTING = dict(
        checkpoint_path_without_model_name="/dhc/groups/adni_transformer/checkpoints/",
        pretrained_path=None,
        enable_checkpointing=True,
        # "PRETRAINED_PATH" = "/dhc/groups/adni_transformer/checkpoints/benchmarks/LitADNIShuffleNetV2/2023-06-14 09:23:42-epoch=01-val_loss=0.63.ckpt"
    ),
)

MODEL_DEFAULTS = dict(

    ShuffleNetV2=dict(
        width_mult=1.0,
    ),

    ResNet18=dict(),
    ViT=dict(),
    SimCLR=dict(),
    BYOL=dict(),
)

