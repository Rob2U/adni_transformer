"""This is the configuration file for the project."""
"""When adding new hyperparameters, make sure to add them to the parser as well."""

DEFAULTS = dict(

    HYPERPARAMETERS = dict(
        model_name="MaskedAutoencoder",
        optimizer="Adam",
        learning_rate=1e-3,
        batch_size=50,
        train_fraction=0.8,
        validation_fraction=0.0,
        test_fraction=0.0,
        train_type="pretrain",
    ),

    DATALOADING = dict(
        dataset="PretrainADNI",
        data_dir="/dhc/groups/adni_transformer/adni_128_int/",
        meta_file_path="/dhc/groups/adni_transformer/adni_metadata/df_procAlex_MMSE.csv",
        num_workers=4,
    ),

    COMPUTATION = dict(
        accelerator="cuda",
        devices=1,
        max_epochs=20,
        precision="16",
        compile=False,
    ),

    WANDB = dict(
        wandb_project="pretrainMaskedAutoencoder",
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
    MaskedAutoencoder = dict(
        img_size = 128,  # pixels per edge of image   
        in_chans=1,  # number of input channels       
        num_frames=16,  # how many of the 128 slices are actually used
        patch_size=8,  # size of one patch-edge regarding the dimensions not corresponding to slicing
        t_patch_size=2, # how many slices contribute to one patch
        encoder_embed_dim=1024, # embedding dimension the decoder uses for each token (i.e each patch)
        encoder_depth=8, # number of encoder layers
        encoder_num_heads=8, # number of attention heads in encoder
        decoder_embed_dim=512, # embedding dimension the decoder uses for each token internally (i.e each patch)
        decoder_depth=2, # number of decoder layers
        decoder_num_heads=4, # number of attention heads in decoder
        mlp_ratio=4.0, # ratio of mlp hidden dim to embedding dim
        mask_ratio=0.9, # ratio of masked patches
    ),
)
