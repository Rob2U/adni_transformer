SWEEP_CONFIG = dict(
    name="ShuffleNetV2_SimCLR_Sweep",
    project="ShuffleNetV2_SimCLR_ADNI",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="train_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            # values=["SimCLR", "BYOL"],
            values=["SimCLR"],
        ),
        backbone=dict(
            # values=["ResNet", "ShuffleNetV2", "ViT"],
            values=["ShuffleNetV2"],
        ),
        learning_rate=dict(
            # values=[1e-2, 1e-3], 
            values=[1e-3],
        ),
        batch_size=dict(
            values=[64], # may have to increase this value
            #values=[4, 8],
        ),
        max_epochs=dict(
            values=[100],
        ),
        benchmark=dict(
            values=[False],
        ),
        hidden_dim_proj_head=dict(
            values=[1024],
            # values=[512, 1024],
        ),
        output_dim_proj_head=dict(
            # values=[256],
            values=[512],
        ),
    ),
)