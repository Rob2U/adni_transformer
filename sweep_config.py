SWEEP_CONFIG = dict(
    name="test_sweep",
    project="pretrainingOnADNI",
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
            #values=[16, 32], # may have to increase this value
            values=[32],
        ),
        max_epochs=dict(
            values=[5],
        ),
        benchmark=dict(
            values=[True],
        ),
        hidden_dim_proj_head=dict(
            values=[1024],
            # values=[512, 1024],
        ),
        output_dim_proj_head=dict(
            values=[256],
            # values=[128, 256],
        ),
    ),
)