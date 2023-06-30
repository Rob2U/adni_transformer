SWEEP_CONFIG = dict(
    name="shufflenetv2_vit_sweep",
    project="model-comparison",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            values=["ViT", "ShuffleNetV2"],
        ),
        train_fraction=dict(
            values=[0.5, 0.7],
        ),
        learning_rate=dict(
            values=[1e-1, 1e-2, 1e-3],
        ),
        batch_size=dict(
            values=[64, 128],
        ),
        max_epochs=dict(
            values=[10],
        ),
        benchmark=dict(
            values=[True],
        ),
    ),
)