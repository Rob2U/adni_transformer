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
            values=["SimCLR", "BYOL"],
        ),
        backbone=dict(
            values=["ResNet18", "ShuffleNetV2", "ViT"],
        ),
        learning_rate=dict(
            values=[1e-2, 1e-3], 
        ),
        batch_size=dict(
            values=[16, 32], # may have to increase this value
        ),
        max_epochs=dict(
            values=[10],
        ),
        benchmark=dict(
            values=[True],
        ),
    ),
)