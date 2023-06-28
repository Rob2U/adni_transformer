SWEEP_CONFIG = dict(
    name="test_all_basic",
    project="test_all_basic",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            values=["ResNet18", "ShuffleNetV2"],
        ),
        train_fraction=dict(
            values=[0.7],
        ),
        learning_rate=dict(
            values=[1e-2, 1e-3, 1e-4, 1e-5],
        ),
        batch_size=dict(
            values=[4, 8, 12],
        ),
        max_epochs=dict(
            values=[10],
        ),
        benchmark=dict(
            values=[True],
        ),
    ),
)