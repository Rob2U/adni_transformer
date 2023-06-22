SWEEP_CONFIG = dict(
    name="exp_shufflenet_1",
    project="shufflenetv2",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            values=["ShuffleNetV2"],
        ),
        train_fraction=dict(
            values=[0.7],
        ),
        learning_rate=dict(
            values=[1e-2, 1e-3, 1e-4],
        ),
        batch_size=dict(
            values=[4, 8],
        ),
        max_epochs=dict(
            values=[10],
        ),
        benchmark=dict(
            values=[True],
        ),
    ),
)
