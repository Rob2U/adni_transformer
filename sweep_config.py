SWEEP_CONFIG = dict(
    name="shufflenetv2_m3t",
    project="m3t-adni",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            values=["M3T"],
        ),
        train_fraction=dict(
            values=[0.5, 0.7],
        ),
        learning_rate=dict(
            values=[1e-1, 1e-2, 1e-3],
        ),
        batch_size=dict(
            values=[16, 32],
        ),
        max_epochs=dict(
            values=[5],
        ),
        benchmark=dict(
            values=[True],
        ),
    ),
)