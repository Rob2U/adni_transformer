""" SWEEP_CONFIG = dict(
    name="Test sweep",
    project="test_sweeps",
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
            values=[0.5,0.6,0.7],
        ),
        learning_rate=dict(
            values=[1e-2, 1e-3, 1e-4, 1e-4],
        ),
        batch_size=dict(
            values=[4, 8],
        ),        
    ),
) """

SWEEP_CONFIG = dict(
    name="Precision Benchmark",
    project="benchmarks",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="val_loss",
        goal="minimize",
    ),
    parameters=dict(
        precision=dict(
            values=["16-mixed", "32"] #"bf16-mixed",
        ),
        model_name=dict(
            values=["ShuffleNetV2"],
        ),
        benchmark=dict(
            values=[True],
        ),
        max_epochs=dict(
            values=[3],
        ),
        batch_size=dict(
            values=[8],
        ),
        num_workers=dict(
            values=[8],
        ),
    ),
)