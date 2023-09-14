SWEEP_CONFIG = dict(
    name="ShuffleNetV2_ADNI_bs32_100eps",
    project="ShuffleNetV2_ADNI",
    program="src/train.py",
    method="grid",
    metric=dict(
        name="train_loss",
        goal="minimize",
    ),
    parameters=dict(
        model_name=dict(
            # values=["SimCLR", "BYOL"],
            values=["ShuffleNetV2"],
        ),
        width_mult=dict(
            # values=["ResNet", "ShuffleNetV2", "ViT"],
            values=[1.0, 2.0],
            # values=[1.0],
        ),
        learning_rate=dict(
            # values=[1e-2, 1e-3], 
            values=[1e-2, 1e-3, 1e-4], # 1e-3 originally
        ),
        batch_size=dict(
            values=[16], # may have to increase this value
            #values=[4, 8],
        ),
        max_epochs=dict(
            values=[25],
        ),
        benchmark=dict(
            values=[False],
        ),
        # hidden_dim_proj_head=dict(
        #     values=[1024, 2048],
        #     # values=[512, 1024],
        # ),
        # output_dim_proj_head=dict(
        #     # values=[256],
        #     values=[128, 256],
        # ),
        dataset=dict(
            values=["ADNI"],
        ),
    ),
)