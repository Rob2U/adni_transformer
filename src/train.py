""" Train a model. """

from argparse import ArgumentParser
import os
import lightning as L
import torch
from pytorch_lightning.loggers import WandbLogger
from trainer import MyTrainer
from dataset import ADNIDataset, ADNIDatasetRAM, ADNIDataModule
from resnet import LitADNIResNet
from shufflenetV2 import LitADNIShuffleNetV2
from config import WANDB_PROJECT


def get_model(**kwargs):
    """Decides which model to use"""

    if kwargs["model_name"] == "LitADNIResNet":
        model = LitADNIResNet(**kwargs)
    elif kwargs["model_name"] == "LitADNIShuffleNetV2":
        model = LitADNIShuffleNetV2(**kwargs)
    return model


def train_model(model, trainer, data):
    """Trains a model and returns the best model after training."""
    trainer.fit(model, data)
    # load best checkpoint after training
    model = model.__class__.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return model


# loads a pretrained model
def get_pretrained_model(pretrained_filename):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    # Automatically loads the model with the saved hyperparameters
    model = model.__class__.load_from_checkpoint(pretrained_filename)

    return model


def run_model(model, data, trainer, **kwargs):
    """Tests a trained or loaded model."""
    pretrained_filename = os.path.join(
        os.path.join(kwargs["checkpoint_path"], kwargs["root"]),
        "3DMLP-epoch=00-val_loss=1.56.ckpt",
    )
    print("\n pretrained_filename:", pretrained_filename, "\n\n")

    if os.path.isfile(pretrained_filename):
        model = get_pretrained_model(pretrained_filename)
    else:
        model = train_model(model, trainer, data)

    results = trainer.test(model, datamodule=data)
    # wandb.save(model)
    return model, results


def main(args):
    """Main function."""
    dict_args = vars(args)
    model = get_model(**dict_args)

    wandb_logger = WandbLogger(project=WANDB_PROJECT, log_model=dict_args["log_model"])
    wandb_logger.log_hyperparams(
        {
            "batch_size": dict_args["batch_size"],
            "learning_rate": dict_args["learning_rate"],
            "num_epochs": dict_args["max_epochs"]
        }
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(dict_args["checkpoint_path"], dict_args["root"]),
        filename=dict_args["root"] + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = MyTrainer(wandb_logger, callbacks=[checkpoint_callback], **dict_args)
    data = ADNIDataModule(**dict_args)
    model, results = run_model(model, data, trainer, **dict_args)
    print(results)


if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # beginn with parsing arguments
    parser = ArgumentParser()
    parser = MyTrainer.add_trainer_args(parser)
    parser = ADNIDataModule.add_data_specific_args(parser)
    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="LitADNIResNet",
        help="LitADNIResNet or different model",
    )
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # let the model add what it needs
    if temp_args.model_name == "LitADNIResNet":
        parser = LitADNIResNet.add_model_specific_args(parser)
    elif temp_args.model_name == "LitADNIShuffleNetV2":
        parser = LitADNIShuffleNetV2.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
