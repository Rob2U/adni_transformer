""" Train a model. """

from argparse import ArgumentParser
import os
import lightning as L
import torch
from dataset import MNIST3DModule
from model import LitBasicMLP
from config import ACCELERATOR, CHECKPOINT_PATH
from pytorch_lightning.loggers import WandbLogger
from trainer import MyTrainer


def get_model(**dict_args):
    """Decides which model to use"""

    if dict_args["model_name"] == "LitBasicMLP":
        model = LitBasicMLP(**dict_args)
    # elif args.model == "ResNet":
    #   model = ResNet(**dict_args)
    return model


def train_model(model, trainer, data):
    """Trains a model and returns the best model after training."""
    trainer.fit(model, data)
    # load best checkpoint after training
    model = LitBasicMLP.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return model


# loads a pretrained model
def get_pretrained_model(pretrained_filename):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Found pretrained model at {pretrained_filename}, loading...")
    # Automatically loads the model with the saved hyperparameters
    model = LitBasicMLP.load_from_checkpoint(pretrained_filename)

    return model


def run_model(model, data, trainer, **dict_args):
    """Tests a trained or loaded model."""
    pretrained_filename = os.path.join(
        os.path.join(CHECKPOINT_PATH, dict_args["root"]),
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

    wandb_logger = WandbLogger(project="MNIST3D", log_model=dict_args["log_model"])
    wandb_logger.log_hyperparams(
        {
            "batch_size": dict_args["batch_size"],
            "learning_rate": dict_args["learning_rate"],
            "num_epochs": dict_args["max_epochs"],
            "input_dim": dict_args["input_dim"],
            "hidden_dim": dict_args["hidden_dim"],
            "dropout": dict_args["dropout"],
        }
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(CHECKPOINT_PATH, dict_args["root"]),
        filename=dict_args["root"] + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = MyTrainer(wandb_logger, callbacks=[checkpoint_callback], **dict_args)
    data = MNIST3DModule(**dict_args)
    model, results = run_model(model, data, trainer, **dict_args)
    print(results)


if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Device:", ACCELERATOR)

    # beginn with parsing arguments
    parser = ArgumentParser()
    parser = MyTrainer.add_trainer_args(parser)
    parser = MNIST3DModule.add_data_specific_args(parser)
    # figure out which model to use
    parser.add_argument(
        "--model_name",
        type=str,
        default="LitModel",
        help="LitModel or different model",
    )
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # let the model add what it needs
    if temp_args.model_name == "LitModel":
        parser = LitModel.add_model_specific_args(parser)
        parser = LitModel.add_model_specific_args(parser)
    # elif temp_args.model_name == "mnist":
    #    parser = LitMNIST.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
