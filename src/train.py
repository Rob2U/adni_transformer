""" Train a model. """

from argparse import ArgumentParser
import os
import lightning as L
import torch
from time import gmtime, strftime
from pytorch_lightning.loggers import WandbLogger
from trainer import MyTrainer
from dataset import ADNIDataset, ADNIDatasetRAM, ADNIDataModule
from models.resnet import LitADNIResNet
from models.shufflenetV2 import LitADNIShuffleNetV2
from config import WANDB_PROJECT, PRETRAINED_PATH, CHECKPOINT_PATH_WITHOUT_MODELNAME

def load_pretrained_model(pretrained_path, model):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Loading pretrained model from {pretrained_path} ...")
    # Automatically loads the model with the saved hyperparameters
    model = model.__class__.load_from_checkpoint(pretrained_path)
    return model

def get_model(**kwargs):
    """Decides which model to use"""

    if kwargs["model_name"] == "LitADNIResNet":
        model = LitADNIResNet(**kwargs)
    elif kwargs["model_name"] == "LitADNIShuffleNetV2":
        model = LitADNIShuffleNetV2(**kwargs)
    return model

def get_logger(**kwargs):
    """ Returns a WANDB logger for the model. """
    wandb_logger = WandbLogger(project=kwargs["wandb_project"], log_model=kwargs["log_model"])
    wandb_logger.log_hyperparams(
        {
            "batch_size": kwargs["batch_size"],
            "learning_rate": kwargs["learning_rate"],
            "num_epochs": kwargs["max_epochs"]
        }
    )
    return wandb_logger

def train_model(model, logger, **kwargs):
    """Trains a model and returns the best model after training."""
    # add logger and checkpoint callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(kwargs["checkpoint_path"]),
        filename=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = MyTrainer(logger, callbacks=[checkpoint_callback], **kwargs)
    data = ADNIDataModule(**kwargs)
    
    trainer.fit(model, data)
    
    # load best checkpoint after training
    model = model.__class__.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    return model

def main(args):
    """Main function."""
    dict_args = vars(args)
    model = get_model(**dict_args) # get the specified model
    
    if dict_args["pretrained_path"] is not None:
        model = load_pretrained_model(dict_args["pretrained_path"], model)
    else:
        model = train_model(model,get_logger(**dict_args), **dict_args)
    
    model, results = trainer.test(model, data)
    print(results)

def add_global_args(parser):
    """Adds global arguments to the parser."""
    parser.add_argument(
        "--model_name",
        type=str,
        default="LitADNIResNet",
        help="LitADNIResNet or different model",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=PRETRAINED_PATH,
        help="Path to pretrained model",
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="adni_transformer",
        help="Name of the wandb project",
    )
    return parser

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
    parser = add_global_args(parser)
    # figure out which model to use
    
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()
    # let the model add what it needs
    if temp_args.model_name == "LitADNIResNet":
        parser = LitADNIResNet.add_model_specific_args(parser)
    elif temp_args.model_name == "LitADNIShuffleNetV2":
        parser = LitADNIShuffleNetV2.add_model_specific_args(parser)
        
    # add modelname to checkpoint path
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=os.path.join(CHECKPOINT_PATH_WITHOUT_MODELNAME, temp_args.model_name)
    )
        
    args = parser.parse_args()
    main(args)
