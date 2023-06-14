""" Train a model. """

from argparse import ArgumentParser
import os
import lightning as L
import torch
from time import gmtime, strftime
from pytorch_lightning.loggers import WandbLogger
from benchmarks.benchmarks import SamplesPerSecondBenchmark
from trainer import MyTrainer
from dataset import ADNIDataset, ADNIDatasetRAM, ADNIDataModule
from models.resnet import LitADNIResNet
from models.shufflenetV2 import LitADNIShuffleNetV2
from config import SHUFFLENETV2_CONFIG, OPTIMIZER_CONFIG, DATA_CONFIG, TRAINER_CONFIG, WANDB_CONFIG, CHECKPOINT_CONFIG

def load_pretrained_model(pretrained_path, model):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Loading pretrained model from {pretrained_path} ...")
    # Automatically loads the model with the saved hyperparameters
    model = model.__class__.load_from_checkpoint(pretrained_path)
    return model

def get_model(model_name, model_args, optimizer_args):
    """Decides which model to use"""

    if model_name == "LitADNIResNet":
        model = LitADNIResNet(model_args, optimizer_args)
    elif model_name == "LitADNIShuffleNetV2":
        model = LitADNIShuffleNetV2(model_args, optimizer_args)
    return model

def get_logger(dict_args):
    """ Returns a WANDB logger for the model. """
    # set run name to current time
    wandb_logger = WandbLogger(name=strftime("%Y-%m-%d %H:%M:%S", gmtime()), project=dict_args["wandb_project"], log_model=dict_args["log_model"])
    wandb_logger.log_hyperparams(
        {
            "batch_size": dict_args["batch_size"],
            "num_epochs": dict_args["max_epochs"]
        }
    )
    wandb_logger.experiment.define_metric("SamplesPerSecond", summary="mean")
    return wandb_logger

def get_trainer(dict_args, trainer_args):
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=os.path.join(dict_args["checkpoint_path"]),
        filename=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=5,
        mode="min",
    )
    samplesPerSecondBenchmark = SamplesPerSecondBenchmark()

    trainer = MyTrainer(get_logger(dict_args), [checkpoint_callback, samplesPerSecondBenchmark], trainer_args)
    
    return trainer

def main(args):
    """Main function."""
    dict_args = vars(args)
    model_name = dict_args["model_name"]
    model_specific_arg_keys = set(SHUFFLENETV2_CONFIG.keys())    # when there is a new model-config add it this way: set(SHUFFLENETV2_CONFIG.keys()) | set(NEW_CONFIG.keys()) | ...)
    model_args = {key: dict_args[key] for key in dict_args.keys() & model_specific_arg_keys}
    optimizer_args = {key: dict_args[key] for key in dict_args.keys() & set(OPTIMIZER_CONFIG)}
    trainer_args = {key: dict_args[key] for key in dict_args.keys() & set(TRAINER_CONFIG)}
    data_args = {key: dict_args[key] for key in dict_args.keys() & set(DATA_CONFIG)}
    model = get_model(model_name, model_args, optimizer_args) # get the specified model
    trainer = get_trainer(dict_args, trainer_args) # get the trainer
    data = ADNIDataModule(data_args) # get the data
    
    if dict_args["pretrained_path"] is not None:
        model = load_pretrained_model(dict_args["pretrained_path"], model)
    else:
        trainer.fit(model, data)
    
        # load best checkpoint after training
        model = model.__class__.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    
    results = trainer.test(model, data)
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
        default=CHECKPOINT_CONFIG["pretrained_path"],
        help="Path to pretrained model",
    )
    
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=WANDB_CONFIG["wandb_project"],
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
        default=os.path.join(CHECKPOINT_CONFIG["checkpoint_path_without_model_name"], temp_args.wandb_project+"/", temp_args.model_name)
    )
        
    args = parser.parse_args()
    main(args)
