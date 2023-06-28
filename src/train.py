""" Train a model. """

from argparse import ArgumentParser
import os
import lightning as L
import torch
from time import gmtime, strftime
from pytorch_lightning.loggers import WandbLogger
from benchmarks.benchmarks import SamplesPerSecondBenchmark, GpuMetricsBenchmark
from dataset import ADNIDataset, ADNIDatasetRAM, ADNIDataModule
from models.resnet import LitADNIResNet
from models.shufflenetV2 import LitADNIShuffleNetV2
from mlparser import ADNIParser
from defaults import DEFAULTS, MODEL_DEFAULTS

def load_pretrained_model(pretrained_path, model):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Loading pretrained model from {pretrained_path} ...")
    # Automatically loads the model with the saved hyperparameters
    model = model.__class__.load_from_checkpoint(pretrained_path)
    return model

def get_model_arguments(model_name, parsed_arguments):
    model_args = {key: parsed_arguments[key] for key in parsed_arguments & MODEL_DEFAULTS[model_name].keys()}
    if model_name == "ResNet18":
        model_args["accelerator"] = parsed_arguments["accelerator"]
    # elif model_name ==
    model_args["learning_rate"] = parsed_arguments["learning_rate"]
    return model_args

def get_model(model_name, model_arguments):
    """Decides which model to use"""
    if model_name == "ResNet18":
        model = LitADNIResNet(model_arguments)
    elif model_name == "ShuffleNetV2":
        model = LitADNIShuffleNetV2(model_arguments)
    return model

def get_logger(arguments):
    """ Returns a WANDB logger for the model. """
    # set run name to current time
    wandb_logger = WandbLogger(name=strftime("%Y-%m-%d %H:%M:%S", gmtime()), project=arguments["wandb_project"], log_model=arguments["log_model"])
    wandb_logger.log_hyperparams(
        {
            "batch_size": arguments["batch_size"],
            "num_epochs": arguments["max_epochs"]
        }
    )
    if(arguments["benchmark"]):
        wandb_logger.experiment.define_metric("SamplesPerSecond", summary="mean")
    return wandb_logger

def get_callbacks(arguments):
    """Returns a list of callbacks for the model."""
    
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(arguments["checkpoint_path"]),
            filename=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,
            mode="min",
        )
    samplesPerSecondBenchmark =  SamplesPerSecondBenchmark()
    gpuMetricsBenchmark = GpuMetricsBenchmark()

    callbacks = [
        checkpoint_callback,
    ]
    # add benchmarking specific callbacks only if neccessary
    if(arguments["benchmark"]):
        callbacks.append(samplesPerSecondBenchmark)
        callbacks.append(gpuMetricsBenchmark)

    return callbacks


def main(args):
    """Main function."""
    dict_args = vars(args)
    wandb_logger = get_logger(dict_args)
    model_name = dict_args["model_name"]
    model_specific_arguments = get_model_arguments(model_name, parsed_arguments=dict_args)
    model = get_model(model_name=model_name, model_arguments=model_specific_arguments) # get the specified model
    callbacks = get_callbacks(dict_args)
    trainer = L.pytorch.Trainer(
        accelerator=dict_args["accelerator"],
        devices=dict_args["devices"],
        precision=dict_args["precision"],
        min_epochs=1,
        max_epochs=dict_args["max_epochs"],
        enable_checkpointing=dict_args["enable_checkpointing"],
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    data = ADNIDataModule(
        dataset=dict_args["dataset"],
        batch_size=dict_args["batch_size"],
        num_workers=dict_args["num_workers"],
        data_dir=dict_args["data_dir"],
        meta_file_path=dict_args["meta_file_path"],
        train_fraction=dict_args["train_fraction"],
        validation_fraction=dict_args["validation_fraction"],
        test_fraction=dict_args["test_fraction"],
    )

    
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

if __name__ == "__main__":
    # Set seed for reproducibility
    L.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # beginn with parsing arguments
    parser = ADNIParser()

    # figure out which model to use
    temp_args, _ = parser.parse_known_args()
    # let the model add what it needs
    if temp_args.model_name == "ResNet18":
        parser = LitADNIResNet.add_model_specific_args(parser)
    elif temp_args.model_name == "ShuffleNetV2":
        parser = LitADNIShuffleNetV2.add_model_specific_args(parser)
        
    # add modelname to checkpoint path
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=os.path.join(DEFAULTS["CHECKPOINTING"]["checkpoint_path_without_model_name"], temp_args.wandb_project+"/", temp_args.model_name)
    )        
    args = parser.parse_args()

    # start the training with the parsed arguments
    main(args)
