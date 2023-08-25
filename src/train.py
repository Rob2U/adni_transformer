""" Train a model. """

from datetime import datetime
from argparse import ArgumentParser
import os
import lightning as L
import torch
from time import gmtime, strftime
from pytorch_lightning.loggers import WandbLogger
# from benchmarks import SamplesPerSecondBenchmark, GpuMetricsBenchmark

from dataset import ADNIDataModule

from models.resnet import LitADNIResNet
from models.shufflenetV2 import LitADNIShuffleNetV2
from models.vit import LitADNIViT
from models.m3t import LitADNIM3T

from pretraining import SimCLRFrame, BYOLFrame, ResNetBackbone, ViTBackbone, ShuffleNetBackbone

from mlparser import ADNIParser
from defaults import DEFAULTS, MODEL_DEFAULTS

def get_model_class(model_name):
    """Returns the model class for a given model name."""
    
    if model_name == "ResNet18":
        return LitADNIResNet
    elif model_name == "ShuffleNetV2":
        return LitADNIShuffleNetV2
    elif model_name == "ViT":
        return LitADNIViT
    elif model_name == "M3T":
        return LitADNIM3T
    elif model_name == "BYOL": 
        return BYOLFrame
    elif model_name == "SimCLR":
        return SimCLRFrame
    else:
        raise ValueError(f"Model {model_name} not implemented.")
    
def get_backbone_class(model_name):
    """ Returns the backbone class for a given model name."""
    
    if model_name=="ResNet":
        return ResNetBackbone
    elif model_name=="ShuffleNetV2":
        return ShuffleNetBackbone
    elif model_name=="ViT":
        return ViTBackbone
    else:
        raise ValueError(f"Backbone for model {model_name} not implemented.")

def get_backbone_out_dim(model_name):
    if model_name == "ResNet":
        return 512
    elif model_name == "ShuffleNetV2":
        return 1024
    elif model_name == "ViT":
        return 1024
    else:
        raise ValueError(f"Backbone for model {model_name} not implemented.")
        

def load_pretrained_model(pretrained_path, model_class):
    """Loads a pretrained model from a checkpoint file."""
    print(f"Loading pretrained model from {pretrained_path} ...")
    # Automatically loads the model with the saved hyperparameters
    model = model_class.load_from_checkpoint(pretrained_path)
    return model

def get_model_arguments(model_name, parsed_arguments):
    model_args = {key: parsed_arguments[key] for key in parsed_arguments & MODEL_DEFAULTS[model_name].keys()}
    model_args["accelerator"] = parsed_arguments["accelerator"]
    # elif model_name ==
    model_args["learning_rate"] = parsed_arguments["learning_rate"]
    
    if model_name == "SimCLR" or model_name == "BYOL":
        if not parsed_arguments["backbone"]:
            raise ValueError("Backbone not specified for pretraining method.")

        model_args["backbone"] = get_backbone_class(parsed_arguments["backbone"])
        model_args["backbone_out_dim"] = get_backbone_out_dim(parsed_arguments["backbone"])
        model_args["hidden_dim_proj_head"] = parsed_arguments["hidden_dim_proj_head"]
        model_args["output_dim_proj_head"] = parsed_arguments["output_dim_proj_head"]
        model_args["max_epochs"] = parsed_arguments["max_epochs"]
    
    return model_args

def get_model(model_class, model_arguments):
    """instantiates a model with the given arguments."""
    model = model_class(model_arguments)
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
            monitor="train_loss",
            dirpath=os.path.join(arguments["checkpoint_path"]),
            filename=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "-{epoch:02d}-{val_loss:.2f}",
            save_top_k=5,
            mode="min",
        )
    # samplesPerSecondBenchmark =  SamplesPerSecondBenchmark()
    # gpuMetricsBenchmark = GpuMetricsBenchmark()

    callbacks = [
        checkpoint_callback,
    ]
    # add benchmarking specific callbacks only if neccessary
    # if(arguments["benchmark"]):
    #     callbacks.append(samplesPerSecondBenchmark)
    #     callbacks.append(gpuMetricsBenchmark)

    return callbacks


def main(args):
    """Main function."""
    dict_args = vars(args)
    wandb_logger = get_logger(dict_args)
    model_name = dict_args["model_name"]
    model_class = get_model_class(model_name)
    model_specific_arguments = get_model_arguments(model_name, parsed_arguments=dict_args)
    model = get_model(model_class=model_class, model_arguments=model_specific_arguments) # get the specified model
    
    # add the backbone output dimension to the arguments
    dict_args["backbone_out_dim"] = get_backbone_out_dim(dict_args["backbone"])
    
    if dict_args["compile"]:
        model = torch.compile(model)
    callbacks = get_callbacks(dict_args)
    
    trainer = L.pytorch.Trainer(
        accelerator=dict_args["accelerator"],
        devices=dict_args["devices"],
        precision=dict_args["precision"],
        min_epochs=1,
        max_epochs=dict_args["max_epochs"],
        enable_checkpointing=dict_args["enable_checkpointing"],
        #num_sanity_val_steps=0,
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
        model = load_pretrained_model(dict_args["pretrained_path"], model_name)
    else:
        trainer.fit(model, data)

        # load best checkpoint after training
        model = model_class.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
    
    # results = trainer.test(model, data)
    # print(results)

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
    elif temp_args.model_name == "ViT":
        parser = LitADNIViT.add_model_specific_args(parser)
    elif temp_args.model_name == "M3T":
        parser = LitADNIM3T.add_model_specific_args(parser)
        
    # add modelname to checkpoint path
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=os.path.join(DEFAULTS["CHECKPOINTING"]["checkpoint_path_without_model_name"], temp_args.wandb_project+"/", temp_args.model_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )        
    args = parser.parse_args()

    # start the training with the parsed arguments
    main(args)
