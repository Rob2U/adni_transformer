import argparse
from defaults import DEFAULTS


class ADNIParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.hyperparameters = self.add_argument_group("Hyperparameters")
        self.dataloading = self.add_argument_group("Dataloading")
        self.computation = self.add_argument_group("Computation")
        self.wandb = self.add_argument_group("Wandb")
        self.checkpointing = self.add_argument_group("Checkpointing")
        self.set_defaults()

    def set_hyperparamter_defaults(self):
        self.hyperparameters.add_argument("--model_name", type=str, default=DEFAULTS['HYPERPARAMETERS']['model'])
        self.hyperparameters.add_argument("--learning_rate", type=float, default=DEFAULTS['HYPERPARAMETERS']['learning_rate'])
        self.hyperparameters.add_argument("--batch_size", type=int, default=DEFAULTS['HYPERPARAMETERS']['batch_size'])
        self.hyperparameters.add_argument("--train_fraction", type=float, default=DEFAULTS['HYPERPARAMETERS']['train_fraction'])
        self.hyperparameters.add_argument("--validation_fraction", type=float, default=DEFAULTS['HYPERPARAMETERS']['validation_fraction'])
        self.hyperparameters.add_argument("--test_fraction", type=float, default=DEFAULTS['HYPERPARAMETERS']['test_fraction'])

    def set_dataloading_defaults(self):
        self.dataloading.add_argument("--dataset", type=str, default=DEFAULTS['DATALOADING']['dataset'])
        self.dataloading.add_argument("--data_dir", type=str, default=DEFAULTS['DATALOADING']['data_dir'])
        self.dataloading.add_argument("--meta_file_path", type=str, default=DEFAULTS['DATALOADING']['meta_file_path'])
        self.dataloading.add_argument("--num_workers", type=int, default=DEFAULTS['DATALOADING']['num_workers'])
    
    def set_computation_defaults(self):
        self.computation.add_argument("--accelerator", type=str, default=DEFAULTS['COMPUTATION']['accelerator'])
        self.computation.add_argument("--devices", type=int, default=DEFAULTS['COMPUTATION']['devices'])
        self.computation.add_argument("--max_epochs", type=int, default=DEFAULTS['COMPUTATION']['max_epochs'])

    def set_wandb_defaults(self):
        self.wandb.add_argument("--wandb_project", type=str, default=DEFAULTS['WANDB']['wandb_project'])

    def set_checkpointing_defaults(self):
        self.checkpointing.add_argument("--checkpoint_path_without_model_name", type=str, default=DEFAULTS['CHECKPOINTING']['checkpoint_path_without_model_name'])
        self.checkpointing.add_argument("--pretrained_path", type=str, default=DEFAULTS['CHECKPOINTING']['pretrained_path'])
        self.checkpointing.add_argument("--enable_checkpointing", type=bool, default=DEFAULTS['CHECKPOINTING']['enable_checkpointing'])
    
    def set_defaults(self):
        self.set_hyperparamter_defaults()
        self.set_dataloading_defaults()
        self.set_computation_defaults()
        self.set_wandb_defaults()
        self.set_checkpointing_defaults()