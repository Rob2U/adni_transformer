"""Contains a base class for the trainer."""

import os
from config import TRAINER_CONFIG
import lightning as L


class MyTrainer(L.Trainer):
    """A base class for the trainer."""

    def __init__(self, logger, callbacks, **kwargs):
        """Initializes the trainer.

        Args:
            logger (pytorch_lightning.loggers.base.LightningLoggerBase): logger
            callbacks (list): list of callbacks
            **kwargs: keyword arguments
        """
        super().__init__(
            default_root_dir=os.path.join(kwargs["checkpoint_path"], kwargs["root"]),
            accelerator=kwargs["accelerator"],
            devices=kwargs["devices"],
            min_epochs=kwargs["min_epochs"],
            max_epochs=kwargs["max_epochs"],
            enable_checkpointing=kwargs["enable_checkpointing"],
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=0, # num_sanity_val_steps=0 is ok because dataloading is tested
        )
        

    @staticmethod
    def add_trainer_args(parent_parser):
        """Adds relevant arguments (for this particular trainer) to the parser.

        Args:
            parent_parser (argparse.ArgumentParser): the parser

        Returns:
            argparse.ArgumentParser: the parser with the added arguments
        """
        parser = parent_parser.add_argument_group("MyTrainer")
        parser.add_argument("--root", type=str, default="3DMLP")  #What is this for?
        parser.add_argument(
            "--accelerator", type=str, default=TRAINER_CONFIG["accelerator"], help="accelerator"
        )
        parser.add_argument(
            "--devices", type=int, default=TRAINER_CONFIG["devices"], help="number of devices"
        )
        parser.add_argument("--min_epochs", type=int, default=TRAINER_CONFIG["min_epochs"])
        parser.add_argument("--max_epochs", type=int, default=TRAINER_CONFIG["max_epochs"])
        parser.add_argument("--enable_checkpointing", type=bool, default=True)
        parser.add_argument("--log_model", type=bool, default=False)
        return parent_parser
