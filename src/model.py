""" A basic model. """

import lightning.pytorch as L
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, LEARNING_RATE, DROPOUT


class aModel(nn.Module):
    """A Model."""

    def __init__(
        self, **kwargs
    ):  #!you need to register new arguments for the parser!
        super().__init__()

    def forward(self, x):
        """Forward pass."""
        return x


class LitModel(L.LightningModule):
    """A lit Model."""

    def __init__(self, learning_rate=LEARNING_RATE, **kwargs):
        super().__init__()
        self.model = aModel()(**kwargs)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="provides learning rate for the optimizer")
        return parent_parser

    def forward(self, x, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")
