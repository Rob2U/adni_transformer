""" A basic MLP model for MNIST3D. """

import lightning.pytorch as L
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, LEARNING_RATE, DROPOUT


class BasicMLP(nn.Module):
    """A basic MLP."""

    def __init__(
        self, input_dim, hidden_dim, output_dim, dropout=0.0, **kwargs
    ):  #!you need to register new arguments for the parser!
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass."""

        x = self.linear_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        x = self.activation(x)
        x = F.softmax(x, dim=1)
        return x


class LitBasicMLP(L.LightningModule):
    """A basic MLP."""

    def __init__(self, learning_rate=LEARNING_RATE, **kwargs):
        super().__init__()
        self.model = BasicMLP(**kwargs)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitBasicMLP")
        parser.add_argument("--input_dim", type=int, default=INPUT_DIM)
        parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
        parser.add_argument("--output_dim", type=int, default=OUTPUT_DIM)
        parser.add_argument("--dropout", type=float, default=DROPOUT)
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=LEARNING_RATE,
            help="provides learning rate for the optimizer",
        )
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
