import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from monai.networks.nets import resnet18
from config import LEARNING_RATE

class ADNIResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        resnet = resnet18(pretrained=False, n_input_channels=1)
        resnet.bn1 = torch.nn.Identity()
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.add_module("fc", nn.Linear(512, 2))  # number of features is 512, number of classes is 2s

    def forward(self, x):
        return self.model(x)
    
class LitADNIResNet(L.LightningModule):
    """A lit Model."""

    def __init__(self, learning_rate, **kwargs):
        super().__init__()
        self.model = ADNIResNet(**kwargs).model
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitADNIResNet")
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
        self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    
