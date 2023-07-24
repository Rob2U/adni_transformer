import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

import lightning as L

from torchmetrics import Accuracy, AUROC, F1Score
from torch import optim
import torch.nn.functional as F

from . import transforms


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

def getSIMCLRLoss():
    return NTXentLoss()
    


# build a LightningModule for SimCLR that takes a backbone as input
class SimCLRFrame(L.LightningModule):
    def __init__(self, model_arguments):
        super().__init__()
        self.model = SimCLR(model_arguments)
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()
        self.iteration_preds = torch.Tensor([], device="cpu")
        self.iteration_labels = torch.Tensor([], device="cpu")
        self.criterion = getSIMCLRLoss().to(model_arguments["ACCELERATOR"])
        self.transforms = transforms.get_train_tfms()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""
        parser = parent_parser.add_argument_group("LitSimCLR")
        # parser.add_argument("--width_mult", type=float, default=MODEL_DEFAULTS["SimCLR"]["width_mult"], help="provides width multiplier for the model")
        return parent_parser 
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        # x is a batch of images
        x_0 = self.transforms(batch) 
        x_1 = self.transforms(batch)
        
        x_0 = x_0.to(self.device)
        x_1 = x_1.to(self.device)
        
        z_0 = self.model(x_0)
        z_1 = self.model(x_1)
        
        loss = self.criterion(z_0, z_1)
        
        self.log(f"{mode}_loss", loss.item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")
