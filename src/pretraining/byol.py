import copy

import torch
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from torch import optim
import lightning as L


from . import transforms


class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


def getBYOLLoss():
    return NegativeCosineSimilarity()


class BYOLFrame(L.LightningModule):
    def __init__(self, model_arguments):
        super().__init__()
        self.model = BYOL(model_arguments)
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()
        self.iteration_preds = torch.Tensor([], device="cpu")
        self.iteration_labels = torch.Tensor([], device="cpu")
        self.criterion = getBYOLLoss().to(model_arguments["ACCELERATOR"])
        self.transforms = transforms.get_train_tfms()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""
        parser = parent_parser.add_argument_group("LitBYOL")
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
        momentum_val = cosine_schedule(self.current_epoch, self.hparams.max_epochs, 0.996, 1)
        
        x_0 = self.transforms(batch)
        x_1 = self.transforms(batch)        
        update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum_val)
        update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=momentum_val) 
        
        p_0 = self.model(x_0)
        z_0 = self.model.forward_momentum(x_0)
        p_1 = self.model(x_1)
        z_1 = self.model.forward_momentum(x_1)
        
        loss = 0.5 * (self.criterion(p_0, z_1) + self.criterion(p_1, z_0))
        
        self.log(f"{mode}_loss", loss.item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

