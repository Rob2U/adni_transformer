import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.utils.scheduler import cosine_schedule


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

# resnet = torchvision.models.resnet18()
# backbone = nn.Sequential(*list(resnet.children())[:-1])
# model = BYOL(backbone)

def getBYOLLoss():
    return NegativeCosineSimilarity()

# TODO: add momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
def performBYOLIter(byol_model, optimizer, x_0, x_1, momentum_val, device):
    if not performBYOLIter.criterion:
        performBYOLIter.criterion = getBYOLLoss()
    performBYOLIter.criterion.to(device)
    criterion = performBYOLIter.criterion
    
    x_0 = x_0.to(device)
    x_1 = x_1.to(device)
    
    update_momentum(byol_model.backbone, byol_model.backbone_momentum, m=momentum_val)
    update_momentum(byol_model.projection_head, byol_model.projection_head_momentum, m=momentum_val) 
    
    p_0 = byol_model(x_0)
    z_0 = byol_model.forward_momentum(x_0)
    p_1 = byol_model(x_1)
    z_1 = byol_model.forward_momentum(x_1)
    
    loss = 0.5 * (criterion(p_0, z_1) + criterion(p_1, z_0))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()
