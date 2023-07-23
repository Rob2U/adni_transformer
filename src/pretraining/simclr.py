import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform


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

def performSimCLRIter(simclr_model, optimizer, x_0, x_1, device):
    if not performSimCLRIter.criterion:
        performSimCLRIter.criterion = getSIMCLRLoss()
    performSimCLRIter.criterion.to(device)
    criterion = performSimCLRIter.criterion
    
    x_0 = x_0.to(device)
    x_1 = x_1.to(device)
    
    z_0 = simclr_model(x_0)
    z_1 = simclr_model(x_1)
    
    loss = criterion(z_0, z_1)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()   
