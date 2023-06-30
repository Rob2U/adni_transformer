import torch
from models import summary
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

class zweiDCNNPretrained(nn.Module): # TODO
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights,progress=True)
        #self.resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-2])) # cut off adaptive average pooling and linear layer
        self.transforms = weights.transforms()
        summary.summary(self.resnet18, (8, 10, 3*128, 128, 128), batch_size=32)
        
        ("Number of parameters: ", sum(p.numel() for p in self.resnet18.parameters() if p.requires_grad))

    def forward(self, x):
        with torch.no_grad():
            x = self.transforms(x)
            x = self.resnet18(x)
        return x
    
if __name__ == "__main__":
    resnet = zweiDCNNPretrained()