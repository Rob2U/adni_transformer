import torch
from torch import nn

from src.models.shufflenetV2 import ADNIShuffleNetV2
from src.models.vit import ADNIViT, ViT
from src.models.resnet import ADNIResNet
from src.models.summary import summary

# Modify all models in order to be the backbone for the pretraining methods
class ShuffleNetBackbone(ADNIShuffleNetV2):
    def __init__(self, model_arguments, sample_size=128, num_classes=2):
        super().__init__(model_arguments, sample_size=sample_size, num_classes=num_classes)
        self.classifier = nn.Identity()
        # outputs shape of 1024
        
class ViTBackbone(ADNIViT):
    def __init__(self, model_aruments):
        super().__init__(model_aruments)
        
        self.vit = ViT(in_channels=1, img_size=(128,128,128), patch_size=(16,16,16), pos_embed='perceptron', classification=False)
        # output shape of 512x768 -> 768 is the embedding size of ViT Base (may consider using ViT Large)
    
    def forward(self, x):
        return self.vit(x)[0] # only return the generated cls token (shape of 768)
        

class ResNetBackbone(ADNIResNet):
    def __init__(self, model_args):
        super().__init__(model_args)
        self.model.linear = nn.Identity()
        # outputs shape of 512
        
if __name__ == "__main__":
    # Test the models
    # dict_args = {"width_mult": 1.0,}
    # sn_bb = ShuffleNetBackbone(dict_args).to("cpu")
    # summary(sn_bb, (1, 128, 128, 128), batch_size=32, device="cpu")
    # print(sn_bb)
    
    # dict_args = {}
    # vit_bb = ViTBackbone(dict_args).to("cpu")
    # test_input = torch.randn(1, 1, 128, 128, 128)
    # print(vit_bb(test_input).shape)
    # print(vit_bb)
    
    dict_args = {"accelerator": "cpu"}
    rn_bb = ResNetBackbone(dict_args).to("cpu")
    summary(rn_bb, (1, 128, 128, 128), batch_size=32, device="cpu")
    print(rn_bb)