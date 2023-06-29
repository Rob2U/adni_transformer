import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from monai.networks.nets import ViT
from defaults import MODEL_DEFAULTS

class ADNIViT(nn.Module):
    def __init__(self, model_aruments):
        
        super().__init__()
        self.vit = ViT(in_channels=1, img_size=(128,128,128), patch_size=(8,8,8), pos_embed='conv', classification=True, num_classes=2, post_activation=False)
        print(self.vit)
    
    def forward(self, x):
        return self.vit(x)[0]
    
class LitADNIViT(L.LightningModule):
    """A lit Model."""

    def __init__(self, model_arguments):
        super().__init__()
        self.model = ADNIViT(model_arguments)
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitADNIViT")
        return parent_parser

    def forward(self, x): 
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        img, label = batch
        preds = self.forward(img)
        loss = F.cross_entropy(preds, label)
        acc = (preds.argmax(dim=-1) == label).float().mean()

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")

    
if __name__ == "__main__":
    with torch.no_grad():
        model = ADNIViT()
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
        print(model)

        input_var = Variable(torch.randn(8, 1, 128, 128, 128))
        output = model(input_var)
        print(output)
        print(output.shape)