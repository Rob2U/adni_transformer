import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from monai.networks.nets import ViT
# from src.defaults import MODEL_DEFAULTS
from defaults import MODEL_DEFAULTS
from torchmetrics import AUROC, Accuracy, F1Score

class ADNIViT(nn.Module):
    def __init__(self, **model_aruments):
        
        super().__init__()
        self.vit = ViT(in_channels=1, img_size=(128,128,128), patch_size=(16,16,16), pos_embed='perceptron', classification=True, num_classes=2, post_activation=False)
    
    def forward(self, x):
        return self.vit(x)[0]
    
class LitADNIViT(L.LightningModule):
    """A lit Model."""

    def __init__(self, **model_arguments):
        super().__init__()
        self.model = ADNIViT(**model_arguments)
        self.learning_rate = model_arguments["learning_rate"]
        self.save_hyperparameters()
        self.iteration_preds = torch.Tensor([], device="cpu")
        self.iteration_labels = torch.Tensor([], device="cpu")
        # see https://lightning.ai/docs/pytorch/1.6.3/common/hyperparameters.html

    @staticmethod  # register new arguments here
    def add_model_specific_args(parent_parser):
        """Adds model-specific arguments to the parser."""

        parser = parent_parser.add_argument_group("LitADNIViT")
        return parent_parser

    def forward(self, x): 
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        

        preds = self.forward(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        
        if mode == "val":
            labels = labels.cpu()
            preds = preds.cpu()
            self.iteration_labels = torch.cat((self.iteration_labels, labels), dim=0)
            self.iteration_preds = torch.cat((self.iteration_preds, preds), dim=0)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")
    
    def on_validation_epoch_end(self):
        y_true = self.iteration_labels.long()
        y_pred = self.iteration_preds
        y_pred = F.softmax(y_pred, dim=1)  # get the probability of the positive class
        
        self.iteration_labels = torch.Tensor([], device="cpu")
        self.iteration_preds = torch.Tensor([], device="cpu")

        acc = Accuracy(task="multiclass", num_classes=2)(y_pred, y_true)
        roc = AUROC(task="multiclass", num_classes=2)(y_pred, y_true)
        f1 = F1Score(task="multiclass", num_classes=2)(y_pred, y_true)

        self.log('valid_acc', acc)
        self.log('valid_roc', roc)
        self.log('valid_f1', f1)
    

    
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