import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, F1Score
from monai.networks.nets import resnet18
from config import LEARNING_RATE
from models import summary

class ADNIResNet(nn.Module):
    def __init__(self, **kwargs):
        
        super().__init__()
        resnet = resnet18(pretrained=False, n_input_channels=1, num_classes=2, spatial_dims=3)
        #resnet.bn1 = torch.nn.Identity()
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.model.add_module("flatten", nn.Flatten())
        self.model.add_module("linear", list(resnet.children())[-1])
        self.model.to(kwargs["accelerator"])
        self.model.bn1 = torch.nn.Identity()
        #summary.summary(self.model, (1, 128, 128, 128), batch_size=32)
        #
        # ("Number of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
    

    def forward(self, x):
        return self.model(x)
    
class LitADNIResNet(L.LightningModule):
    """A lit Model."""

    def __init__(self, learning_rate, **kwargs):
        super().__init__()
        #self.device = kwargs["accelerator"]
        self.model = ADNIResNet(**kwargs)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.iteration_preds = torch.Tensor([], device="cpu")
        self.iteration_labels = torch.Tensor([], device="cpu")
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
        
        #optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
    
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        imgs = imgs
        labels = labels.cpu()
        
        preds = self.forward(imgs).cpu()
        loss = F.cross_entropy(preds, labels)
        """
        # goal is to take auc as indicator for performance in validation
        preds = F.softmax(self.forward(imgs), dim=1)
        auc = self._calculate_auc(imgs, labels, mode=mode)
        """
        
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        

        self.log(f"{mode}_loss", loss, prog_bar=True)
        self.log(f"{mode}_acc", acc, prog_bar=True)
        
        self.iteration_labels = torch.cat((self.iteration_labels, labels), dim=0)
        self.iteration_preds = torch.cat((self.iteration_preds, preds), dim=0)
        
        return loss
    
    def on_train_epoch_end(self):
        self.iteration_labels = torch.Tensor([], device="cpu")
        self.iteration_preds = torch.Tensor([], device="cpu")
    
    def on_test_epoch_end(self):
        self.iteration_labels = torch.Tensor([], device="cpu")
        self.iteration_preds = torch.Tensor([], device="cpu")
    
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

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="test")