import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from monai.networks.nets import ViT
from defaults import MODEL_DEFAULTS
from torchmetrics import AUROC, Accuracy, F1Score
from torchvision.models import resnet18, ResNet18_Weights



# temporary parameter collection
IN_DIM = 128
# dreiDCNN
IN_CHANNELS = 1
C3D = 32 # channel number of the 3D CNN
FILTER_SHAPE_3D_CNN = (5,5,5)

# SplitModule
C2D_split = 128 # IDK
LINEAR_HIDDEN_NEURONS_split = 512
TOKEN_D = 256

# Transformer Parameters
TRANSFORMER_LAYERS = 8
TRANSFORMER_HIDDEN_SIZE = 768
TRANSFORMER_MLP_SIZE = 768
TRANSFORMER_ATTENTION_HEADS = 8


class ADNIM3T(nn.Module):
    def __init__(self, model_aruments):
        
        super().__init__()
        
    
    def forward(self, x):
        return x
    
class dreiDCNN(nn.Module):
    def __init__(self, model_aruments):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=IN_CHANNELS,
            out_channels=C3D,
            kernel_size=FILTER_SHAPE_3D_CNN,
            stride=1,
            padding="same"
        )
        self.bn1 = nn.BatchNorm3d(num_features=C3D)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=C3D,
            out_channels=C3D,
            kernel_size=FILTER_SHAPE_3D_CNN,
            stride=1,
            padding="same"
        )
        self.bn2 = nn.BatchNorm3d(num_features=C3D)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        conv_block_1_out = self.relu1(self.bn1(self.conv1(x)))
        conv_block_2_out = self.relu2(self.bn2(self.conv2(conv_block_1_out)))
        
        return conv_block_2_out

class SplitModule(nn.Module):
    def __init__(self, model_aruments):
        super().__init__()

    def split_coronal(self, x):
        coronal = torch.split(x, 1, dim=0)
        coronal = torch.cat(coronal)
        coronal = torch.reshape(coronal,(IN_DIM, IN_DIM, IN_DIM))
        return coronal

    def split_sagittal(self, x):
        sagittal = torch.split(x, 1, dim=1)
        sagittal = torch.cat(sagittal)
        sagittal = torch.reshape(sagittal,(IN_DIM, IN_DIM, IN_DIM))
        return sagittal

    def split_axial(self, x):
        axial = torch.split(x, 1, dim=2)
        axial = torch.cat(axial)
        axial = torch.reshape(axial,(IN_DIM, IN_DIM, IN_DIM))
        return axial

    def forward(self, x):
        return torch.cat((self.split_coronal(x), self.split_sagittal(x), self.split_axial(x)))
    

class ProjectionBlock(nn.Module):
    def __init__(self, model_aruments):
        super().__init__()

        self.zweiDCNN = zweiDCNNPretrained()
        self.projection = NonLinearProjection()
        self.embedding = PositionPlaneEmbedding()

    def forward(self, x):
        return self.embedding(self.projection(self.zweiDCNN(x)))

# ------ part of Projection Block
class zweiDCNNPretrained(nn.Module): # TODO
    def __init__(self, model_aruments):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        resnet18 = resnet18(weights=weights,progress=True)
        self.resnet18 = torch.nn.Sequential(*(list(resnet18.children())[:-2])) # cut off adaptive average pooling and linear layer
        self.transforms = weights.transforms()

    def forward(self, x):
        with torch.no_grad():
            x = self.transforms(x)
            x = self.resnet18(x)
        return x


class NonLinearProjection(nn.Module):
    def __init__(self, model_aruments):
        super().__init__()

        self.linear1 = nn.Linear(in_features=C2D_split, out_features=LINEAR_HIDDEN_NEURONS_split)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=LINEAR_HIDDEN_NEURONS_split, out_features=TOKEN_D)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear2(self.relu1(self.linear1(x))))


class PositionPlaneEmbedding(nn.Module):
    def __init__(self, model_aruments):
        super().__init__()

        self.class_tkn = nn.Parameter(torch.zeros(TOKEN_D))
        self.sep_tkn = nn.Parameter(torch.zeros(TOKEN_D))

        self.entry_amount_per_plane = IN_DIM # in paper: S
        self.sequence_length = 3 * self.entry_amount_per_plane + 4  # in paper just 3 * S + 4
        
        self.positional_embedding = nn.Parameter(torch.zeros((self.sequence_length, TOKEN_D)))

        self.coronal_plane_embedding = nn.Parameter(torch.zeros(TOKEN_D))
        self.saggital_plane_embedding = nn.Parameter(torch.zeros(TOKEN_D))
        self.axial_plane_embedding = nn.Parameter(torch.zeros(TOKEN_D))


    def forward(self, x):
        # construct correct sequence
        # x needs to have shape self.entry_amount_per_plane*3,d
        Z_0 = torch.cat([
            torch.tensor([self.class_tkn]), 
            x[0:self.entry_amount_per_plane] + self.coronal_plane_embedding, # broadcast plane embedding
            torch.tensor([self.sep_tkn]),
            x[self.entry_amount_per_plane:self.entry_amount_per_plane*2] + self.saggital_plane_embedding,
            torch.tensor([self.sep_tkn]),
            x[self.entry_amount_per_plane*2:self.entry_amount_per_plane*3] + self.axial_plane_embedding,
            torch.tensor([self.sep_tkn])
        ])
        return Z_0 + self.positional_embedding

# -------

class Transformer(nn.Module): # TODO
    def __init__(self, model_aruments):
        super().__init__()

        self.decoder = TransformerDecoder()
        self.encoder = TransformerEncoder()

    def forward(self, x):
        return self.encoder(self.decoder(x))

class TransformerDecoder(nn.Module): # TODO
    def __init__(self, model_aruments):
        super().__init__()

    def forward(self, x):
        return x


class TransformerEncoder(nn.Module): #TODO
    def __init__(self, model_aruments):
        super().__init__()
        

    def forward(self, x):
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_arguments):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=TOKEN_D, num_heads=TRANSFORMER_ATTENTION_HEADS)
        self.layer_norm2 = nn.LayerNorm()
        self.linear = nn.Linear(in_features=TRANSFORMER_MLP_SIZE, out_features=TRANSFORMER_HIDDEN_SIZE) # 

    def forward(self, x):
        normalized_x = self.layer_norm1(x)
        mha_x = self.multi_head_attention(normalized_x, normalized_x, normalized_x)
        normalized_mha_x = self.layer_norm2(mha_x)
        return self.linear(x)


    
class LitADNIM3T(L.LightningModule):
    """A lit Model."""

    def __init__(self, model_arguments):
        super().__init__()
        self.model = ADNIViT(model_arguments)
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