from pathlib import Path
from models.shufflenetv2 import ShuffleNetV2

import pytorch_lightning as pl
import torch
import wandb
from monai.networks.nets import resnet18
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC, F1Score
from utils import check_configs, get_wandb_config
import numpy as np
import os

from data.adni_dataset import ADNIDataset, get_train_tfms, get_test_tfms


PROJECT_NAME = 'test'

config = dict()
config['DEBUG'] = False
config['WB_LOGGER'] = True
config['checkpoint_dir'] = '/dhc/home/arkadiusz.kwasigroch/experiments/pretraining_ssl/noble-frost-9'

config['lr'] = 1e-3
config['optimizer'] = 'adam'

config['batch_size'] = 40
config['EPOCHS'] = 60
config['TRAIN_FRACTION'] = 100

config['IMAGING_MODEL_NAME'] = "shufflenet"  # can be "resnet18", "unet"

config['ACCUMULATE_GRAD_BATCHES'] = 1
config['EVAL_EVERY_N_EPOCHS'] = 1
config['PRECISION'] = 32
config['GRADIENT_CLIP_VAL'] = 0
config['freeze_backbone'] = False
config['SCHEDULER'] = True
config['SEED'] = 42
config['JOB_ID'] = os.getenv('SLURM_JOB_ID')


class AdniModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if config['IMAGING_MODEL_NAME'] == 'resnet18':
            resnet = resnet18(pretrained=False, n_input_channels=1)
            resnet.bn1 = torch.nn.Identity()
            self.backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
            nb_features = 512

        elif config['IMAGING_MODEL_NAME'] == 'shufflenet':
            net = ShuffleNetV2(width_mult=1., image_channels=1)
            self.backbone = net
            nb_features = 1024

        if config['ssl_pretrained']:
            checkopoint_path = list(Path(config['checkpoint_dir']).glob('*.ckpt'))[0]
            state_dict = torch.load(checkopoint_path)['state_dict']
            for key in list(state_dict.keys()):
                if 'image_backbone.' in key:
                    state_dict[key.replace('image_backbone.', '')] = state_dict.pop(key)
                if 'ssl_model.backbone.' in key:
                    state_dict[key.replace('ssl_model.backbone.', '')] = state_dict.pop(key)
            if config['IMAGING_MODEL_NAME'] == 'resnet18':
                state_dict['0.weight'] = state_dict['0.weight'][:, 0:1, :, :, :]
            elif config['IMAGING_MODEL_NAME'] == 'shufflenet':
                state_dict['conv1.0.weight'] = state_dict['conv1.0.weight'][:, 0:1, :, :, :]
            self.backbone.load_state_dict(state_dict, strict=False)

        if config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        train_tfms = get_train_tfms()
        test_tfms = get_test_tfms()
        self.train_dataset = ADNIDataset(tfms=train_tfms, train_fraction=config['TRAIN_FRACTION'], split='train')
        self.valid_dataset, self.test_dataset = [ADNIDataset(tfms=test_tfms, split=split) for split in ['val', 'test']]

        nb_classes = len(self.train_dataset.classes)

        self.fc = torch.nn.Linear(nb_features, nb_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        if config['WB_LOGGER']:
            wandb.define_metric('valid_acc', summary='max')
            wandb.define_metric('valid_roc', summary='max')
            wandb.define_metric('valid_f1', summary='max')

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        y_hat = self.forward(img)
        loss = self.loss(y_hat.squeeze(), label)

        if loss.isnan() or loss.isinf():
            print('NaN value')
            return None
        else:
            self.log("train_loss", loss)
            return loss

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        y_hat = self.forward(img)
        loss = self.loss(y_hat.squeeze(), label)
        self.log('valid_loss', loss)

        return {'label': label, 'pred': y_hat.squeeze()}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([output['label'] for output in outputs], dim=0)
        y_pred = torch.cat([output['pred'] for output in outputs], dim=0)
        y_pred = torch.nn.Softmax(dim=1)(y_pred)[:, 1]  # get the probability of the positive class

        acc = Accuracy()(y_pred.cpu(), y_true.cpu())
        roc = AUROC()(y_pred.cpu(), y_true.cpu())
        f1 = F1Score()(y_pred.cpu(), y_true.cpu())

        self.log('valid_acc', acc)
        self.log('valid_roc', roc)
        self.log('valid_f1', f1)

    def test_step(self, batch, batch_idx):
        img, label, _ = batch
        y_hat = self.forward(img)
        return {'label': label, 'pred': y_hat.squeeze()}

    def test_epoch_end(self, outputs):
        y_true = torch.cat([output['label'] for output in outputs], dim=0)
        y_pred = torch.cat([output['pred'] for output in outputs], dim=0)
        y_pred = torch.nn.Softmax(dim=1)(y_pred)[:, 1]  # get the probability of the positive class

        acc = Accuracy()(y_pred.cpu(), y_true.cpu())
        weight_acc = Accuracy(average='weighted', num_classes=2, multiclass=True)(y_pred.cpu(), y_true.cpu())
        roc = AUROC()(y_pred.cpu(), y_true.cpu())
        f1 = F1Score()(y_pred.cpu(), y_true.cpu())

        self.log('test_acc', acc)
        self.log('test_weight_acc', weight_acc)
        self.log('test_roc', roc)
        self.log('test_f1', f1)

        np.save(Path(self.test_output) / 'y_true.npy', y_true.cpu().numpy())
        np.save(Path(self.test_output) / 'y_pred.npy', y_pred.cpu().numpy())

    def configure_optimizers(self):
        if config['freeze_backbone']:
            if config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(self.fc.parameters(), lr=config['lr'])
            elif config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(self.fc.parameters(), lr=config['lr'], momentum=0.9, nesterov=True)
        else:
            if config['optimizer'] == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'])
            elif config['optimizer'] == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9, nesterov=True)

        if config['SCHEDULER']:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['EPOCHS'], eta_min=0, last_epoch=-1
            )
            return [optimizer], [scheduler]
        return optimizer

    def train_dataloader(self):
        loader = DataLoader(dataset=self.train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=16)
        return loader

    def val_dataloader(self):
        loader = DataLoader(dataset=self.valid_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)
        return loader

    def test_dataloader(self):
        loader = DataLoader(dataset=self.test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=16)
        return loader


if __name__ == '__main__':

    if config['WB_LOGGER']:
        logger_callback = WandbLogger()
        wandb_log_path = '/dhc/home/arkadiusz.kwasigroch/experiments'
        wandb.init(project=PROJECT_NAME, config=config, dir=wandb_log_path, allow_val_change=True)
        check_configs(config, wandb.config)

        ssl_pretrained = True if wandb.config.checkpoint_dir else False
        wandb.config.update({'ssl_pretrained': ssl_pretrained})

        config = wandb.config
        run_id = wandb.run.id


        if config['checkpoint_dir']:
            pretraining_config = get_wandb_config(config['checkpoint_dir'])
            if pretraining_config is not None:
                config.update({'IMAGING_MODEL_NAME': pretraining_config['IMAGING_MODEL_NAME']}, allow_val_change=True)


        # update project name, in case there is different project name in a sweep
        PROJECT_NAME = wandb.run.project

        checpoints_path = Path('/dhc/home/arkadiusz.kwasigroch/experiments')
        experiment_checpoint_path = checpoints_path / PROJECT_NAME / run_id
    else:
        logger_callback = True
        experiment_checpoint_path = None
        config['ssl_pretrained'] = True if config['checkpoint_dir'] else False
        if config['checkpoint_dir']:
            pretraining_config = get_wandb_config(config['checkpoint_dir'])
            if pretraining_config is not None:
                config['IMAGING_MODEL_NAME'] = pretraining_config['IMAGING_MODEL_NAME']

    pl.seed_everything(config['SEED'], workers=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_checpoint_path,
        filename="model-{epoch:02d}-{valid_loss:.2f}",
        mode='max',
        monitor='valid_acc'
    )

    lr_monitor_callback = LearningRateMonitor()
    check_nan_callback = EarlyStopping(monitor='valid_loss', check_finite=True, verbose=True, patience=10, mode='min')

    model = AdniModule()
    trainer = pl.Trainer(
        max_epochs=config['EPOCHS'],
        deterministic=False,
        gpus=1,
        accumulate_grad_batches=config['ACCUMULATE_GRAD_BATCHES'],
        check_val_every_n_epoch=config['EVAL_EVERY_N_EPOCHS'],
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, lr_monitor_callback, check_nan_callback],
        precision=config['PRECISION'],
        logger=logger_callback,
        gradient_clip_val=config['GRADIENT_CLIP_VAL'],
        log_every_n_steps=20,
    )
    trainer.fit(model)
    print("Finished Training")
    #
    # print("Testing the model on the test split...")
    # result = trainer.test(model)
    # print(result)
    print("Done.")