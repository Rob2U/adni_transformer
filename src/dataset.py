""" dataset module """

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from config import BATCH_SIZE, NUM_WORKERS, DATA_DIR


class aDataset(Dataset):
    def __init__(self, root, train, download, transform):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download    
        
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def __download__(self):
        pass


class aDataModule(L.LightningDataModule):
    """MNIST3D dataset module"""

    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = MNIST

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-specific arguments to the parser."""
        parser = parent_parser.add_argument_group("MNIST3DModule")
        parser.add_argument("--data_dir", type=str, default=DATA_DIR)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        return parent_parser

    # execute only on 1 GPU
    def prepare_data(self):
        self.dataset(self.data_dir, train=True, download=Truem, transform=None)
        self.dataset(self.data_dir, train=False, download=True, transform=None)

    # execute on every GPU
    def setup(self, stage):
        test_transform = None
        train_transform = None
        
        # Loading the training dataset. We need to split it into a training and validation part
        # We need to do a little trick because the validation set should not use the augmentation.
        train_dataset = self.dataset(
            root=self.data_dir,
            train=True,
            transform=train_transform,
        )
        val_dataset = self.dataset(
            root=self.data_dir,
            train=True,
            transform=test_transform,
        )
        L.seed_everything(42)
        self.train_ds, _ = torch.utils.data.random_split(train_dataset, [55000, 5000])
        L.seed_everything(42)
        _, self.val_ds = torch.utils.data.random_split(val_dataset, [55000, 5000])

        self.test_ds = self.dataset(
            root=self.data_dir, train=False, transform=test_transform, download=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
