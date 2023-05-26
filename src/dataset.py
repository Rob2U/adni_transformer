""" dataset module """

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from config import BATCH_SIZE, NUM_WORKERS, DATA_DIR

import os
import numpy as np

#TODO: add missing labels to the dataset
class ADNIDataset(Dataset):
    def __init__(self, root, transform, train=false, download=false):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.data = self.load_data()
        
        assert(train == False and download == False)
        
    def __len__(self):
        len(self.data)
    
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        
        self.data[index]

    def load_data(self):
        # load every .npy.npz file in the root folder into list
        for file in os.listdir(self.root):
            data_from_file = np.load(os.path.join(self.root, file))
            data.append(data_from_file)
        # convert list of npz files into a tensor
        data = [entry["arr_0"] for entry in data]
        data = np.array(data)
        data = torch.from_numpy(data)
        
        return data

# TODO: test
class ADNIDataModule(L.LightningDataModule):
    """ADNI datamodule"""

    def __init__(self, data_dir, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = ADNIDataset

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-specific arguments to the parser."""
        parser = parent_parser.add_argument_group("ADNIDataModule")
        parser.add_argument("--data_dir", type=str, default=DATA_DIR)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        return parent_parser

    # execute on every GPU
    def prepare_data(self):
        self.dataset(self.data_dir, train=False, transform=None)

    # execute on every GPU
    #TODO: test
    def setup(self, stage):
        test_transform = None
        train_transform = None
        
        # Loading the training dataset. We need to split it into a training and validation part
        # We need to do a little trick because the validation set should not use the augmentation.
        dataset = self.dataset(root=self.data_dir)
        
        len_train = len(dataset) * 10 // 8 # 80% of the dataset
        len_val = len_train * 1 // 10 # 10% of the training set
        
        L.seed_everything(42)
        train_val_ds, self.test_ds = torch.utils.data.random_split(train_dataset, [len_train, len(dataset) - len_train])
        self.train_ds, self.val_ds = torch.utils.data.random_split(train_val_ds, [len_train - len_val, len_val])
        

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
