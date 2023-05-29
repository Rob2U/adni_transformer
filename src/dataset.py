""" dataset module """

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from config import BATCH_SIZE, NUM_WORKERS, DATA_DIR, META_FILE_PATH

import os
import numpy as np
import pandas as pd

# 
class ADNIDataset(Dataset):
    def __init__(self, data_dir, meta_file, transform=None):
        super().__init__()
        self.classes = ['CN', 'AD']
        self.data_dir = data_dir
        self.transform = transform
        
        self.data = pd.read_csv(meta_file) # first of all load metadata
        self.preprocess_metadata() # then preprocess it
        self.load_data() # then load the data from the .npy.npz files and reduce to relevant columns
        
    def __len__(self):
        return len(self.data['DX'])
        
    
    def __getitem__(self, index):
        lbl, img = self.data.iloc[index][['DX','data']]
        
        lbl = 0 if lbl == 'CN' else 1
        lbl = torch.tensor(lbl)
        
        img = torch.tensor(img)
        
        if self.transform:
            return self.transform(img), lbl
        else:
            return img, lbl

    def load_data_of_row(self, row):
        file = 'file_' + row['IMAGEUID'] + '.npy.npz'
        data_from_file = np.load(os.path.join(self.data_dir, file))
        row['data'] = data_from_file['arr_0']
        return row
    
    def load_data(self):
        # load every .npy.npz file in the root folder into dataframe
        self.data['data'] = None
        self.data = self.data.apply(lambda row: self.load_data_of_row(row), axis=1)
        self.data = self.data[['DX', 'data']]

    #TODO: try to allow multiple occurances of the same patient but split the patient into train, val, test
    def ensure_single_occurance_per_patient(self, seed=42):
        # randomly shuffle the dataframe then drop duplicates
        np.random.seed(seed)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data.drop_duplicates(subset=['PTID'], inplace=True)
    
    def preprocess_metadata(self):
        npy_files = os.listdir(self.data_dir)
        img_uid = [file_name.split('.')[0].split('_')[-1] for file_name in npy_files] # all image uids
        self.data = self.data[self.data.IMAGEUID.isin(img_uid)]

        self.data = self.data[self.data.DX.isin(self.classes)] # filter out MCI class
        self.ensure_single_occurance_per_patient()
        
    

# TODO: test
class ADNIDataModule(L.LightningDataModule):
    """ADNI datamodule"""

    def __init__(self, data_dir, meta_file_path, batch_size, num_workers, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.meta_file_path = meta_file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = ADNIDataset

    @staticmethod
    def add_data_specific_args(parent_parser):
        """Adds data-specific arguments to the parser."""
        parser = parent_parser.add_argument_group("ADNIDataModule")
        parser.add_argument("--data_dir", type=str, default=DATA_DIR)
        parser.add_argument("--meta_file_path", type=str, default=META_FILE_PATH)
        parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
        parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
        return parent_parser

    # execute on single GPU
    def prepare_data(self):
        pass

    # execute on every GPU
    def setup(self, stage):
        test_transform = None
        train_transform = None
        
        # Loading the training dataset. We need to split it into a training and validation part
        # We need to do a little trick because the validation set should not use the augmentation.
        dataset = self.dataset(data_dir=self.data_dir, meta_file=self.meta_file_path, transform=train_transform)
        
        len_train = len(dataset) * 8 // 10 # 80% of the dataset
        len_val = len_train * 1 // 10 # 10% of the training set
        
        L.seed_everything(42)
        train_val_ds, self.test_ds = torch.utils.data.random_split(dataset, [len_train, len(dataset) - len_train])
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
