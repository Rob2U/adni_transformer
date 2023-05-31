""" dataset module """

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from config import BATCH_SIZE, NUM_WORKERS, DATA_DIR, META_FILE_PATH, TRAIN_FRACTION, VALIDATION_FRACTION, TEST_FRACTION

import os
import numpy as np
import pandas as pd

class ADNIDataset(Dataset):
    def __init__(self, data_dir, meta_file, transform=None, split='train'):
        super().__init__()
        self.classes = ['CN', 'AD']
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        self.data = pd.read_csv(meta_file) # first of all load metadata
        self.preprocess_metadata() # then preprocess it
        
        
    def __len__(self):
        return len(self.data['DX'])
        
    
    def __getitem__(self, index):
        lbl, image_uid = self.data.iloc[index][['DX', 'IMAGEUID']]
        
        lbl = 0 if lbl == 'CN' else 1
        lbl = torch.tensor(lbl)
        
        img = torch.tensor(self.load_image(image_uid))
        img = (img - img.min()) / (img.max() - img.min()) # normalize
        img = img[None, ...]  # add channel dim
        
        if self.transform:
            return self.transform(img), lbl
        else:
            return img, lbl

    def load_image(self, image_uid):
        file = 'file_' + image_uid + '.npy.npz'
        data_from_file = np.load(os.path.join(self.data_dir, file))
        return data_from_file['arr_0']
    
    def perform_split(self):
        np.random.seed(42)
        patients = self.data['PTID'].unique()
        patients = np.random.permutation(patients)

        if self.split == 'train':
            patients = patients[:int(patients.shape[0] * TRAIN_FRACTION)]
        elif self.split == 'val':
            patients = patients[int(patients.shape[0] * TRAIN_FRACTION):int(patients.shape[0] * (TRAIN_FRACTION + VALIDATION_FRACTION))]
        elif self.split == 'test':
            patients = patients[int(patients.shape[0] * (TRAIN_FRACTION + VALIDATION_FRACTION)):]
        else:
            raise ValueError('split must be one of train, val, test')
            
        self.data = self.data[self.data.PTID.isin(patients)]
    
    def preprocess_metadata(self):
        npy_files = os.listdir(self.data_dir)
        img_uid = [file_name.split('.')[0].split('_')[-1] for file_name in npy_files] # all image uids
        self.data = self.data[self.data.IMAGEUID.isin(img_uid)]

        self.data = self.data[self.data.DX.isin(self.classes)] # filter out MCI class
        self.perform_split()
        
        
class ADNIDatasetRAM(Dataset):
    def __init__(self, data_dir, meta_file, transform=None, split='train'):
        super().__init__()
        self.classes = ['CN', 'AD']
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        self.data = pd.read_csv(meta_file) # first of all load metadata
        self.preprocess_metadata() # then preprocess it
        self.load_data() # then load the data from the .npy.npz files and reduce to relevant columns
        # drop everything else
        
        
    def __len__(self):
        return len(self.data['DX'])
        
    def __getitem__(self, index):
        lbl, img = self.data.iloc[index][['DX','data']]
        
        lbl = 0 if lbl == 'CN' else 1
        lbl = torch.tensor(lbl)
        
        img = torch.tensor(img)
        img = (img - img.min()) / (img.max() - img.min()) # normalize
        img = img[None, ...]  # add channel dim
        
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
    
    def perform_split(self):
        np.random.seed(42)
        patients = self.data['PTID'].unique()
        patients = np.random.permutation(patients)

        if self.split == 'train':
            patients = patients[:int(patients.shape[0] * TRAIN_FRACTION)]
        elif self.split == 'val':
            patients = patients[int(patients.shape[0] * TRAIN_FRACTION):int(patients.shape[0] * (TRAIN_FRACTION + VALIDATION_FRACTION))]
        elif self.split == 'test':
            patients = patients[int(patients.shape[0] * (TRAIN_FRACTION + VALIDATION_FRACTION)):]
        else:
            raise ValueError('split must be one of train, val, test')
            
        self.data = self.data[self.data.PTID.isin(patients)]
    
    def preprocess_metadata(self):
        npy_files = os.listdir(self.data_dir)
        img_uid = [file_name.split('.')[0].split('_')[-1] for file_name in npy_files] # all image uids
        self.data = self.data[self.data.IMAGEUID.isin(img_uid)]

        self.data = self.data[self.data.DX.isin(self.classes)] # filter out MCI class
        self.perform_split()
        
    
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
        
        self.train_ds = self.dataset(self.data_dir, self.meta_file_path, train_transform, split='train')
        self.val_ds = self.dataset(self.data_dir, self.meta_file_path, test_transform, 'val')
        self.test_ds = self.dataset(self.data_dir, self.meta_file_path, test_transform, 'test')
            
        
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
