""" dataset module """

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import monai

def get_tfms():
    transforms = monai.transforms.Compose([monai.transforms.RandScaleCrop([0.95, 0.95, 0.95]),
                                           monai.transforms.Resize([128, 128, 128]),
                                           monai.transforms.ToTensor(dtype=torch.float),

                                           ])
    transforms.set_random_state(42)
    return transforms

def get_train_tfms(seed=42):
    transforms = monai.transforms.Compose([
        monai.transforms.RandSpatialCrop( roi_size=(120, 120, 120), random_size=True),
        monai.transforms.Resize(spatial_size=(128, 128, 128)),
        monai.transforms.RandFlip(prob=0.5, spatial_axis=0),
        monai.transforms.RandFlip(prob=0.5, spatial_axis=1),
        monai.transforms.RandFlip(prob=0.5, spatial_axis=2),
        monai.transforms.RandAdjustContrast(prob=0.7, gamma=(0.5, 2.5)),
        monai.transforms.RandShiftIntensity(offsets=0.125, prob=0.7),
        monai.transforms.ToTensor(dtype=torch.float),
    ])
    transforms.set_random_state(seed)
    return transforms

def get_test_tfms():
    transforms = monai.transforms.Compose([ monai.transforms.ToTensor(dtype=torch.float),
                                           ])
    transforms.set_random_state(42)
    return transforms


class ADNIDataset(Dataset):
    def __init__(self, data_dir, meta_file_path, train_fraction, validation_fraction, test_fraction, transform=None, split='train'):
        super().__init__()
        
        self.data_dir = data_dir
        self.meta_file_path = meta_file_path
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction        
        self.transform = transform
        self.split = split

        self.classes = ['CN', 'AD']
        self.data = pd.read_csv(self.meta_file_path) # first of all load metadata
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
            patients = patients[:int(patients.shape[0] * self.train_fraction)]
        elif self.split == 'val':
            patients = patients[int(patients.shape[0] * self.train_fraction):int(patients.shape[0] * (self.train_fraction + self.validation_fraction))]
        elif self.split == 'test':
            patients = patients[int(patients.shape[0] * (self.train_fraction + self.validation_fraction)):] # leave test_fraction cause unnecessary
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
    def __init__(self, data_dir, meta_file_path, train_fraction, validation_fraction, test_fraction, transform=None, split='train'):
        super().__init__()
        self.data_dir = data_dir
        self.meta_file_path = meta_file_path
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction        
        self.transform = transform
        self.split = split

        self.classes = ['CN', 'AD']
        self.data = pd.read_csv(self.meta_file_path) # first of all load metadata
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
            patients = patients[:int(patients.shape[0] * self.train_fraction)]
        elif self.split == 'val':
            patients = patients[int(patients.shape[0] * self.train_fraction):int(patients.shape[0] * (self.train_fraction + self.validation_fraction))]
        elif self.split == 'test':
            patients = patients[int(patients.shape[0] * (self.train_fraction + self.validation_fraction)):]
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

    def __init__(self, dataset, batch_size, num_workers, data_dir, meta_file_path, train_fraction, validation_fraction, test_fraction, transform=None, split='train'):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.meta_file_path = meta_file_path
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.transform = transform
        self.split = split

    # execute on single GPU
    def prepare_data(self):
        pass

    # execute on every GPU
    def setup(self, stage):
        test_transform = None
        train_transform = None
        
        if self.dataset == "ADNI":
            dataset = ADNIDataset
        elif self.dataset == "ADNIRAM":
            dataset = ADNIDatasetRAM
        else:
            raise ValueError("dataset must be one of ADNI, ADNIRAM")
        
        self.train_ds = dataset(self.data_dir, self.meta_file_path, self.train_fraction, self.validation_fraction, self.test_fraction, train_transform, split='train')
        self.val_ds = dataset(self.data_dir, self.meta_file_path, self.train_fraction, self.validation_fraction, self.test_fraction, test_transform, split='val')
        self.test_ds = dataset(self.data_dir, self.meta_file_path, self.train_fraction, self.validation_fraction, self.test_fraction, test_transform, split='test')
               
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
