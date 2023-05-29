from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
import toml
import monai
import torch

config_path = Path(__file__).parents[1] / 'paths.toml'
config = toml.load(config_path)


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
    def __init__(self, tfms=None,
                    train_fraction=100,
                 split='train',
                 split_seed=42,
                 train_pct=0.6,
                 val_pct=0.2, ):
        df = pd.read_csv(config['ADNI_CSV'])
        self.adni_dir = Path(config['ADNI_NPY_DATASET'])

        self.classes = ['CN', 'AD']

        self.df = self.preprocess_dataframe(df)

        # split of patients
        self.patients = self.df.PTID.unique()
        rng = np.random.RandomState(split_seed)
        patients_nb = len(self.patients)
        permutation = rng.permutation(patients_nb)
        train_last_idx = int(patients_nb * train_pct)
        val_last_idx = int(patients_nb * (train_pct + val_pct))
        if split == "train":
            if train_fraction<100:
                idx = train_last_idx * train_fraction // 100
                train_patients = self.patients[permutation[:idx]]
                print(f'Using the fraction of the dataset ({train_fraction} %) - {idx} patients')
            else:
                train_patients = self.patients[permutation[:train_last_idx]]


            self.df = self.df[self.df.PTID.isin(train_patients)]
        elif split in ["val", "valid"]:
            valid_patients = self.patients[permutation[train_last_idx:val_last_idx]]
            self.df = self.df[self.df.PTID.isin(valid_patients)]
        elif split == "test":
            test_patients = self.patients[permutation[val_last_idx:]]
            self.df = self.df[self.df.PTID.isin(test_patients)]
        else:
            raise ValueError(f"split {split} not a valid option")

        self.idx_to_class = {i: j for i, j in enumerate(self.classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}
        self.tfms = tfms

    def preprocess_dataframe(self, df):
        npy_files = list(self.adni_dir.glob('*'))
        img_uid = [path.name.split('.')[0].split('_')[-1] for path in npy_files]
        df = df[df.IMAGEUID.isin(img_uid)]

        df = df[df.DX.isin(self.classes)]

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_uid = self.df.iloc[item]['IMAGEUID']
        dx = self.df.iloc[item]['DX']

        image_path = list(self.adni_dir.glob(f'file_{img_uid}.npy*'))[0]

        img = np.load(image_path)
        if isinstance(img, np.lib.npyio.NpzFile):
            img = img['arr_0']
        img = (img - img.min()) / (img.max() - img.min())
        img = img[None, ...]  # add channel dim

        if self.tfms:
            img = self.tfms(img)

        label = self.class_to_idx[dx]

        return img, label, dx


if __name__ == '__main__':
    datasets = []
    for split in ['train', 'val', 'test']:
        datasets.append(ADNIDataset(split=split))
        datasets[0][0]
        

    print('succes')