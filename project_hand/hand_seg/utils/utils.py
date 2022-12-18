import numpy as np
import torch
from torch import nn
import pandas as pd
from PIL import Image


# dataset load


class Dataset_synth(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        df_1 = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\synthesis_hand\lut.csv')
        df_2 = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\RHD_hand\lut_masks_negative.csv')
        df = pd.concat([df_1, df_2])
        self.data = df[:]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data.iloc[idx]['image'], self.data.iloc[idx]['mask']
        
        img = np.array(Image.open(img_path))
        mask = (np.array(Image.open(mask_path)) / 255) > .5
        mask = mask.astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask


class Dataset_freihand(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\frei_hand\lut.csv')
        self.data = df[:10000]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data.iloc[idx]['imgs'], self.data.iloc[idx]['masks']
        
        img = np.array(Image.open(img_path))
        mask = (np.array(Image.open(mask_path)) / 255) > .5
        mask = mask.astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask


class Dataset_egohand(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\egohand\lut.csv')
        self.data = df[:]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data.iloc[idx]['img'], self.data.iloc[idx]['mask']
        
        img = np.array(Image.open(img_path))
        mask = (np.array(Image.open(mask_path)) / 255) > .5
        mask = mask.astype(np.float32)

        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask


class Dataset_RHD(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\RHD_hand\lut_mask.csv')
        # df_2 = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\RHD_hand\lut_masks_negative.csv')
        # df_3 = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\RHD_hand\lut_masks_negative_2.csv')
        self.data = df[:5340]
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data.iloc[idx]['img'], self.data.iloc[idx]['mask']
        
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        mask = mask.astype(np.float32)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']

        return img, mask


class Dataset_DA_target(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv(r'\\wsl.localhost\Ubuntu-20.04\home\hebb\ml\datasets\RHD_hand\lut_target_domain.csv')
        self.data = df[:5340]
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['img']
        
        img = np.array(Image.open(img_path))
        
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img


# dice
def BCEDice(pred, gt):
    criterion = nn.BCELoss()
    
    bce = criterion(pred, gt)
    dice = 1 - get_dice(pred, gt)
    loss = bce + dice

    return loss


def get_dice(pred, gt):
    summ = torch.sum(gt) + torch.sum(pred)
    inter = torch.sum(gt * pred)
    dice = 2 * inter / summ
    
    return dice