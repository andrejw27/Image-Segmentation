import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class VisionDataset(Dataset):

    def __init__(self, img_dir, channel=None, transform=None):
        """
        Args:
            img_dir(string): Path to the csv file with paths to image files.
            channel (string): Number of channels to read (either RGB channel or All/Multispectral).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img = pd.read_csv(img_dir)
        self.channel = channel
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = self.img.iloc[idx,0]
        segimg_file = self.img.iloc[idx,1]
        
        img_raster = rasterio.open(img_file) #open a file with .tif extension using restario
        segimg_raster = rasterio.open(segimg_file)
        norm_img = self.norm_image(img_raster, self.channel)     
        img_raster.close()

        seg_img = segimg_raster.read()
        if np.max(seg_img) == 255: #there is apparently a segmented image with values of 255 which are beyond the number of classes  
            seg_img[seg_img==255] = 1
        segimg_raster.close()
        
        if self.transform: #randomly flip along width (left-right) and height(up-down)
            np.random.seed(1)
            n = np.random.randint(2,3)
            if n != 0:
                norm_img  = torch.tensor(norm_img).flip(n)
                seg_img = torch.tensor(seg_img).flip(n)
                
        sample = {'image': np.array(norm_img), 'label': np.array(seg_img)}
        return sample
    
    def norm_image(self, src, channel):
        if channel == 'RGB':
            idx = [4,3,2]
        else:
            idx = channel

        data = src.read(idx)

        return rasterio.plot.adjust_band(data, kind='linear')


def create_dataloader(dataset,batch_size, shuffle_dataset = False):
    train_test_split = .8
    random_seed= 1

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_test_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Creating data indices for training, validation, and testing splits:
    # divide the whole dataset into training and testing dataset: training:testing = 80:20
    # from the training dataset, divide it into training and validation dataset: training:validation = 80:20
    train_indices_temp, test_indices = indices[:split], indices[split:]
    train_size = len(train_indices_temp)
    val_indices = list(range(train_size))
    split_val = int(np.floor(train_test_split * train_size))
    train_indices, val_indices = val_indices[:split_val], val_indices[split_val:]

    # Creating data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=test_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=val_sampler)
    
    return train_loader,test_loader,val_loader
