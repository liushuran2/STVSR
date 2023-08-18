
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import h5py

class TestDataset(Dataset):
    def __init__(self, h5_file, patch_size, scale):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file
        self.patch_size = patch_size
        self.scale = scale

    @staticmethod
    def random_crop(lr, hr, size, scale):
        lr_left = random.randint(0, lr.shape[3] - size)
        lr_right = lr_left + size
        lr_top = random.randint(0, lr.shape[2] - size)
        lr_bottom = lr_top + size
        hr_left = lr_left * scale
        hr_right = lr_right * scale
        hr_top = lr_top * scale
        hr_bottom = lr_bottom * scale
        lr = lr[:, :, lr_top:lr_bottom, lr_left:lr_right]
        hr = hr[:, :, hr_top:hr_bottom, hr_left:hr_right]
        return lr, hr

    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, :, ::-1].copy()
            hr = hr[:, :, :, ::-1].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, :, ::-1, :].copy()
            hr = hr[:, :, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(3, 2)).copy()
            hr = np.rot90(hr, axes=(3, 2)).copy()
        return lr, hr

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            video_id = idx // 1
            num = idx % 1
            lr_big = f['lr'][str(video_id)]
            #lr = f['lr'][str(idx)]
            lr_big = np.array(lr_big)
            h = num // 4
            w = num % 4
            lr = lr_big
            #lr = lr_big[:,:,w*128:w*128+128,h*128:h*128+128]
            #lr = lr_big[:,:,w*96:w*96+128,h*96:h*96+128] #128patch的1/8
            #lr = lr_big[:,:,w*144:w*144+192,h*144:h*144+192] #192patch的1/8
            #lr = lr_big[:,:,w*144:w*144+224,h*144:h*144+224]
            #lr = lr_big[:,:,w*256:w*256+256,h*256:h*256+256]
            #lr = lr_big[:,:,w*64:w*64+128,h*64:h*64+128]  #128patch的1/4
            #lr = lr_big[:,:,w*32:w*32+64,h*32:h*32+64]
            hr_big = f['hr'][str(video_id)]
            hr = hr_big
            #hr = hr_big[:,:,w*384:w*384+384,h*384:h*384+384]
            #hr = hr_big[:,:,3*w*96:3*w*96+3*128,3*h*96:3*h*96+3*128]#128patch的1/8
            #hr = hr_big[:,:,2*w*144:2*w*144+2*192,2*h*144:2*h*144+2*192]#928patch的1/8
            #hr = hr_big[:,:,3*w*144:3*w*144+3*224,3*h*144:3*h*144+3*224]#224patch的1/4
            #hr = hr_big[:,:,3*w*256:3*w*256+3*256,3*h*256:3*h*256+3*256]
            #hr = hr_big[:,:,3*w*32:3*w*32+3*64,3*h*32:3*h*32+3*64]
            #lr, hr = self.random_crop(lr, hr, self.patch_size, self.scale)
            #lr, hr = self.random_horizontal_flip(lr, hr)
            #lr, hr = self.random_vertical_flip(lr, hr)
            #lr, hr = self.random_rotate_90(lr, hr)
            gt = hr[3, :, :, :]
            return lr.astype(np.float32), gt.astype(np.float32)

    def __len__(self):
        return 5*1