import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

from datasets import register
from utils import to_pixel_samples, to_pixel_samples2
from torchvision.transforms import InterpolationMode

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))

def resize_fn_multi(img, size):
    [channel, _, _] = img.shape
    data = torch.zeros([channel, size[0], size[1]])
    for i in range(channel):
        temp = img[i, :, :].unsqueeze(0)
        temp = transforms.ToTensor()(transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(temp)))
        data[i, :, :] = torch.squeeze(temp)
    return data

def resize_fn_T(img, size):
    temp = torch.unsqueeze(img, 0)
    temp = F.interpolate(temp, size)
    temp = torch.squeeze(temp, 0)
    return temp

@register('sr-implicit-downsampled-pair-output')
class SRImplicitDownsampledPairOutput(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]   # load a pair of high/low resolution data in dataset
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img_hr.shape[-2] / s + 1e-9)
            w_lr = math.floor(img_lr.shape[-1] / s + 1e-9)
            img_down = resize_fn(img_hr, (h_lr, w_lr))   # downsample high resolution image
            crop_lr, crop_hr = img_lr, img_down
        else:
            w_lr = self.inp_size
            s0 = img_hr.shape[-2] // img_lr.shape[-2]  # assume int scale of low and high resolution image pixels

            x0 = random.randint(0, img_lr.shape[-2] - w_lr)   # crop for low resolution images
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]

            w_hr_down = round(w_lr * s0 / s)  # crop corresponding high-resolution image
            w_hr = round(w_lr * s0)
            x1 = round(x0*s0)   # crop for ground truth high resolution images
            y1 = round(y0*s0)
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]
            crop_hr = resize_fn_T(crop_hr, (w_hr_down, w_hr_down))  # here, I downsample the high-resolution images, which may not necessary in our case

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples2(crop_hr.contiguous()) #get the coordinates and the corresponding value pairs

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)   # here, it's the pixel size of image
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }