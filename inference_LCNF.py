import argparse
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import scipy.io as sci
import models
from utils import make_coord
from test import batched_predict
from skimage import io
import scipy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution')
    parser.add_argument('--output', default='output.png')
    args = parser.parse_args()

    img = np.load(args.input).astype('float32')
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=0)
    img = torch.tensor(img)
    temp = img

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda()
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h
    cell[:, 1] *= 2 / w
    pred = batched_predict(model, ((temp - 0) / 1).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = (pred * 1 + 0).clamp(0, 1).view(h, w, 1).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred.squeeze()).save(args.output)   # save as .png file
    # save as .mat file
    saveFile = args.output[:-3] + 'mat'
    data = pred.squeeze()
    data = data.numpy()
    mat = {'pred': data}
    sci.savemat(saveFile, mat)
