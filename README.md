# LCNF for large-scale phase retrieval

This repository contains the implementation for LCNF introduced in the following paper:

[Local Conditional Neural Fields for Versatile and Generalizable Large-Scale Reconstructions in Computational Imaging](https://arxiv.org/abs/2307.06207)

![overview](https://github.com/bu-cisl/LCNF/assets/56607928/686e26c2-065c-4002-b559-12c810011ced)

## Environment
- Python 3.8.10
- pytorch 1.13.1
- TensorboardX
- PyYAML, numpy, tqdm, imageio

## Network Training
Run `python train_LCNF.py --config configs/test.yaml`.

--config:  predefined network training parameters in .yaml file

Here, we provide ethanol-fixed Hela cells at [Dataset](Dataset) for training (one pair of data)

## Network Inference
Run `python inference_LCNF.py --input Dataset/Inference/22.npy --model save/_test/epoch-best.pth --resolution 1500,1500 --output Inference/pred22.png`.

Once the network finished training, we will use the model saved in 'save/_test/epoch-best.pth' to reconstruct the high-resolution phase image. 

--input:  input preprocessed low-resolution image

--model:  trained model, saved in the [save](save) folder

--resolution:  queried pixel resolution for high-resolution image

--output:  save reconstructed image

## Trained models
To use our trained models, first download models (https://drive.google.com/drive/folders/1zIBaYvLkABGXDJFIzgbtJoXavZIBUmAU?usp=sharing), then change the --model name to the trained models when running the inference_LCNF.py

To request more data, please contact the author: Hao Wang, wanghao6@bu.edu





