# Baseline Code
### Very Simple Image Classification Code (PyTorch)
[Gogumatang](https://github.com/Gogumatang)

## This repository contains:
- Python3 / Pytorch code for multi-class image classification

## Prerequisites
- See `requirements.txt` for details.
~~~ME
torch == 1.7.1
torchvision == 0.8.2
torchmetrics == 0.7.2
tensorboard == 2.8.0
efficientnet_pytorch
image
~~~

## How to run
- For test: `python train.py -dataset='AIhub' -epoch=100 -ngpu=Your GPU number -model=efficientnet_b0 -name=0 -batch_size=1024 -lr=0.001 -resume`
- IF You really want, you can only change epoch and batch size
- IF ERROR Try This: `❌ Unknown dataset: {}: ERROR -> remove '' FOR dataset ex) -dataset='AIhub' -> -dataset=AIhub`
- You have to located your train data in `dataroot/AIhub/train => eachclass` and val data in `dataroot/AIhub/val => eachclass`
- You can remove sample folder and sample image 
  
## Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/alinlab/cs-kd
