import time
import torch
import os

from torch import nn, optim
import cv2
from EndomicroscopyImage import EndomicroscopyImage
import EndomicroscopyDataset
from vgg_pretrained import get_vgg19_model
from train import train
from EndomicroscopyDataset import get_transform
from meat import get_trained_model

if __name__ == "main":
    modelpath = "meat/2023-12-12-20-03.pt"
    model = get_trained_model(modelpath,3)

