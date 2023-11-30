import pandas as pd
import torch.nn as nn
from torchvision import models
import torch
from torch import optim, nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
import time
import EndomicroscopyDataset
from EndomicroscopyImage import EndomicroscopyImage

def get_vgg19_model(pretrained=True, num_classes=2, dropout = 0.25):
    vgg19 = models.vgg19(pretrained=pretrained)
    for i in vgg19.parameters():
        i.requires_grad = False
    # Replace the classifier to match the number of classes
    vgg19.classifier[5] = nn.Dropout(dropout)
    vgg19.classifier[6] = nn.Linear(vgg19.classifier[6].in_features, num_classes)
    # vgg19.classifier.add_module("7", nn.ReLU(inplace=True))
    # vgg19.classifier.add_module("9", nn.Linear(32, num_classes))
    # vgg19.classifier.add_module("10", nn.Softmax())
    print(vgg19)
    return vgg19


