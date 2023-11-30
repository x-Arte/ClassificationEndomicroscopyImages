import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import EndomicroscopyDataset
from torchvision import models

def get_resnet18_model(pretrained=True, num_classes=2):
    resnet18 = models.resnet18(pretrained = pretrained)
    for i in resnet18.parameters():
        i.requires_grad = False
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)
    print(resnet18)
    return resnet18




    