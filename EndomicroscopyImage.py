import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

class EndomicroscopyImage:
    def __init__(self,
                 filename: str,
                 number: int,
                 label: int,
                 image):
        self.label = label
        self.filename = filename
        self.number = number
        self.image = image

    def save(self, outputpath='dataset/images'):
        torch.save(self, f'{outputpath}{self.filename}_{self.number}.pt')



