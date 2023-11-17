import torch
import os
import cv2
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

def save_png(dicname, targetdic):
    if not(dicname):
        raise NotADirectoryError
    if not(targetdic):
        raise NotADirectoryError
    cnt = 0
    for filename in os.listdir(dicname):

        cnt+=1
        data = torch.load(os.path.join(dicname, filename))
        image = data.image
        filename = filename[:filename.rfind('.')]
        filename = filename + '.png'
        print(filename)
        cv2.imwrite(os.path.join(targetdic,filename),image)
        print(cnt)

if __name__ == '__main__':
    sourcedic = 'dataset/images/test'
    targetdic = 'dataset/images/test_png'
    save_png(sourcedic, targetdic)