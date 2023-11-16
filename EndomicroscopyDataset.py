import torch
import os
import random
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from EndomicroscopyImage import EndomicroscopyImage

class EndomicroscopyDataset(Dataset):
    def __init__(self, path, transform='vgg',shuffle=False):
        super(EndomicroscopyDataset, self).__init__()
        if not os.path.isdir(path):
            raise NotADirectoryError
        self.root = path
        self.ls = os.listdir(path)
        if shuffle:
            random.shuffle(self.ls)
        if transform == 'vgg':
            self.transform = transforms.Compose([
                #transforms.Grayscale(num_output_channels=3),  # gray to likeRGB
                transforms.ToTensor(),
                transforms.Resize((224, 224)),  # adjust the size
                #transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
        #elif:
        else:
            self.transform = transform


    def __getitem__(self, idx):
        img = torch.load(os.path.join(self.root, self.ls[idx]))
        image = img.image
        image = self.transform(image)
        return image, img.label, img.filename

    def __len__(self):
        return len(self.ls)