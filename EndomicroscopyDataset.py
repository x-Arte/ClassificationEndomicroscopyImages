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
        image = torch.concat((image, image, image), dim=0)
        return image, img.label, img.filename

    def __len__(self):
        return len(self.ls)

def set_label(path):
    if not os.path.isdir(path):
        raise NotADirectoryError
    cnt = 0
    cnt_1 = 0
    for filename in os.listdir(path):
        img = torch.load(path+filename)
        label = -1
        if filename.startswith('G') or filename.startswith('g'):
            label = 0
        elif filename.startswith('M') or filename.startswith('m'):
            label = 1
        img.label = label
        cnt += 1
        if label == -1:
            cnt_1 += 1
            print(cnt_1,filename)
        img.save(path)
    print(cnt)




if __name__ == '__main__':
    path = "dataset/test/"
    set_label(path)