from torch.utils.data import Dataset, random_split
import os
import torch
from torch.utils.data import random_split,Dataset
import os
import torch

class EndomicroscopyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        get a data according to the idx
        """
        return self.dataset[idx]

def load_and_randomsplit_dataset(root_dir, train_size=0.8):
    """
    load EndomicroscopyImage data and split them into train dataset and test data set
    :param root_dir: the path of .pt
    :param train_size:
    :return: train dataset and test dataset
    """
    items = []

    # load EndomicroscopyImage
    for file in os.listdir(root_dir):
        if file.endswith('.pt'):
            item = torch.load(os.path.join(root_dir, file))
            items.append(item)

    # split
    train_len = int(len(items) * train_size)
    test_len = len(items) - train_len
    train_items, test_items = random_split(items, [train_len, test_len])

    train_dataset = EndomicroscopyDataset(train_items)
    test_dataset = EndomicroscopyDataset(test_items)

    return train_dataset, test_dataset

def save_dataset(dataset, file_path):
    """
    save the dataset into files
    :param dataset:
    :param file_path:
    """
    torch.save(dataset, file_path)

if __name__ == "__main__":
    root_dic = 'dataset/images/train/'
    save_dic = 'dataset/'
    train_size = 1
    train_dataset, test_dataset = load_and_randomsplit_dataset(root_dic, train_size)
    print("train_dataset size:"+str(train_dataset.__len__()))
    print("test_dataset size:" + str(test_dataset.__len__()))
    save_dataset(train_dataset,save_dic+'train1.dat')
    save_dataset(test_dataset,save_dic+'test1.dat')




