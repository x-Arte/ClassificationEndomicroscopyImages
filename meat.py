import time
import torch
import os

from torch import nn, optim
import cv2
from EndomicroscopyImage import EndomicroscopyImage
import EndomicroscopyDataset
from vgg_pretrained import get_vgg19_model
from train import train

def get_trained_model(modelpath, num_classes):
    model = get_vgg19_model()
    model.load_state_dict(torch.load(modelpath))
    for i in model.parameters():
        i.requires_grad = False
    # Replace the classifier to match the number of classes
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        model.classifier[2],
        model.classifier[3],
        model.classifier[4],
        model.classifier[5],
        nn.Linear(model.classifier[6].in_features, model.classifier[6].in_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20, inplace=False),
        nn.Linear(model.classifier[6].out_features, num_classes)
    )
    model.classifier[9] = nn.Linear(model.classifier[6].out_features, num_classes)
    print(model)
    return model
def save_pt(path, label, outputpath):
    if not os.path.isdir(path):
        raise NotADirectoryError
    cnt = 0
    for filename in os.listdir(path):
        temp_path = path + '/' + filename
        img = cv2.imread(temp_path, 0)
        save_label = -1
        if label == "beef":
            save_label = 0
        elif label == "ck":
            save_label = 1
        elif label == "pork":
            save_label = 2
        eimg = EndomicroscopyImage(label+filename[:filename.rfind('.')], cnt, save_label, img)
        eimg.save(outputpath)
        cnt += 1
    print(cnt)

def transfer_train():
    modelpath = "model/2023-12-12-18-04.pt"
    model = get_trained_model(modelpath, 3)
    # Hyperparameters
    num_classes = 3  # Two classes
    learning_rate = 0.0001
    batch_size = 64
    epochs = 150
    dropout = 0.20

    # Datasets and DataLoaders
    train_dataset = EndomicroscopyDataset.EndomicroscopyDataset('meat/dataset/train/')
    train_loader = EndomicroscopyDataset.DataLoader(train_dataset, batch_size)

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('meat/dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    # Train the model
    start_time = time.time()
    train(model, train_loader, test_loader, criterion, optimizer, epochs=epochs, num_classes=num_classes)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")


if __name__ == "__main__":
    #save_pt("E:\OneDrive\IC\group project\\new_dataset_pcle (2)\\new_dataset_pcle\\pork\\train","pork","meat/dataset/train/")
    transfer_train()




