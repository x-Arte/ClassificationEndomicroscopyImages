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
import vgg_pretrained
import ResNet

def train(model, train_loader, test_loader, criterion, optimizer, epochs=25):
    """
    Train a model.

    Parameters:
    - model (torch.nn.Module): The VGG19/ResNet18 model to train
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test set
    - criterion: Loss function
    - optimizer: Optimizer
    - epochs (int): Number of training epochs
    """
    if torch.cuda.is_available():
        print('yes gpu')
    else:
        print('oh god cpu')
    device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Start:"+str(start_time))
    tot_data = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, name in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Training loss: {running_loss / len(train_loader)}")

        # Evaluate on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, name in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the test images: {100 * correct / total} %')
        tot_data.append([epoch,running_loss / len(train_loader), 100 * correct / total])

    print('Finished Training')
    print('Wait to save')

    torch.save(model.state_dict(),'model/'+time.strftime('%Y-%m-%d-%H-%M', time.localtime())+'.pt')
    df = pd.DataFrame(tot_data, columns=['epoch', 'training loss', 'accuracy'])
    df.to_csv('model/acc/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())+'.csv')

if __name__ == '__main__':
    # Hyperparameters
    num_classes = 2  # Two classes
    learning_rate = 0.0001
    batch_size = 64
    epochs = 50
    dropout = 0.20 # Resnet don‘t have to change

    # Datasets and DataLoaders
    train_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/train/')
    train_loader = EndomicroscopyDataset.DataLoader(train_dataset, batch_size)

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, batch_size)

    # Model, Loss, and Optimizer
    ################### model ###########################
    vgg19 = vgg_pretrained.get_vgg19_model(pretrained=True, num_classes=num_classes, dropout=dropout)  # Set pretrained=False for training from scratch
    #resnet18 = ResNet.get_resnet18_model(pretrained=True, num_classes=num_classes)
    optimizer = optim.Adam(vgg19.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(resnet18.parameters(), lr=learning_rate)
    ################### model ###########################
    criterion = nn.CrossEntropyLoss()

    # Train the model
    start_time = time.time()
    ################### model ###########################
    train(vgg19, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    #train(resnet18, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    ################### model ###########################
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")
