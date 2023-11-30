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


def train(model, train_loader, test_loader, criterion, optimizer, epochs=25):
    """
    Train a VGG19 model.

    Parameters:
    - model (torch.nn.Module): The VGG19 model to train
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
    learning_rate = 0.0002
    batch_size = 64  # 16&64:orch.cuda.OutOfMemoryError: CUDA out of memory.
    epochs = 25
    dropout = 0.20

    # Datasets and DataLoaders
    train_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/train/')
    train_loader = EndomicroscopyDataset.DataLoader(train_dataset, batch_size)

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, batch_size)

    # Model, Loss, and Optimizer
    vgg19 = get_vgg19_model(pretrained=True, num_classes=num_classes, dropout=dropout)  # Set pretrained=False for training from scratch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg19.parameters(), lr=learning_rate)

    # Train the model
    start_time = time.time()
    train(vgg19, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")

