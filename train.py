import pandas as pd
import torch
from torch import optim, nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import time
import EndomicroscopyDataset
import vgg_pretrained
import ResNet
import numpy as np

def train(model, train_loader, test_loader, criterion, optimizer, epochs=25, num_classes = 2):
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
    #print("Start:"+str(start_time))
    tot_data = []

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0.0
        testing_loss = 0.0
        for inputs, labels, name in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # forward
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"----------------------- \n Epoch {epoch + 1}/{epochs} - Training loss: {running_loss / len(train_loader)} - Accuracy: {100 * train_correct / train_total} %")

        # Evaluate on the test set
        model.eval()
        # calculate the accuracy
        test_correct = 0
        test_total = 0
        # calculate auc
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for inputs, labels, name in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                testing_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                # logits to possibility
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_outputs.extend(probabilities)
        if num_classes == 2:
            # 2
            auc_score = roc_auc_score(all_labels, np.array(all_outputs)[:, 1])
        else:
            # >= 2
            all_labels_binarized = label_binarize(all_labels, classes=range(num_classes))
            auc_score = roc_auc_score(all_labels_binarized, np.array(all_outputs), multi_class='ovr')
        print(f' Testing loss: {testing_loss / len(test_loader)} Accuracy: {100 * test_correct / test_total} % - Test AUC: {auc_score}')

        tot_data.append([epoch,running_loss / len(train_loader),testing_loss / len(test_loader), 100 * train_correct / train_total, 100 * test_correct / test_total, auc_score])

    print('Finished Training')
    print('Wait to save')

    torch.save(model.state_dict(),'model/'+time.strftime('%Y-%m-%d-%H-%M', time.localtime())+'.pt')
    df = pd.DataFrame(tot_data, columns=['epoch', 'training loss', 'testing loss', 'train accuracy', 'test accuracy', 'AUC score'])
    df.to_csv('model/acc/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())+'.csv')

if __name__ == '__main__':
    # Hyperparameters
    num_classes = 2  # Two classes
    learning_rate = 0.0001
    batch_size = 64
    epochs = 100
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
