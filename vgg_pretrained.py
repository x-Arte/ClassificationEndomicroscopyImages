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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle
import seaborn as sns


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
def classify(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels, name in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("Accuracy:", accuracy)
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    y_score = model(inputs).numpy()
    fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    ### cancer
    modelpath = "model/2023-12-12-18-04.pt"
    model = get_vgg19_model()
    model.load_state_dict(torch.load(modelpath))

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, 64)





