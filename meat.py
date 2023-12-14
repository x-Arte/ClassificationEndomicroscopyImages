import time
import numpy as np
import torch
import os
from torch import nn, optim
import cv2
from EndomicroscopyImage import EndomicroscopyImage
import EndomicroscopyDataset
from vgg_pretrained import get_vgg19_model
from train import train
from vgg_pretrained import classify

def get_trained_model(modelpath, num_classes, num_features):
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
        nn.Linear(model.classifier[6].in_features, num_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.20, inplace=False),
        nn.Linear(num_features, num_classes)
    )
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
    model = get_trained_model(modelpath, 3,64)
    # Hyperparameters
    num_classes = 3  # Two classes
    learning_rate = 0.0001
    batch_size = 64
    epochs = 150

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

def extract_features(model,data_loader,save_name):
    model.eval()
    device = torch.device('cuda')  # if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    all_features = []
    labels = []
    cnt = 0
    with torch.no_grad():
        for inputs, label, name in data_loader:
            cnt+=1
            inputs = inputs.to(device)
            features = model(inputs)
            features_np = features.cpu().numpy()
            all_features.append(features_np)
            labels.append(label.cpu().numpy())
    all_features_array = np.concatenate(all_features, axis=0)
    all_labels_array = np.array(labels)
    if all_features_array.shape[0] == all_labels_array.shape[0]:
        combined_data = np.column_stack((all_features_array, all_labels_array))
    else:
        print("features and label are not in same length")
        combined_data = None
    print(cnt)
    np.savetxt(save_name, combined_data, delimiter=",")

def save_features():
    modelpath = "meat/2023-12-13-16-44.pt"
    model = get_trained_model("model/2023-12-12-18-04.pt", 3, 64)
    model.load_state_dict(torch.load(modelpath))
    model.classifier = nn.Sequential(*list(model.classifier.children())[:7])

    print(model)
    train_dataset = EndomicroscopyDataset.EndomicroscopyDataset('meat/dataset/train/')
    train_loader = EndomicroscopyDataset.DataLoader(train_dataset, 1)
    extract_features(model, train_loader,"vgg_train.csv")

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('meat/dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, 1)
    extract_features(model, test_loader,"vgg_test.csv")



if __name__ == "__main__":
    #save_pt("E:\OneDrive\IC\group project\\new_dataset_pcle (2)\\new_dataset_pcle\\pork\\train","pork","meat/dataset/train/")
    #transfer_train()
    #save_features()
    ###meat
    modelpath = "meat/2023-12-13-16-44.pt"
    model = get_trained_model("model/2023-12-12-18-04.pt", 3, 64)
    model.load_state_dict(torch.load(modelpath))
    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('meat/dataset/test/')
    test_loader = EndomicroscopyDataset.DataLoader(test_dataset, 64)

    ### ————————————————————————————————————————————————####
    classify(model, test_loader)









