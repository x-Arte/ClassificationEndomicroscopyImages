import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import EndomicroscopyDataset

class VGG19(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG19, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Adjust the input features to match your image size
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # Change to 3 if you are dealing with RGB images
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

def train(model, train_loader, test_loader, criterion, optimizer, epochs=25):
    if torch.cuda.is_available():
        print('yes gpu')
    else:
        print('oh god cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels, _ in train_loader:
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
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct / total} %')
    print('Finished Training')

if __name__ == '__main__':
    num_classes = 2  # Update based on the number of classes in your dataset
    learning_rate = 0.001
    batch_size = 8
    epochs = 25

    train_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/images/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = EndomicroscopyDataset.EndomicroscopyDataset('dataset/images/test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    vgg19 = VGG19(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vgg19.parameters(), lr=learning_rate)

    start_time = time.time()
    train(vgg19, train_loader, test_loader, criterion, optimizer, epochs=epochs)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}s")
