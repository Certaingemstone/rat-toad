# for training a model to distinguish images of rats and toads

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split

from rattoadset import RatToadDataset

num_epochs = 15

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# modify pretrained model to have 2 output classes at final FC layer
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# input transformation for resnet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ratstoads = RatToadDataset("labels.csv", "all", transform=transform)

trainset, testset = random_split(ratstoads, [0.9, 0.1])
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

# training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters())

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader)}")

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test accuracy: {100 * correct / total}%")

torch.save(model.state_dict(), "finetuned.pth")