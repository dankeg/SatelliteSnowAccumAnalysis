"""Sets up an initial CNN, designed for the Snow Trend Prediction Task"""
import json
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from SatelliteSnowAccumAnalysis.Training.ImageDataset import ImageDataset

classification_labels = {
    "melting": 0,
    "stable": 1,
    "accumulating": 2,
}

class ClassifierCNN(nn.Module):
    def __init__(self, image_resolution_h, image_resolution_w):
        super().__init__()
        conv_kernel_size = 3
        conv_padding = conv_kernel_size // 2
        pooled_width = image_resolution_w // 4
        pooled_height = image_resolution_h // 4

        self.conv1 = nn.Conv2d(3, 16, kernel_size=conv_kernel_size, padding=conv_padding)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=conv_kernel_size, padding=conv_padding)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.ff1 = nn.Linear(32 * pooled_height * pooled_width, 64)
        self.relu3 = nn.ReLU()

        self.ff2 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()

        self.ff3 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.ff1(x)
        x = self.relu3(x)

        x = self.ff2(x)
        x = self.relu4(x)

        x = self.ff3(x)

        return x
        

dataset = ImageDataset("satellite_img_data.jsonl")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassifierCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

torch.save(model.state_dict(), "cnn_classifier_model.pt")