"""Basic Mockup of an RNN, which can be used along with the Bayesian Model as its preprocessor"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SatelliteSnowAccumAnalysis.Training.ImageDataset import ImageDataset

classification_labels = {
    "melting": 0,
    "stable": 1,
    "accumulating": 2,
}


class RNNClassifier(nn.Module):
    def __init__(self, image_resolution_h, image_resolution_w):
        super().__init__()

        self.image_resolution_h = image_resolution_h
        self.image_resolution_w = image_resolution_w
        self.lstm_input_size = 3 * image_resolution_w
        self.lstm_hidden_size = 64

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.ff1 = nn.Linear(self.lstm_hidden_size, 64)
        self.relu1 = nn.ReLU()

        self.ff2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.ff3 = nn.Linear(32, 3)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.image_resolution_h, self.lstm_input_size)
        x, (hidden_state, cell_state) = self.lstm(x)
        x = hidden_state[-1]
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.relu2(x)
        x = self.ff3(x)
        return x


image_resolution_h = 128
image_resolution_w = 128

dataset = ImageDataset("satellite_img_data.jsonl")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RNNClassifier(image_resolution_h, image_resolution_w).to(device)
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

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

torch.save(model.state_dict(), "rnn_classifier.pt")