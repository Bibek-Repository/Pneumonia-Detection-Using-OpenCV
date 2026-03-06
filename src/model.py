import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes

        # Dropout (prevents overfitting)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))  # 224 → 112

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))  # 112 → 56

        # Block 3
        x = self.pool(F.relu(self.conv3(x)))  # 56 → 28

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x