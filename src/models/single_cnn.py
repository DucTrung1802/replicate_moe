import torch.nn as nn
import torch.nn.functional as F
import torchinfo


class SingleCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(SingleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(128 * 8 * 8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 64, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 128, 8, 8]
        x = x.view(x.size(0), -1)  # Flatten -> [B, 8192]
        x = self.fc(x)  # -> [B, num_classes]
        return x


if __name__ == "__main__":
    model = SingleCNN(input_channels=3, num_classes=10)
    torchinfo.summary(model, input_size=(1, 3, 32, 32))
