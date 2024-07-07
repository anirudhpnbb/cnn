import torch.nn as nn
import torch.nn.functional as F


class PneumoniaDetectionCNN(nn.Module):

    def __init__(self):
        super(PneumoniaDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128*28*28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
       x = self.pool1(self.bn1(F.relu(self.conv1(x))))
       x = self.pool2(self.bn2(F.relu(self.conv2(x))))
       x = self.pool3(self.bn3(F.relu(self.conv3(x))))
       x = x.view(-1, 128 * 28 * 28)
       x = F.relu(self.fc1(x))
       x = self.dropout(x)
       x = self.fc2(x)
       return x