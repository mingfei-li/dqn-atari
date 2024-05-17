from torch import nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, output_units):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, output_units)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x