import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_features, out_features=16)
        self.linear2 = nn.Linear(in_features=16, out_features=32)
        self.linear3 = nn.Linear(in_features=32, out_features=32)
        self.linear4 = nn.Linear(in_features=32, out_features=32)
        self.linear5 = nn.Linear(in_features=32, out_features=16)
        self.output_layer = nn.Linear(in_features=16, out_features=out_features)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = self.output_layer(x)
        return x