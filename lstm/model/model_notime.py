import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU()
        )
        self.fc3= nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU()
        )
        
        self.fc4 = nn.Linear(16, 1)
        
    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        return y