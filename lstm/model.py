import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, conditional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)

        if conditional:
            self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_features+conditional_features, out_features=64),
            nn.ReLU()
        )
        else:
            self.fc1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=64),
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
   
   def forward(self, x, conditional):
        out, (h,c) = self.lstm(x.unsqueeze(1))
        out = out.squeeze(1)

        if conditional:
            out = torch.cat([out,conditional], dim=1)
        
        y = self.fc1(out)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        return y