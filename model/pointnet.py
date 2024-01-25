import torch
import torch.nn as nn

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.network = nn.Sequential(
            # Conv
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.Relu(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.Relu(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.Relu(inplace=True),
            # MaxPool
            nn.MaxPool1d(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Relu(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Relu(inplace=True),
            nn.Linear(256, k * k),
        )

    def forward(self, input):
        # Initialises as identity matrix
        init = torch.eye(self.k, requires_grad=True).repeat(input.size(0), 1, 1)
        output = self.network(input).view(-1, self.k, self.k) + init
        return output
