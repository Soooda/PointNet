import torch
import torch.nn as nn

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2048),
        )
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, k * k),
        )

    def forward(self, input):
        # Initialises as identity matrix
        init = torch.eye(self.k, requires_grad=True).repeat(input.size(0), 1, 1)
        output = self.conv(input)
        output = nn.MaxPool1d(output.size(-1))(output)
        output = self.fc(output).view(-1, self.k, self.k)
        init = init.to(output.get_device())
        output += init
        return output
    

class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # Batch Matrix Multiplication
        output = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)
        output = self.conv1(output)

        matrix64x64 = self.feature_transform(output)
        output = torch.bmm(torch.transpose(output, 1, 2), matrix64x64).transpose(1, 2)

        output = self.conv2(output)
        output = self.conv3(output)
        output = nn.MaxPool1d(output.size(-1))(output)
        output = nn.Flatten(1)(output)
        return output, matrix3x3, matrix64x64
    

class PointNet(nn.Module):
    def __init__(self, classes=40):
        super().__init__()
        self.transform = Transform()
        self.network = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input):
        output, matrix3x3, matrix64x64 = self.transform(input)
        output = self.network(output)
        return output, matrix3x3, matrix64x64
    
if __name__ == "__main__":
    model = PointNet()
    print(model)
