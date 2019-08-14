import torch
from torch import nn


class ConvNet(torch.nn.Module):
    def __init__(self, fc: int, first_conv: int = 32, first_fc: int = 1000):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, first_conv, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(first_conv, first_conv * 2, kernel_size=3), nn.ReLU(), nn.MaxPool2d(kernel_size=2)
        )
        self.drop_out = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(6 * 6 * 64, first_fc)  # Fully Connected
        self.fc2 = nn.Linear(first_fc, fc)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flat
        out = self.drop_out(out)
        out = self.fc1(out).clamp(min=0)
        out = self.fc2(out)
        x_coord, y_coord = torch.split(out, 1, dim=1)
        return x_coord, y_coord
