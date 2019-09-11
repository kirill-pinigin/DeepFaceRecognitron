import torch
import torch.nn as nn


class SILU(torch.nn.Module):
    def __init__(self):
        super(SILU, self).__init__()

    def forward(self, x):
        out = torch.mul(x, torch.sigmoid(x))
        return out


class Perceptron(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(Perceptron, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x