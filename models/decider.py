import torch
from torch import nn


class KeyDecider(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, x):
        channels = x.size(0).item()
        if channels % 2 != 0:
            raise Exception('channels of heatmaps dont equal uncertainty maps')
        k = x.size(0).item() / 2

        x = x.view(channels, 1, -1)
        for i in range(k):
            v = x[i]
            sm = self.softmax(v)

