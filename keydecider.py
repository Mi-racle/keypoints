import torch
from torch import nn


class KeyDecider:
    def __init__(self, imgsz):
        self.image_size = imgsz
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, x):
        batch_size = x.size(0)
        channels = x.size(1)
        width = x.size(2)
        height = x.size(3)

        if channels % 2 != 0:
            raise Exception('Channels of heatmaps don\'t equal uncertainty maps')
        k = int(channels / 2)

        x = x.view(batch_size, channels, 1, -1)

        m = []
        for i in range(batch_size):
            xyvs = []
            xi = x[i]
            for j in range(k):
                h = xi[j]
                v = xi[j + k]
                w = self.softmax(h)
                p = torch.arange(end=w.size(1))
                ki = torch.sum(w * p)
                co_x = ki % width / width * self.image_size[0]
                co_y = (ki - ki % width) / width / height * self.image_size[1]
                vi = torch.sum(w * v)
                xyv = torch.stack((co_x, co_y, vi))
                xyvs.append(xyv)
            xyvs = torch.stack(xyvs)
            m.append(xyvs)
        m = torch.stack(m)

        return m
