from torch import nn

from models.common import KeyResnet, KeyDecider


class KeyPointNet(nn.Module):
    def __init__(self, depth, imgsz, heatmaps, visualize=False):
        super().__init__()
        self.resnet = KeyResnet(depth, heatmaps, visualize)
        self.decider = KeyDecider(imgsz)

    def forward(self, x):
        x = self.resnet(x)
        x = self.decider(x)
        return x
