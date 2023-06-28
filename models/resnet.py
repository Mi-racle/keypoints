from torch import nn

from common import Resnet, Deconv, Conv


class BasicBlock(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv = Conv(cin, 64, 7, 2)


class Backbone(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.conv = Conv(3, 64, 3, 1, 0)
        self.resnet = Resnet(depth)
        self.deconv = Deconv(2048, 512, 1, 1, 0, 0)

    def forward(self, x):
        rn = self.conv(x)
        dc = self.deconv(rn)
        return dc
