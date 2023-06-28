from torch import nn

from common import Resnet, Deconv, Conv


class CommonBlock(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv = Conv(cin, cout=64, k=7, s=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Backbone(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.conv = Conv(3, 64, 3, 1, 0)
        self.resnet = Resnet(depth)
        self.deconv = Deconv(64, 512, 1, 1, 0, 0)

    def forward(self, x):
        rn = self.conv(x)
        dc = self.deconv(rn)
        return dc


class KeyResnet(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.common_block = CommonBlock(cin=3)

    def forward(self, x):
        x = self.common_block(x)
        return x

