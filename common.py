from torch import nn
from torchvision import models


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Deconv(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=None, pout=None, g=1, act=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, k, s, p, pout, g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU() if True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


class Resnet(nn.Module):
    def __init__(self, depth=50):
        """
        Initialize Resnet Module.

        :param int depth: The number of layers of Resnet. The following are available: 18, 34, 50, 101, 152
        """
        super().__init__()
        if depth == 18:
            self.resnet = models.resnet18()
        elif depth == 34:
            self.resnet = models.resnet34()
        elif depth == 50:
            self.resnet = models.resnet50()
        elif depth == 101:
            self.resnet = models.resnet101()
        elif depth == 152:
            self.resnet = models.resnet152()
        else:
            raise Exception('Value of \'depth\' is not valid.')

    def forward(self, x):
        return self.resnet(x)
