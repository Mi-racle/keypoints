from pathlib import Path

import torch
from torch import nn
from torchvision import models

from utils import draw_heatmap, increment_path


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


class CommonBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = Conv(cin, cout, k=7, s=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.conv = Conv(cin, cout, k=3, s=s)
        self.conv2 = Conv(cout, cout, k=3, s=1, act=False)
        self.down_sample = Conv(cin, cout, k=1, s=s, act=False)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.conv2(out)
        if residual.size() != out.size():
            residual = self.down_sample(residual)
        out += residual
        return self.act(out)


class Bottleneck(nn.Module):
    def __init__(self, cin, cout, s=1):
        super().__init__()
        expansion = 4
        self.conv = Conv(cin, cout, k=1, s=1)
        self.conv2 = Conv(cout, cout, k=3, s=s)
        self.conv3 = Conv(cout, cout * expansion, k=1, s=1, act=False)
        self.down_sample = Conv(cin, cout * expansion, k=1, s=s, act=False)
        self.act = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if residual.size() != out.size():
            residual = self.down_sample(residual)
        out += residual
        return self.act(out)


class KeyResnet(nn.Module):
    def __init__(self, depth, heatmaps, visualize):
        super().__init__()
        self.visualize = visualize
        self.resnets = {
            18: {
                'module': BasicBlock,
                'cins': [64, 64, 128, 256],
                'couts': [64, 128, 256, 512],
                'repeats': [2, 2, 2, 2]
            },
            34: {
                'module': BasicBlock,
                'cins': [64, 64, 128, 256],
                'couts': [64, 128, 256, 512],
                'repeats': [3, 4, 6, 3]
            },
            50: {
                'module': BasicBlock,
                'cins': [64, 256, 512, 1024],
                'couts': [64, 128, 256, 512],
                'repeats': [3, 4, 6, 3]
            },
            101: {
                'module': BasicBlock,
                'cins': [64, 256, 512, 1024],
                'couts': [64, 128, 256, 512],
                'repeats': [3, 4, 23, 3]
            },
            152: {
                'module': BasicBlock,
                'cins': [64, 256, 512, 1024],
                'couts': [64, 128, 256, 512],
                'repeats': [3, 8, 36, 3]
            }
        }
        self.common_block = CommonBlock(3, 64)
        resnet = self.resnets.get(depth)
        if resnet is None:
            raise Exception('Value of \'depth\' is not valid.')
        self.layer = self._make_layer(resnet['module'], resnet['cins'][0], resnet['couts'][0], resnet['repeats'][0])
        self.layer2 = self._make_layer(resnet['module'], resnet['cins'][1], resnet['couts'][1], resnet['repeats'][1])
        self.layer3 = self._make_layer(resnet['module'], resnet['cins'][2], resnet['couts'][2], resnet['repeats'][2])
        self.layer4 = self._make_layer(resnet['module'], resnet['cins'][3], resnet['couts'][3], resnet['repeats'][3])
        self.deconv = Deconv(
            cin=resnet['couts'][3] * (1 if resnet['module'] is BasicBlock else 4),
            cout=256,
            k=3,
            s=2,
            p=1,
            pout=1,
        )
        self.deconv2 = Deconv(cin=256, cout=256, k=3, s=2, p=1, pout=1)
        self.deconv3 = Deconv(cin=256, cout=256, k=3, s=2, p=1, pout=1)
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=heatmaps * 2, kernel_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.common_block(x)
        x = self.layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.deconv(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.final_layer(x)

        if self.visualize:
            file = Path(__file__).resolve()
            root = file.parents[0].parents[0]
            dst_path = increment_path(root / 'heatmaps/heatmap.jpg')
            draw_heatmap(4, 4, x.detach().numpy(), dst_path)

        return x

    @staticmethod
    def _make_layer(module, cin, cout, repeats, s=1):
        layers = [module(cin, cout, s)]
        for i in range(1, repeats):
            layers.append(module(cout, cout))
        return nn.Sequential(*layers)


class KeyDecider(nn.Module):
    def __init__(self, imgsz):
        super().__init__()
        self.image_size = imgsz
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
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
            xyv = []
            xi = x[i]
            for j in range(k):
                h = xi[j]
                v = xi[j + k]
                w = self.softmax(h)
                p = torch.arange(end=w.size(1))
                ki = round(torch.sum(w * p).item())
                co_x = ki % width / width * self.image_size[0]
                co_y = ki // width / height * self.image_size[1]
                vi = torch.sum(w * v)
                xyv.append([co_x, co_y, vi])
            m.append(xyv)

        return torch.Tensor(m)
