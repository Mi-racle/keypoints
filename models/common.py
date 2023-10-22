import os
from pathlib import Path

import torch
from torch import nn
import torch_geometric.nn as gnn
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


class GCNConv(nn.Module):
    # Standard convolution
    def __init__(self, cin, cout, act=True):  # ch_in, ch_out

        super().__init__()

        self.conv = gnn.GraphConv(cin, cout, bias=False)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x, edge_index):

        return self.act(self.conv(x, edge_index))

    def forward_fuse(self, x, edge_index):

        return self.act(self.conv(x, edge_index))


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

    def __init__(self, cin, mid, cout, s=1, residual=True):

        super().__init__()

        self.residual = residual
        self.conv = Conv(cin, mid, k=3, s=s)
        self.conv2 = Conv(mid, cout, k=3, s=1, act=False)
        self.down_sample = Conv(cin, cout, k=1, s=s, act=False)
        self.act = nn.ReLU()
        # self.act = nn.SiLU()

    def forward(self, x):

        residual = x
        out = self.conv(x)
        out = self.conv2(out)

        if self.residual:

            if residual.size() != out.size():

                residual = self.down_sample(residual)
                out += residual

        return self.act(out)


class Bottleneck(nn.Module):

    def __init__(self, cin, mid, cout, s=1, residual=True):

        super().__init__()

        self.residual = residual
        self.conv = Conv(cin, mid, k=1, s=s)
        self.conv2 = Conv(mid, mid, k=3, s=1)
        self.conv3 = Conv(mid, cout, k=1, s=1, act=False)
        self.down_sample = Conv(cin, cout, k=1, s=s, act=False)
        self.act = nn.ReLU()
        # self.act = nn.SiLU()

    def forward(self, x):

        residual = x
        out = self.conv(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.residual:

            if residual.size() != out.size():

                residual = self.down_sample(residual)

            out += residual

        return self.act(out)


class KeyResnet(nn.Module):

    def __init__(self, depth, views=1, type_num=1, visualize=False):

        super().__init__()

        self.views = views
        self.visualize = visualize
        self.resnets = {
            18: {
                'module': BasicBlock,
                'cins': [64, 64, 128, 256],
                'mids': [64, 128, 256, 512],
                'couts': [64, 128, 256, 512],
                'repeats': [2, 2, 2, 2]
            },
            34: {
                'module': BasicBlock,
                'cins': [64, 64, 128, 256],
                'mids': [64, 128, 256, 512],
                'couts': [64, 128, 256, 512],
                'repeats': [3, 4, 6, 3]
            },
            50: {
                'module': Bottleneck,
                'cins': [64, 256, 512, 1024],
                'mids': [64, 128, 256, 512],
                'couts': [256, 512, 1024, 2048],
                'repeats': [3, 4, 6, 3]
            },
            101: {
                'module': Bottleneck,
                'cins': [64, 256, 512, 1024],
                'mids': [64, 128, 256, 512],
                'couts': [256, 512, 1024, 2048],
                'repeats': [3, 4, 23, 3]
            },
            152: {
                'module': Bottleneck,
                'cins': [64, 256, 512, 1024],
                'mids': [64, 128, 256, 512],
                'couts': [256, 512, 1024, 2048],
                'repeats': [3, 8, 36, 3]
            }
        }
        self.common_block = CommonBlock(3, 64)
        resnet = self.resnets.get(depth)

        if resnet is None:

            raise Exception('Value of \'depth\' is not valid.')

        self.layer = self._make_layer(resnet['module'], resnet['cins'][0], resnet['mids'][0], resnet['couts'][0], resnet['repeats'][0])
        self.layer2 = self._make_layer(resnet['module'], resnet['cins'][1], resnet['mids'][0], resnet['couts'][1], resnet['repeats'][1])
        self.layer3 = self._make_layer(resnet['module'], resnet['cins'][2], resnet['mids'][0], resnet['couts'][2], resnet['repeats'][2])
        self.layer4 = self._make_layer(resnet['module'], resnet['cins'][3], resnet['mids'][0], resnet['couts'][3], resnet['repeats'][3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet['couts'][3] * 2, type_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        assert x.size(0) % self.views == 0

        x = self.common_block(x)
        x = self.layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.visualize:

            file = Path(__file__).resolve()
            root = file.parents[0].parents[0]

            if not os.path.exists(root / 'heatmaps'):

                os.mkdir(root / 'heatmaps')

            dst_path = increment_path(root / 'heatmaps/heatmap.jpg')
            draw_heatmap(4, 4, x.detach().numpy(), dst_path)

        # [batch size, resnet['couts'][3], height, width]

        out = []

        for i in range(x.size(0)):

            t = torch.concat(
                [
                    x[i],
                    torch.sum(x[i + 1: i + self.views, :, :, :], dim=0)
                ],
                dim=0
            )

            out.append(t)

        x = torch.stack(out, dim=0)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

    @staticmethod
    def _make_layer(module, cin, mid, cout, repeats):

        layers = [module(cin, mid, cout, 2, False)]

        for i in range(1, repeats):

            layers.append(module(cout, mid, cout))

        return nn.Sequential(*layers)


class Classifier(nn.Module):

    def __init__(self, type_num, features):

        super().__init__()

        self.conv = GCNConv(features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.fc = nn.Linear(64, type_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch):

        x = self.conv(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = gnn.global_mean_pool(x, batch)
        x = self.fc(x)
        x = self.softmax(x)[0]

        return x

