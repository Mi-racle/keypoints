import os
from pathlib import Path

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
    def __init__(self, cin, mid, cout, s=1, residual=True):
        super().__init__()
        self.residual = residual
        self.conv = Conv(cin, mid, k=3, s=s)
        self.conv2 = Conv(mid, cout, k=3, s=1, act=False)
        self.down_sample = Conv(cin, cout, k=1, s=s, act=False)
        self.act = nn.ReLU()

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
    def __init__(self, depth, keypoints, visualize=False):
        super().__init__()
        self.keypoints = keypoints
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
        # self.deconv = Deconv(cin=resnet['couts'][3], cout=256, k=3, s=2, p=1, pout=1)
        # self.deconv2 = Deconv(cin=256, cout=256, k=3, s=2, p=1, pout=1)
        # self.deconv3 = Deconv(cin=256, cout=256, k=3, s=2, p=1, pout=1)
        # for ArgSoftmaxDecider
        # self.final_layer = nn.Conv2d(in_channels=256, out_channels=keypoints * 2, kernel_size=1, stride=1, padding=1)
        # for GridBasedDecider
        # self.attention = nn.MultiheadAttention(12, 4, batch_first=True)
        self.penultimate_layer = nn.Conv2d(in_channels=resnet['couts'][2], out_channels=1, kernel_size=1, padding=1)
        self.final_layer = nn.Conv2d(in_channels=resnet['couts'][3], out_channels=1, kernel_size=1, padding=1)
        # self.penultimate_layer = Conv(resnet['couts'][2], 1, p=1)
        # self.final_layer = Conv(resnet['couts'][3], 1, p=1)
        self.fc = nn.Linear(144, keypoints)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.common_block(x)
        x = self.layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        p = self.layer4(x)
        # x = self.deconv(x)
        # x = self.deconv2(x)
        # x = self.deconv3(x)
        p = self.final_layer(p)

        if self.visualize:
            file = Path(__file__).resolve()
            root = file.parents[0].parents[0]
            if not os.path.exists(root / 'heatmaps'):
                os.mkdir(root / 'heatmaps')
            dst_path = increment_path(root / 'heatmaps/heatmap.jpg')
            draw_heatmap(4, 4, x.detach().numpy(), dst_path)

        p = p.view(p.size(0), p.size(1), -1).contiguous()
        p = self.fc(p)
        p = self.sigmoid(p)
        p = p.transpose(1, 2).contiguous()

        x = self.penultimate_layer(x)
        x = self.sigmoid(x)

        return x, p

    @staticmethod
    def _make_layer(module, cin, mid, cout, repeats):
        layers = [module(cin, mid, cout, 2, False)]
        for i in range(1, repeats):
            layers.append(module(cout, mid, cout))
        return nn.Sequential(*layers)
