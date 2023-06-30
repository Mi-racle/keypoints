import torch
from torch import nn

from common import Resnet, Deconv, Conv


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
    def __init__(self, depth):
        super().__init__()
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
        self.final_layer = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1, stride=1, padding=1)

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
        return x

    @staticmethod
    def _make_layer(module, cin, cout, repeats, s=1):
        layers = [module(cin, cout, s)]
        for i in range(1, repeats):
            layers.append(module(cout, cout))
        return nn.Sequential(*layers)

