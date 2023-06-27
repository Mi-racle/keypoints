from torch import nn
from torchvision import models


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
