from torch import nn


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred, target):
        # TODO
        return pred
