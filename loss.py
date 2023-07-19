import torch
from torch import nn

from keydeciders import ArgSoftmaxDecider, GridBasedDecider, GravitationDecider


class DistanceLoss(nn.Module):
    def __init__(self, norm: float = 1.0):
        r"""
        Compute the pairwise distance between input vectors.
        :param norm: the norm degree. 1.0 results in Manhattan Distance; 2.0 results in Euclidean Distance.
        """
        super(DistanceLoss, self).__init__()
        self.distance = nn.PairwiseDistance(p=norm)

    def forward(self, pred, target):
        dis = self.distance(pred, target)

        return torch.mean(dis)


class GravitationLoss(nn.Module):
    def __init__(self, norm: float = 2.0):
        super(GravitationLoss, self).__init__()
        self.distance = nn.PairwiseDistance(p=norm)

    def forward(self, pred):
        dis = self.distance(pred, torch.zeros_like(pred))

        return torch.sum(dis)


class LossComputer:
    def __init__(self, **kwargs):
        keypoints = kwargs.get('keypoints')
        imgsz = kwargs.get('imgsz')
        grids = kwargs.get('grids')
        # self.key_decider = ArgSoftmaxDecider(imgsz)
        # self.key_decider = GridBasedDecider(keypoints, imgsz, grids)
        self.key_decider = GravitationDecider(keypoints, imgsz)

        # self.distance_loss = DistanceLoss()
        self.gravitation_loss = GravitationLoss()

    def __call__(self, pred, target):
        pred = self.key_decider(inputs=pred, target=target)
        # keypoints = pred[:, :, 0: 2]
        # ldis = self.distance_loss(keypoints, target)
        lgra = self.gravitation_loss(pred)
        return lgra
