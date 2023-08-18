import torch
from torch import nn

from keydeciders import GravitationDecider, OrdinaryDecider
from utils import make_graph, kuhn_kunkres


class DistanceLoss:
    def __init__(self, norm: float = 1.0):
        r"""
        Compute the pairwise distance between input vectors.
        :param norm: the norm degree. 1.0 results in Manhattan Distance; 2.0 results in Euclidean Distance.
        """
        super(DistanceLoss, self).__init__()
        self.distance = nn.PairwiseDistance(p=norm)

    def __call__(self, pred, target):
        dis = self.distance(pred, target)

        return torch.mean(dis)


class GravitationLoss:
    def __init__(self, imgsz):
        super(GravitationLoss, self).__init__()
        self.distance = nn.PairwiseDistance(p=2.0)
        self.image_size = imgsz

    def __call__(self, pred, heatmap):

        keypoints = pred.size(1)
        batch_size = heatmap.size(0)
        height = heatmap.size(-2)
        width = heatmap.size(-1)
        inputs = heatmap.view(batch_size, height * width)

        yns = torch.linspace(0, self.image_size[0], height).view(-1, 1).repeat(1, width)
        xns = torch.linspace(0, self.image_size[1], width).repeat(height, 1)
        yxns = torch.cat((yns.unsqueeze(2), xns.unsqueeze(2)), dim=2).view(height * width, 2)

        bkforces = []
        for i in range(batch_size):
            kforces = []
            for j in range(keypoints):
                vectors = yxns - pred[i][j]  # vectors: (height * width, 2)
                vectors = torch.mul(
                    vectors,
                    torch.mul(
                        torch.pow(
                            torch.sum(
                                torch.pow(
                                    vectors,
                                    2
                                ),
                                -1
                            ),
                            -3 / 2
                        ),
                        inputs[i]
                    )
                    .view(-1, 1)
                    .repeat(1, 2)
                )
                vectors = torch.sum(vectors, 0)
                kforces.append(vectors)
            kforces = torch.stack(kforces)
            bkforces.append(kforces)
        bkforces = torch.stack(bkforces)  # bkforces: (batch_size, self.keypoints, 2)

        deviation = self.distance(bkforces, torch.zeros_like(pred))

        return torch.sum(deviation)


class LossComputer:
    def __init__(self, **kwargs):
        keypoints = kwargs.get('keypoints')
        imgsz = kwargs.get('imgsz')
        grids = kwargs.get('grids')
        # self.key_decider = ArgSoftmaxDecider(imgsz)
        # self.key_decider = GridBasedDecider(keypoints, imgsz, grids)
        # self.key_decider = GravitationDecider(keypoints, imgsz)
        self.key_decider = OrdinaryDecider(imgsz)

        self.distance_loss = DistanceLoss(norm=2.0)
        self.gravitation_loss = GravitationLoss(imgsz)

    def __call__(self, pred, target):
        # pred = self.key_decider(inputs=pred, target=target, mode='train')
        heatmap = pred[0]
        keypoints = pred[1]
        keypoints = self.key_decider(inputs=keypoints)

        ldis = self.distance_loss(keypoints, target)
        lgra = self.gravitation_loss(keypoints, heatmap)

        loss = ldis + 400 * lgra

        return loss
