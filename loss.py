import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from keydeciders import OrdinaryDecider


class DistanceLoss:
    def __init__(self, norm: float = 2.0):
        r"""
        Compute the pairwise distance between input vectors.
        :param norm: the norm degree. 1.0 results in Manhattan Distance; 2.0 results in Euclidean Distance.
        """
        super(DistanceLoss, self).__init__()
        self.distance = nn.PairwiseDistance(p=norm)

    def __call__(self, pred, target):

        for i in range(target.size(0)):  # batch

            distance_matrix = []
            ta, pr = target[i], pred[i]

            for j in range(pr.size(0)):  # target point

                distance_vector = []
                p = pr[j]

                for k in range(ta.size(0)):  # pred point

                    t = ta[k]
                    distance = self.distance(p, t)
                    distance_vector.append(distance.item())

                distance_matrix.append(distance_vector)

            row, col = linear_sum_assignment(distance_matrix)
            target[i] = ta[col]

        dis = self.distance(pred, target)

        return torch.mean(dis)


class EdgeLoss:
    def __init__(self, views: int = 1):

        super(EdgeLoss, self).__init__()

        self.views = views

    def __call__(self, pred):

        assert pred.size(0) % self.views == 0

        groups = pred.size(0) // self.views

        total = torch.tensor(0., device=pred.device)

        for i in range(groups):  # group

            matrices = pred[i * self.views: (i + 1) * self.views, :, :]
            matrices = torch.softmax(matrices, dim=-1)
            std = torch.std(matrices, dim=0)
            std = torch.mean(std)
            total += std

        avg = total / groups

        return avg


class TypeLoss:
    def __init__(self):

        super(TypeLoss, self).__init__()

        self.bce = nn.BCELoss()

    def __call__(self, pred, target):

        loss = self.bce(pred, target)

        return torch.mean(loss)


class LossComputer:
    def __init__(self, **kwargs):
        keypoints = kwargs.get('keypoints')
        imgsz = kwargs.get('imgsz')
        grids = kwargs.get('grids')
        views = kwargs.get('views')
        self.views = views
        # self.key_decider = ArgSoftmaxDecider(imgsz)
        # self.key_decider = GridBasedDecider(keypoints, imgsz, grids)
        # self.key_decider = GravitationDecider(keypoints, imgsz)
        self.key_decider = OrdinaryDecider(imgsz)

        self.distance_loss = DistanceLoss(norm=2.0)
        self.edge_loss = EdgeLoss(views=views)
        self.type_loss = TypeLoss()

    def __call__(self, pred, transformed_pred, target_types):

        ltype = self.type_loss(pred, target_types)
        ltype2 = self.type_loss(transformed_pred, target_types)

        loss = ltype + ltype2

        return loss
