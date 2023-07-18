import torch
from torch import nn
from torch.nn import functional as F


class ArgSoftmaxDecider:
    r"""
    Select each heatmap's arg-softmax point as a keypoint.
    """
    def __init__(self, imgsz):
        self.image_size = imgsz
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, **kwargs):
        inputs = kwargs.get('inputs')
        batch_size = inputs.size(0)
        channels = inputs.size(1)
        height = inputs.size(2)
        width = inputs.size(3)

        if channels % 2 != 0:
            raise Exception('Channels of heatmaps don\'t equal uncertainty maps')
        k = int(channels / 2)

        inputs = inputs.view(batch_size, channels, 1, -1)

        m = []
        for i in range(batch_size):
            xyvs = []
            xi = inputs[i]
            for j in range(k):
                h = xi[j]
                v = xi[j + k]
                w = self.softmax(h)
                p = torch.arange(end=w.size(1)).to(w.device)
                ki = torch.sum(w * p)
                co_x = ki % width / width * self.image_size[0]
                co_y = (ki - ki % width) / width / height * self.image_size[1]
                vi = torch.sum(w * v)
                xyv = torch.stack((co_x, co_y, vi))
                xyvs.append(xyv)
            xyvs = torch.stack(xyvs)
            m.append(xyvs)
        m = torch.stack(m)

        return m


class GridBasedDecider:
    def __init__(self, keypoints, imgsz, grids):
        r"""
        Constructor of GridBasedDecider.
        :param keypoints: the number of keypoints to achieve
        :param imgsz: size of images in form of [width, height]
        :param grids: the number of grids along x-axis and y-axis
        """
        self.keypoints = keypoints
        self.image_size = imgsz
        self.grids = grids
        self.softmax = nn.Softmax(dim=3)

    def __call__(self, **kwargs):
        inputs = kwargs.get('inputs')
        batch_size = inputs.size(0)
        channels = inputs.size(1)
        height = inputs.size(2)
        width = inputs.size(3)

        height_padding = (self.grids - height % self.grids) / 2
        width_padding = (self.grids - width % self.grids) / 2
        inputs = F.pad(inputs, (int(height_padding), int(height_padding), int(width_padding), int(width_padding)), 'constant', 0)

        height = inputs.size(2)
        width = inputs.size(3)
        grid_height = height / self.grids
        grid_width = width / self.grids

        filters = torch.ones(int(batch_size), int(channels), int(grid_height), int(grid_width))
        grid_inputs = F.conv2d(inputs, weight=filters, stride=(int(grid_height), int(grid_width)))
        grid_inputs = grid_inputs.view(batch_size, channels, 1, -1)
        topk_indices = torch.topk(grid_inputs, self.keypoints, sorted=False)

        # TODO

        return grid_inputs


class GravitationDecider:
    def __init__(self, keypoints, imgsz):
        self.keypoints = keypoints
        self.image_size = imgsz

        self.distance = nn.PairwiseDistance(2)

    def __call__(self, **kwargs):
        target = kwargs.get('target')
        inputs = kwargs.get('inputs')
        batch_size = inputs.size(0)
        height = inputs.size(2)
        width = inputs.size(3)
        inputs = inputs.view(batch_size, height, width)

        targetn = torch.div(target, torch.tensor(self.image_size))
        target = torch.round(target)

        yns = torch.linspace(0, 1, height).view(-1, 1).repeat(1, width)
        xns = torch.linspace(0, 1, width).repeat(height, 1)
        yxns = torch.cat((yns.unsqueeze(2), xns.unsqueeze(2)), dim=2)
        yxns = yxns.repeat([self.keypoints] + [1 for _ in range(yxns.dim())])
        yxns = yxns.repeat([batch_size] + [1 for _ in range(yxns.dim())])

        for i in range(batch_size):
            for j in range(self.keypoints):
                yxns[i][j] = yxns[i][j] - targetn[i][j]
                k_y, k_x = int(target[i][j][0]), int(target[i][j][1])
                for y in range(height):
                    for x in range(width):
                        force = inputs[i][k_y][k_x] * inputs[i][y][x] * torch.pow(torch.pow(yxns[i][j][y][x][0], 2) + torch.pow(yxns[i][j][y][x][1], 2), 2/3)
                        yxns[i][j][y][x][0] = yxns[i][j][y][x][0] * force
                        yxns[i][j][y][x][1] = yxns[i][j][y][x][1] * force

        return inputs
