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

    def __call__(self, inputs):
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

    def __call__(self, inputs):
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


        return grid_inputs
