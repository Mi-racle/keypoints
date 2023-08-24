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
        inputs = F.pad(inputs, (int(height_padding), int(height_padding), int(width_padding), int(width_padding)),
                       'constant', 0)

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
        mode = kwargs.get('mode')
        batch_size = inputs.size(0)
        height = inputs.size(2)
        width = inputs.size(3)
        inputs = inputs.view(batch_size, height * width)

        yns = torch.linspace(0, 1, height).view(-1, 1).repeat(1, width)
        xns = torch.linspace(0, 1, width).repeat(height, 1)
        yxns = torch.cat((yns.unsqueeze(2), xns.unsqueeze(2)), dim=2).view(height * width, 2)

        if mode == 'train':

            targetn = torch.div(target, torch.tensor(self.image_size))
            target = torch.round(torch.mul(targetn, torch.tensor([height, width])))

            bkforces = []
            for i in range(batch_size):
                kforces = []
                for j in range(self.keypoints):
                    k_y, k_x = int(target[i][j][0]), int(target[i][j][1])
                    k_v = inputs[i][k_y * width + k_x]
                    vectors = yxns - targetn[i][j]  # vectors: (height * width, 2)
                    vectors = torch.mul(
                        vectors,
                        torch
                        .mul(
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
                                k_v
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

            return bkforces

        elif mode == 'detect':

            bkeypoints = []
            for i in range(batch_size):
                all_forces = []
                for j in range(height * width):
                    v = inputs[i][j]
                    vectors = yxns - yxns[j]  # vectors: (height * width, 2)
                    # vectors[j] += float('inf')
                    tmp = torch.pow(
                        torch.sum(
                            torch.pow(
                                vectors,
                                2
                            ),
                            -1
                        ),
                        -3 / 2
                    )
                    tmp[j] = 0
                    vectors = torch.mul(
                        vectors,
                        torch
                        .mul(
                            torch.mul(
                                tmp,
                                v
                            ),
                            inputs[i]
                        )
                        .view(-1, 1)
                        .repeat(1, 2)
                    )
                    vectors = torch.sum(vectors, 0)
                    all_forces.append(vectors)
                all_forces = torch.stack(all_forces)
                all_forces = torch.sum(torch.pow(all_forces, 2), -1)
                _, bottomk = torch.topk(all_forces, self.keypoints, largest=False)
                kyns = (bottomk // width + 0.5) / height
                kxns = (bottomk % width + 0.5) / width
                keypoints = torch.stack([kyns, kxns])
                keypoints = torch.transpose(keypoints, 0, 1)
                keypoints = keypoints * torch.tensor(self.image_size, device=keypoints.device)
                bkeypoints.append(keypoints)

            bkeypoints = torch.stack(bkeypoints)

            return bkeypoints  # bkeypoints: (batch_size, self.keypoints, 2)

        else:
            raise Exception('Gravitation mode must be either \'train\' or \'detect\'!')


class OrdinaryDecider:
    def __init__(self, imgsz):
        self.image_size = imgsz

    def __call__(self, **kwargs):
        inputs = kwargs.get('inputs')
        return inputs * torch.tensor(self.image_size, device=inputs.device)
