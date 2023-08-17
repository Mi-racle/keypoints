import glob
import os
import re
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import ImageDraw
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import KeyPointDataset

ROOT = FILE = Path(__file__).resolve().parents[0]


def load_dataset(
        dataset: Union[str, Path],
        batch_size: int,
        image_size: list[int]
):
    r"""
    Loads data.
    :param Union[str, Path] dataset: path of the dataset from which to load the data
    :param int batch_size: size of one batch
    :param list[int] image_size: size of the longer one of width and height
    """
    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    data = KeyPointDataset(absolute_set, image_size)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    return loaded_set


def draw_heatmap(width, height, x, save_name):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        # 8*8网格
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        # pmin = np.min(img)
        # pmax = np.max(img)
        # img = ((img - pmin) / (pmax - pmin + 1e-6)) * 255  # float在[0，1]之间，转换成0-255
        img *= 255
        img = img.astype(np.uint8)  # 转成unit8
        # 函数applycolormap产生伪彩色图像
        # COLORMAP_JET模式，就常被用于生成我们所常见的 热力图
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # print("{}/{}".format(i, width * height))
    fig.savefig(save_name, dpi=100, overwrite=True)
    fig.clf()
    plt.close()
    # print("time:{}".format(time.time() - tic))


def increment_path(dst_path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    dst_path = Path(dst_path)  # os-agnostic
    if dst_path.exists() and not exist_ok:
        suffix = dst_path.suffix
        dst_path = dst_path.with_suffix('')
        dirs = glob.glob(f"{dst_path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % dst_path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 1  # increment number
        dst_path = Path(f"{dst_path}{sep}{n}{suffix}")  # update path
    _dir = dst_path if dst_path.suffix == '' else dst_path.parent  # directory
    if not _dir.exists() and mkdir:
        _dir.mkdir(parents=True, exist_ok=True)  # make directory
    return dst_path


def plot_image(inputs, bkeypoints, path: Path):
    transformer = transforms.ToPILImage()

    if not os.path.exists(path):
        os.mkdir(path)

    batch_size = inputs.size(0)

    for i in range(batch_size):
        image = inputs[i]
        image = transformer(image)
        drawer = ImageDraw.Draw(image)

        keypoints = bkeypoints[i]
        for keypoint in keypoints:
            y, x = keypoint[0], keypoint[1]
            drawer.ellipse(((x - 5, y - 5), (x + 5, y + 5)), fill=(0, 255, 0))

        image.save(increment_path(path / 'image.jpg'))


def log_epoch(logger, epoch, model, loss, accuracy):
    # 1. Log scalar values (scalar summary)
    info = {
        'loss': loss,
        'accuracy': accuracy
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)

    logger.save_model('best.pt', model)

    print(f'Average loss in this epoch: {loss}')


def kuhn_kunkres(graph: Tensor):
    pred, match = linear_sum_assignment(graph.detach())
    total_distance = torch.sum(graph[pred, match])
    return torch.tensor(match), total_distance


def make_graph(pred: Tensor, target: Tensor):
    keypoints = pred.size(0)
    pred = torch.unsqueeze(pred, 0)
    pred = pred.repeat(keypoints, 1, 1)
    pred = pred.permute(1, 0, 2)
    vector = pred - target
    graph = torch.pow(vector, 2)
    graph = torch.sum(graph, -1)
    graph = torch.sqrt(graph)
    return graph
