import glob
import re
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

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


def log_epoch(logger, epoch, loss, accuracy):
    # 1. Log scalar values (scalar summary)
    info = {
        'loss': loss,
        'accuracy': accuracy
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
