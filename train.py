import argparse
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import KeyPointDataset
from models.keypoint import KeyPointNet

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def train(
        device: str,
        model: nn.Module,
        loaded_set: DataLoader,
):
    # TODO
    device = torch.device(device)
    model.to(device)
    for i, inputs in enumerate(loaded_set):
        # output size: [batch_size, heatmaps, 3], 3 means [xi, yi, vi]
        outputs = model(inputs)
        print(outputs)
        print('train todo')


def load(
        dataset: Union[str, Path],
        imgsz: list
):
    r"""
    Loads data.
    :param Union[str, Path] dataset: path of the dataset from which to load the data.
    :param imgsz: size of the longer one of width and height
    """
    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    train_set = KeyPointDataset(absolute_set, imgsz)
    loaded_train_set = DataLoader(train_set)
    return loaded_train_set


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--depth', default=18, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--heatmaps', default=16, help='the number of heatmaps, which uncertainty maps equal')
    parser.add_argument('--visualize', default=True, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    dataset = opt.data
    device = opt.device if opt.device == 'cpu' else 'cuda:' + str(opt.device)
    epochs = opt.epochs
    depth = opt.depth
    heatmaps = opt.heatmaps
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]

    model = KeyPointNet(depth, imgsz, heatmaps, visualize)
    loaded_set = load(dataset, imgsz)
    for epoch in range(0, epochs):
        train(device, model, loaded_set)
    # TODO
    print('run todo')
