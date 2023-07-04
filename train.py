import argparse
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import KeyPointDataset
from models.resnet import KeyResnet
from utils import draw_heatmap

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def train(
        device: str,
        model: nn.Module,
        loaded_set: DataLoader
):
    # TODO
    device = torch.device(device)
    model.to(device)
    for i, inputs in enumerate(loaded_set):
        outputs = model(inputs)
        print(outputs.size())
        print('train todo')
        draw_heatmap(4, 4, outputs.detach().numpy(), 'heatmaps/heatmap' + str(i) + '.jpg')


def load(dataset: Union[str, Path]):
    r"""
    Loads data.
    :param Union[str, Path] dataset: path of the dataset from which to load the data.
    """
    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    train_set = KeyPointDataset(absolute_set)
    loaded_train_set = DataLoader(train_set)
    return loaded_train_set


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--depth', default=18, help='depth of Resnet, 18, 34, 50, 101, 152')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    dataset = opt.data
    device = opt.device if opt.device == 'cpu' else 'cuda:' + str(opt.device)
    epochs = opt.epochs
    depth = opt.depth

    model = KeyResnet(depth)
    loaded_set = load(dataset)
    for epoch in range(0, epochs):
        train(device, model, loaded_set)
    # TODO
    print('run todo')
