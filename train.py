import argparse
from pathlib import Path
from typing import Union

from torch import nn
from torch.utils.data import DataLoader

from dataset import KeyPointDataset
from models.resnet import Backbone

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def train(
        model: nn.Module,
        loaded_set: DataLoader
):
    # TODO

    for i, inputs in enumerate(loaded_set):
        print('train todo')
        outputs = model(inputs)


def load(dataset: Union[str, Path]):
    r"""
    Loads data.
    :param Union[str, Path] dataset: path of the dataset from which to load the data.
    """
    # TODO
    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    train_set = KeyPointDataset(absolute_set)
    loaded_train_set = DataLoader(train_set)
    return loaded_train_set


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset')
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--depth', default=18, help='depth of Resnet, 18, 34, 50, 101, 152')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    dataset = opt.data
    epochs = opt.epochs
    depth = opt.depth

    model = Backbone(depth)
    loaded_set = load(dataset)
    for epoch in range(0, epochs):
        train(model, loaded_set)
    # TODO
    print('run todo')
