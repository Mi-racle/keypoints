import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from dataset import KeyPointDataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def train():
    # TODO
    print('train todo')
    loaded_train_set = load('dataset')
    for i, inputs in enumerate(loaded_train_set):
        print('loaded todo')


def load(dataset: str):
    r"""
    Loads data.

    :param str dataset: name of the dataset from which to load the data.
    """
    # TODO
    train_set = KeyPointDataset()
    loaded_train_set = DataLoader(train_set)
    return loaded_train_set


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'data')
    parser.add_argument('--epochs', default=100)
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    # TODO
    print('run todo')
