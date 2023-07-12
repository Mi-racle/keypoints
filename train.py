import argparse
import os
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KeyPointDataset
from loss import LossComputer
from models.common import KeyResnet
from models.keypoint import KeyPointNet

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
tt = torch.randn(1, 16, 2, requires_grad=True)

def train(
        device: str,
        model: nn.Module,
        loaded_set: DataLoader,
        loss_computer: LossComputer,
        optimizer: Optimizer
):
    # TODO
    device = torch.device(device)
    model.to(device)

    total_loss = 0

    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        # pred size: [batch_size, heatmaps, 3], 3 means [xi, yi, vi]
        pred = model(inputs)
        loss = loss_computer(pred, tt)
        total_loss += loss.item()
        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loaded_set)

    if not os.path.exists(ROOT / 'pts'):
        os.mkdir(ROOT / 'pts')

    log = open(ROOT / 'pts' / 'log.txt', 'a')
    log.write(f'{average_loss}\n')
    log.close()

    torch.save(model.state_dict(), ROOT / 'pts' / 'best.pt')

    print(f'Average loss in this epoch: {average_loss}')


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
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--depth', default=18, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--heatmaps', default=16, type=int, help='the number of heatmaps, which uncertainty maps equal')
    parser.add_argument('--visualize', default=False, help='visualize heatmaps or not')
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

    model = KeyResnet(depth, heatmaps, visualize)
    # model = KeyPointNet(depth, imgsz, heatmaps, visualize)
    # model.load_state_dict(torch.load('best.pt'))
    loaded_set = load(dataset, imgsz)
    loss_computer = LossComputer(imgsz)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(0, epochs):
        model.train()
        train(device, model, loaded_set, loss_computer, optimizer)
    # TODO
    print('run todo')
