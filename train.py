import argparse
import os

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from loss import LossComputer
from models.common import KeyResnet
from utils import load_dataset, log_epoch, ROOT, increment_path


def train(
        device: torch.device,
        model: nn.Module,
        loaded_set: DataLoader,
        loss_computer: LossComputer,
        optimizer: Optimizer
):
    total_loss = 0

    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        inputs, target = inputs.to(device), target.to(device)
        # pred size: [batch_size, heatmaps, 3], 3 means [xi, yi, vi]
        pred = model(inputs)
        loss = loss_computer(pred, target)
        total_loss += loss.item()
        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loaded_set)

    return average_loss


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
    parser.add_argument('--batchsz', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--depth', default=34, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--keypoints', default=16, type=int, help='the number of keypoints, which uncertainty maps equal')
    parser.add_argument('--grids', default=16, type=int)
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    dataset = opt.data
    batch_size = opt.batchsz
    device = opt.device if not torch.cuda.is_available() or opt.device == 'cpu' else 'cuda:' + str(opt.device)
    device = torch.device(device)
    epochs = opt.epochs
    depth = opt.depth
    keypoints = opt.keypoints
    grids = opt.grids
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]

    model = KeyResnet(depth, keypoints, visualize)
    # model.load_state_dict(torch.load('best.pt'))
    model.to(device)
    loaded_set = load_dataset(dataset, batch_size, imgsz)
    loss_computer = LossComputer(keypoints=keypoints, imgsz=imgsz, grids=grids)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    logger = Logger(increment_path(ROOT / 'logs' / 'train'))
    for epoch in range(0, epochs):
        print(f'Epoch {epoch}:')
        model.train()
        loss = train(device, model, loaded_set, loss_computer, optimizer)
        log_epoch(logger, epoch, model, loss, 0)

    # TODO
    print('run todo')


if __name__ == '__main__':
    run()
