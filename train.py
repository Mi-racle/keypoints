import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KeyPointDataset
from logger import Logger
from loss import LossComputer
from models.common import KeyResnet
from utils import log_epoch, ROOT, increment_path


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
        # [batch size, augment, 3, height, width] -> [batch size * augment, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1), inputs.size(2), inputs.size(3), inputs.size(4))
        # [batch size, augment, keypoints, 2] -> [batch size * augment, keypoints, 2]
        target = target.view(target.size(0) * target.size(1), target.size(2), target.size(3))

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
    parser.add_argument('--batchsz', default=2, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--early-stopping', default=30, type=int)
    parser.add_argument('--depth', default=152, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--keypoints', default=16, type=int, help='the number of keypoints')
    parser.add_argument('--grids', default=16, type=int)
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels of width and height')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--augment', default=64, type=int, help='0 for no augmenting while positive int for multiple')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    dataset = opt.data
    batch_size = opt.batchsz
    device = 'cpu' if not torch.cuda.is_available() or opt.device == 'cpu' else 'cuda:' + str(opt.device)
    device = torch.device(device)
    epochs = opt.epochs
    early_stopping = opt.early_stopping
    depth = opt.depth
    keypoints = opt.keypoints
    grids = opt.grids
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]
    lr = opt.lr
    augment = opt.augment

    model = KeyResnet(depth, keypoints, visualize)
    model = model.to(device)

    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    data = KeyPointDataset(absolute_set, imgsz, 'train', augment)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)

    loss_computer = LossComputer(keypoints=keypoints, imgsz=imgsz, grids=grids)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    if not os.path.exists(ROOT / 'logs'):
        os.mkdir(ROOT / 'logs')
    output_dir = increment_path(ROOT / 'logs' / 'train')
    logger = Logger(output_dir)

    best_loss = float('inf')
    patience = 0

    for epoch in range(0, epochs):

        print(f'Epoch {epoch}:')
        model.train()
        loss = train(device, model, loaded_set, loss_computer, optimizer)
        log_epoch(logger, epoch, model, loss, best_loss, 0)

        if loss < best_loss:
            best_loss = loss
            patience = 0

        else:
            patience += 1

            if patience > early_stopping:
                break

    print(f'\033[92mResults have saved to {output_dir}\033[0m')


if __name__ == '__main__':
    run()
