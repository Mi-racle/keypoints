import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentor import Augmentor
from dataset import KeyPointDataset
from logger import Logger
from loss import LossComputer, TypeLoss
from models.common import KeyResnet, Classifier
from utils import log_epoch, ROOT, increment_path, get_edge_seqs, get_transformed_data, get_node_features


def train(
        device: torch.device,
        model: nn.Module,
        classifier: nn.Module,
        loaded_set: DataLoader,
        loss_computer: LossComputer,
        optimizer: Optimizer,
        augmentor: Augmentor
):

    total_loss = 0

    for i, (inputs, targets) in tqdm(enumerate(loaded_set), desc='Train: ', total=len(loaded_set)):

        targets, label_seqs = targets
        views = inputs.size(2)

        # [batch size, augment, views, 3, height, width] -> [batch size * augment * views, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
        # [batch size, augment, views, keypoints, 2] -> [batch size * augment * views, keypoints, 2]
        targets = targets.view(targets.size(0) * targets.size(1) * targets.size(2), targets.size(3), targets.size(4))

        label_seqs = label_seqs.view(label_seqs.size(0) * label_seqs.size(1) * label_seqs.size(2), label_seqs.size(3))

        transformed_inputs, transformed_targets = get_transformed_data(inputs, targets, augmentor)

        inputs, targets = inputs.to(device), targets.to(device)
        transformed_inputs, transformed_targets = transformed_inputs.to(device), transformed_targets.to(device)

        # pred: (batched keypoints, batched edge matrices)
        pred = model(inputs)

        transformed_pred = model(transformed_inputs)

        node_features = pred[0]
        node_num = node_features.size(1)
        edge_seqs = get_edge_seqs(pred[1], views).to(device)

        label_seqs = label_seqs.to(device)

        pred_types = []

        for j in range(edge_seqs.size(0)):

            pred_type = classifier(node_features[j], edge_seqs[j], torch.stack([torch.argmax(label_seqs[j]) for _ in range(node_num)]))

            pred_types.append(pred_type)

        pred_types = torch.stack(pred_types)

        loss = loss_computer(pred, targets, transformed_pred, transformed_targets, pred_types, label_seqs)
        total_loss += loss.item()
        # print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loaded_set)

    return average_loss


def val(
        device: torch.device,
        model: nn.Module,
        classifier: nn.Module,
        loaded_set: DataLoader
):

    total, right = 0, 0

    for i, (inputs, targets) in tqdm(enumerate(loaded_set), desc='Val: ', total=len(loaded_set)):

        targets, label_seqs = targets
        views = inputs.size(2)

        inputs, targets, label_seqs = inputs.to(device), targets.to(device), label_seqs.to(device)
        # [batch size, augment, views, 3, height, width] -> [batch size * augment * views, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
        # [batch size, augment, views, keypoints, 2] -> [batch size * augment * views, keypoints, 2]
        targets = targets.view(targets.size(0) * targets.size(1) * targets.size(2), targets.size(3), targets.size(4))

        label_seqs = label_seqs.view(label_seqs.size(0) * label_seqs.size(1) * label_seqs.size(2), label_seqs.size(3))
        # pred: (batched keypoints, batched edge matrices)
        pred = model(inputs)
        # bkeypoints = key_decider(inputs=pred[1])
        # acc = DistanceLoss(norm=2.0)(bkeypoints, targets)
        # acc = acc.item()

        node_features = pred[0]
        node_num = node_features.size(1)
        edge_seqs = get_edge_seqs(pred[1], views).to(device)

        label_seqs = label_seqs.to(device)

        pred_types = []

        for j in range(edge_seqs.size(0)):

            pred_type = classifier(node_features[j], edge_seqs[j],
                                   torch.stack([torch.argmax(label_seqs[j]) for _ in range(node_num)]))

            total += 1

            if torch.argmax(pred_type).item() == torch.argmax(label_seqs[j]).item():

                right += 1

            pred_types.append(pred_type)

        pred_types = torch.stack(pred_types)

        label_seqs = label_seqs.float()
    # acc = TypeLoss()(pred_types, label_seqs).item()
    acc = right / total

    return acc


def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default=ROOT / 'datasets/testset3/train', type=str)
    parser.add_argument('--batchsz', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cpu or 0 (cuda)')
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--early-stopping', default=30, type=int)
    parser.add_argument('--depth', default=152, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--keypoints', default=16, type=int, help='the number of keypoints')
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels of width and height')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--augment', default=0, type=int, help='0 for no augmenting while positive int for augment')
    parser.add_argument('--views', default=2, type=int, help='number of multi views')
    parser.add_argument('--type-num', default=13, type=int)

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
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]
    lr = opt.lr
    augment = opt.augment
    batch_size = max(batch_size // max(augment, 1), 1)
    views = opt.views
    type_num = opt.type_num

    model = KeyResnet(depth, keypoints, visualize)
    model = model.to(device)

    classifier = Classifier(type_num, 2)
    classifier = classifier.to(device)

    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    data = KeyPointDataset(absolute_set, imgsz, 'train', augment, views, type_num)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)

    absolute_valid_set = (dataset if Path(dataset).is_absolute() else ROOT / dataset).parents[0] / 'valid'
    valid_data = KeyPointDataset(absolute_valid_set, imgsz, 'valid', type_num=type_num)
    loaded_valid_set = DataLoader(dataset=valid_data, batch_size=batch_size)

    loss_computer = LossComputer(keypoints=keypoints, imgsz=imgsz, views=views)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    augmentor = Augmentor()

    if not os.path.exists(ROOT / 'logs'):
        os.mkdir(ROOT / 'logs')
    output_dir = increment_path(ROOT / 'logs' / 'train')
    logger = Logger(output_dir)

    best_acc = float('inf')
    patience = 0

    for epoch in range(0, epochs):

        print(f'Epoch {epoch}:')
        model.train()
        loss = train(device, model, classifier, loaded_set, loss_computer, optimizer, augmentor)

        model.eval()
        acc = val(device, model, classifier, loaded_valid_set)

        log_epoch(logger, epoch, model, classifier, loss, acc, best_acc)

        if acc > best_acc:

            best_acc = acc
            patience = 0
            print(f'\033[92mBest accuracy achieved and patience reset to {early_stopping}\033[0m')

        else:

            patience += 1

            if patience > early_stopping:

                print(f'\033[92mPatience consumed and training early stopped\033[0m')
                break

            print(f'\033[92mPatience left: {early_stopping - patience}\033[0m')

    print(f'\033[92mResults saved to {output_dir}\033[0m')


if __name__ == '__main__':

    run()
