import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from augmentor import Augmentor
from dataset import KeyPointDataset, AnimalDataset
from keydeciders import OrdinaryDecider
from logger import Logger
from loss import LossComputer, DistanceLoss, TypeLoss
from models.common import KeyResnet, Classifier
from utils import log_epoch, ROOT, increment_path, make_graphs, get_minimum_spanning_trees


def train(
        device: torch.device,
        model: nn.Module,
        classifier: nn.Module,
        loaded_set: DataLoader,
        loss_computer: LossComputer,
        optimizer: Optimizer,
        augmentor: Augmentor,
        views: int,
):

    total_loss = 0

    for i, (inputs, targets) in tqdm(enumerate(loaded_set), desc='Train: ', total=len(loaded_set)):

        targets, label_seqs = targets

        # [batch size, augment, views, 3, height, width] -> [batch size * augment * views, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
        # [batch size, augment, views, keypoints, 2] -> [batch size * augment * views, keypoints, 2]
        targets = targets.view(targets.size(0) * targets.size(1) * targets.size(2), targets.size(3), targets.size(4))

        label_seqs = label_seqs.view(label_seqs.size(0) * label_seqs.size(1) * label_seqs.size(2), label_seqs.size(3))

        ndarray_inputs = torch.permute(inputs, (0, 2, 3, 1)).numpy()
        transformed_inputs = []
        transformed_targets = []

        for j in range(ndarray_inputs.shape[0]):

            ndarray_input, target = ndarray_inputs[j], targets[j]
            ndarray_input, target = augmentor([ndarray_input], [target], 1)
            ndarray_input, target = ndarray_input[0][0], target[0][0]
            transformed_inputs.append(ndarray_input)
            transformed_targets.append(target)

        transformed_inputs, transformed_targets = np.array(transformed_inputs), np.array(transformed_targets)
        transformed_inputs, transformed_targets = torch.tensor(transformed_inputs), torch.tensor(transformed_targets)
        transformed_inputs = torch.permute(transformed_inputs, (0, 3, 1, 2))

        inputs, targets = inputs.to(device), targets.to(device)
        transformed_inputs, transformed_targets = transformed_inputs.to(device), transformed_targets.to(device)

        # pred: (batched heatmaps, batched keypoints, batched edge matrices)
        pred = model(inputs)

        transformed_pred = model(transformed_inputs)

        edge_matrices = pred[1].view(-1, views, pred[1].size(1), pred[1].size(2))
        edge_matrices = torch.softmax(edge_matrices, dim=-1)
        edge_matrices = torch.mean(edge_matrices, dim=1, keepdim=True)
        edge_matrices = edge_matrices.repeat(edge_matrices.size(0), views, edge_matrices.size(2), edge_matrices.size(3))
        edge_matrices = edge_matrices.view(-1, edge_matrices.size(2), edge_matrices.size(3))

        graphs = make_graphs(edge_matrices)
        trees = get_minimum_spanning_trees(graphs)
        edge_seqs = []

        for tree in trees:

            edge_seq = []

            for edge in sorted(tree.edges(data=True)):

                edge_seq.append(edge[0])
                edge_seq.append(edge[1])

            edge_seqs.append(edge_seq)

        edge_seqs = torch.tensor(edge_seqs, device=device)
        label_seqs = label_seqs.to(device)

        pred_types = classifier(edge_seqs, label_seqs)

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
        loaded_set: DataLoader,
        key_decider,
        views: int,
):

    for i, (inputs, targets) in tqdm(enumerate(loaded_set), desc='Val: ', total=len(loaded_set)):

        targets, label_seqs = targets

        inputs, targets, label_seqs = inputs.to(device), targets.to(device), label_seqs.to(device)
        # [batch size, augment, views, 3, height, width] -> [batch size * augment * views, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
        # [batch size, augment, views, keypoints, 2] -> [batch size * augment * views, keypoints, 2]
        targets = targets.view(targets.size(0) * targets.size(1) * targets.size(2), targets.size(3), targets.size(4))

        label_seqs = label_seqs.view(label_seqs.size(0) * label_seqs.size(1) * label_seqs.size(2), label_seqs.size(3))
        # pred: (batched heatmaps, batched keypoints, batched edge matrices)
        pred = model(inputs)
        # bkeypoints = key_decider(inputs=pred[1])
        # acc = DistanceLoss(norm=2.0)(bkeypoints, targets)
        # acc = acc.item()

        edge_matrices = pred[1].view(-1, views, pred[1].size(1), pred[1].size(2))
        edge_matrices = torch.softmax(edge_matrices, dim=-1)
        edge_matrices = torch.mean(edge_matrices, dim=1, keepdim=True)
        edge_matrices = edge_matrices.repeat(edge_matrices.size(0), views, edge_matrices.size(2), edge_matrices.size(3))
        edge_matrices = edge_matrices.view(-1, edge_matrices.size(2), edge_matrices.size(3))

        graphs = make_graphs(edge_matrices)
        trees = get_minimum_spanning_trees(graphs)
        edge_seqs = []

        for tree in trees:

            edge_seq = []

            for edge in sorted(tree.edges(data=True)):
                edge_seq.append(edge[0])
                edge_seq.append(edge[1])

            edge_seqs.append(edge_seq)

        edge_seqs = torch.tensor(edge_seqs, device=device)

        tgt = [[0] for _ in range(edge_seqs.size(0))]
        tgt = torch.tensor(tgt, device=device)

        pred_types = classifier(edge_seqs, tgt)

        label_seqs = label_seqs.float()
        acc = TypeLoss()(pred_types, label_seqs).item()

        return acc


def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default=ROOT / 'datasets/testset6/train', type=str)
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
    parser.add_argument('--augment', default=0, type=int, help='0 for no augmenting while positive int for augment')
    parser.add_argument('--views', default=4, type=int, help='number of multi views')
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
    grids = opt.grids
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

    classifier = Classifier(keypoints - 1, type_num)
    classifier = classifier.to(device)

    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    data = KeyPointDataset(absolute_set, imgsz, 'train', augment, views, type_num)
    # data = AnimalDataset(absolute_set, imgsz, 'train', augment)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    absolute_valid_set = (dataset if Path(dataset).is_absolute() else ROOT / dataset).parents[0] / 'valid'
    valid_data = KeyPointDataset(absolute_valid_set, imgsz, 'valid', type_num=type_num)
    loaded_valid_set = DataLoader(dataset=valid_data, batch_size=batch_size)

    key_decider = OrdinaryDecider(imgsz)

    loss_computer = LossComputer(keypoints=keypoints, imgsz=imgsz, grids=grids, views=views)

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
        loss = train(device, model, classifier, loaded_set, loss_computer, optimizer, augmentor, views)

        model.eval()
        acc = val(device, model, classifier, loaded_valid_set, key_decider, views)

        log_epoch(logger, epoch, model, classifier, loss, acc, best_acc)

        if acc < best_acc:

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
