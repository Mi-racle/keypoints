import argparse
import os
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KeyPointDataset
from keydeciders import GravitationDecider, OrdinaryDecider
from models.common import KeyResnet
from utils import ROOT, plot_images, increment_path


def detect(
        device: torch.device,
        model: nn.Module,
        loaded_set: DataLoader,
        key_decider,
        output_dir: Union[str, Path]
):

    for i, (inputs, target) in tqdm(enumerate(loaded_set), desc='Detect: ', total=len(loaded_set)):

        inputs, target = inputs.to(device), target.to(device)
        # [batch size, augment, views, 3, height, width] -> [batch size * augment * views, 3, height, width]
        inputs = inputs.view(inputs.size(0) * inputs.size(1) * inputs.size(2), inputs.size(3), inputs.size(4), inputs.size(5))
        # pred: (batched heatmaps, batched keypoints, batched edge matrices)
        pred = model(inputs)
        bkeypoints = key_decider(inputs=pred[1])
        plot_images(inputs, bkeypoints, output_dir)
        # TODO


def parse_opt(known=False):

    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', default=ROOT / 'logs' / 'train39' / 'best.pt')
    parser.add_argument('--data', default=ROOT / 'datasets/testset4/test')
    parser.add_argument('--batchsz', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--depth', default=152, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--keypoints', default=16, type=int, help='the number of keypoints')
    parser.add_argument('--grids', default=16, type=int)
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels')
    parser.add_argument('--mode', default='test', type=str, help='val or test')
    parser.add_argument('--views', default=1, type=int)

    opt = parser.parse_known_args()[0] if known else parser.parse_args()

    return opt


def run():

    opt = parse_opt()
    weights = opt.weights
    dataset = opt.data
    batch_size = opt.batchsz
    device = 'cpu' if not torch.cuda.is_available() or opt.device == 'cpu' else 'cuda:' + str(opt.device)
    device = torch.device(device)
    depth = opt.depth
    keypoints = opt.keypoints
    grids = opt.grids
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]
    mode = opt.mode
    views = opt.views

    model = KeyResnet(depth, keypoints, visualize)
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)

    absolute_set = dataset if Path(dataset).is_absolute() else ROOT / dataset
    data = KeyPointDataset(absolute_set, imgsz, mode, views=views)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)

    key_decider = OrdinaryDecider(imgsz)

    if not os.path.exists(ROOT / 'logs'):
        os.mkdir(ROOT / 'logs')
    output_dir = increment_path(ROOT / 'logs' / 'detect')

    detect(device, model, loaded_set, key_decider, output_dir)

    print(f'\033[92mResults saved to {output_dir}\033[0m')


if __name__ == '__main__':

    run()
