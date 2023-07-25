import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from keydeciders import GravitationDecider
from models.common import KeyResnet
from utils import load_dataset, ROOT


def detect(
        device: torch.device,
        model: nn.Module,
        loaded_set: DataLoader,
        key_decider
):
    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        inputs, target = inputs.to(device), target.to(device)
        pred = model(inputs)
        keypoints = key_decider(inputs=pred, mode='detect')
        # TODO
        pass


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
    parser.add_argument('--batchsz', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--depth', default=34, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--heatmaps', default=16, type=int, help='the number of heatmaps, which uncertainty maps equal')
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
    depth = opt.depth
    heatmaps = opt.heatmaps
    grids = opt.grids
    visualize = opt.visualize
    image_size = opt.imgsz
    image_size = [image_size[0], image_size[0]] if len(image_size) == 1 else image_size[0: 2]
    model = KeyResnet(depth, heatmaps, visualize)
    model.load_state_dict(torch.load(ROOT / 'pts' / 'best.pt'))
    model.to(device)
    loaded_set = load_dataset(dataset, batch_size, image_size)
    key_decider = GravitationDecider(heatmaps, image_size)

    detect(device, model, loaded_set, key_decider)


if __name__ == '__main__':
    run()
