import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from keydeciders import GravitationDecider, OrdinaryDecider
from models.common import KeyResnet
from utils import load_dataset, ROOT, plot_image, increment_path


def detect(
        device: torch.device,
        model: nn.Module,
        loaded_set: DataLoader,
        key_decider
):
    output_dir = increment_path(ROOT / 'logs' / 'detect')

    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        inputs, target = inputs.to(device), target.to(device)
        pred = model(inputs)  # pred: (heatmaps, keypoints)
        bkeypoints = key_decider(inputs=pred[1])
        plot_image(inputs, bkeypoints, output_dir)
        # TODO


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=ROOT / 'logs' / 'train20' / 'best.pt')
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
    parser.add_argument('--batchsz', default=1, type=int)
    parser.add_argument('--device', default='cpu', help='cpu or 0 (cuda)')
    parser.add_argument('--depth', default=152, type=int, help='depth of Resnet, 18, 34, 50, 101, 152')
    parser.add_argument('--keypoints', default=16, type=int, help='the number of keypoints')
    parser.add_argument('--grids', default=16, type=int)
    parser.add_argument('--visualize', default=False, type=bool, help='visualize heatmaps or not')
    parser.add_argument('--imgsz', default=[640], type=int, nargs='+', help='pixels')
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
    image_size = opt.imgsz
    image_size = [image_size[0], image_size[0]] if len(image_size) == 1 else image_size[0: 2]
    model = KeyResnet(depth, keypoints, visualize)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    loaded_set = load_dataset(dataset, batch_size, image_size)
    key_decider = OrdinaryDecider(image_size)

    detect(device, model, loaded_set, key_decider)


if __name__ == '__main__':
    run()
