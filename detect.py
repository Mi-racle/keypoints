import argparse

import torch

from keydeciders import GravitationDecider
from models.common import KeyResnet
from train import ROOT


def detect(key_decider):
    return


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=ROOT / 'datasets/testset2')
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
    device = opt.device if not torch.cuda.is_available() or opt.device == 'cpu' else 'cuda:' + str(opt.device)
    device = torch.device(device)
    depth = opt.depth
    heatmaps = opt.heatmaps
    grids = opt.grids
    visualize = opt.visualize
    imgsz = opt.imgsz
    imgsz = [imgsz[0], imgsz[0]] if len(imgsz) == 1 else imgsz[0: 2]
    model = KeyResnet(depth, heatmaps, visualize)
    model.load_state_dict(torch.load('best.pt'))
    model.to(device)
    key_decider = GravitationDecider(heatmaps, imgsz)

    detect(key_decider)


if __name__=='__main__':
    run()
