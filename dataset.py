import os
from pathlib import Path
from typing import Union

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json

from augmentor import Augmentor
from utils import plot_images


class KeyPointDataset(Dataset):

    def __init__(self, dataset: Union[str, Path], imgsz: list, mode: str, augment: int):
        super().__init__()
        image_path = Path(dataset) / 'images'
        # label_path = Path(dataset) / 'labels'
        self.image_size = imgsz
        self.mode = mode
        self.augment = augment
        self.obj_paths = []
        # self.lbl_paths = []

        for obj_name in os.listdir(image_path):
            obj_path = image_path / obj_name
            self.obj_paths.append(obj_path)
        # for lbl_name in os.listdir(label_path):
        #     lbl_path = label_path / lbl_name
        #     self.lbl_paths.append(lbl_path)
        # TODO

    def __getitem__(self, index: int):
        obj_path = self.obj_paths[index]
        obj = cv2.imread(obj_path.__str__())
        obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        ow, oh = obj.shape[1], obj.shape[0]
        w, h = self.image_size[0], self.image_size[1]
        obj = cv2.resize(obj, (w, h))
        # obj = transforms.ToTensor()(obj)
        # TODO

        if self.mode != 'test':

            lbl_path = obj_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')
            f = open(lbl_path, 'r')
            dic = json.load(f)
            f.close()
            points = [[shape['points'][0][1] / oh * h, shape['points'][0][0] / ow * w] for shape in dic['shapes']]

            if self.augment:

                augmentor = Augmentor()
                obj, points = augmentor(obj, points, self.augment)

            else:

                obj = torch.tensor([obj])

            target = torch.tensor(points)

        else:

            target = torch.tensor([])

        obj = torch.tensor(obj, dtype=torch.float32)
        obj = torch.permute(obj, (0, 3, 1, 2))

        return obj, target

    def __len__(self):
        # TODO
        return len(self.obj_paths)
