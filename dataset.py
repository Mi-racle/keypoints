import os
from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json


class KeyPointDataset(Dataset):

    def __init__(self, dataset: Union[str, Path], imgsz: list):
        super().__init__()
        image_path = Path(dataset) / 'images'
        # label_path = Path(dataset) / 'labels'
        self.image_size = imgsz
        self.obj_paths = []
        # self.lbl_paths = []

        for obj_name in os.listdir(image_path):
            obj_path = image_path / obj_name
            self.obj_paths.append(obj_path)
        # for lbl_name in os.listdir(label_path):
        #     lbl_path = label_path / lbl_name
        #     self.lbl_paths.append(lbl_path)
        # TODO

    def __getitem__(self, index):
        obj_path = self.obj_paths[index]
        obj = Image.open(obj_path)
        w, h = self.image_size[0], self.image_size[1]
        obj = obj.resize((w, h))
        obj = obj.convert('RGB')
        obj = transforms.ToTensor()(obj)
        # TODO

        lbl_path = obj_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')
        f = open(lbl_path, 'r')
        dic = json.load(f)
        f.close()
        points = [shape['points'][0] for shape in dic['shapes']]
        target = torch.tensor(points, requires_grad=True)

        return obj, target

    def __len__(self):
        # TODO
        return len(self.obj_paths)
