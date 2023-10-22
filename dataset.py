import json
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentor import Augmentor


class KeyPointDataset(Dataset):

    def __init__(
            self,
            dataset: Union[str, Path],
            imgsz: list,
            mode: str,
            augment: int = 0,
            views: int = 1,
            type_num: int = 13,
    ):

        super().__init__()

        image_path = Path(dataset)

        if os.path.exists(Path(dataset) / 'images'):
            image_path = (Path(dataset) / 'images')

        self.image_size = imgsz
        self.mode = mode
        self.augment = augment
        self.views = views
        self.type_num = type_num
        self.obj_paths = [image_path / obj_name for obj_name in sorted(os.listdir(image_path))]

    def __getitem__(self, index: int):

        paths = [self.obj_paths[index]]

        start = index // self.views * self.views

        for i in range(start, start + self.views):

            if i != index:

                paths.append(self.obj_paths[i])

        assert self.views == len(paths)

        obj_list = []
        points_list = []
        label_seq_list = []

        for i in range(self.views):

            obj_path = paths[i]
            obj = cv2.imread(obj_path.__str__())
            obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
            ow, oh = obj.shape[1], obj.shape[0]
            w, h = self.image_size[0], self.image_size[1]
            obj = cv2.resize(obj, (w, h))
            obj_list.append(obj)

            if self.mode != 'test':

                lbl_path = obj_path.parents[1] / 'labels' / (os.path.splitext(obj_path.name)[0] + '.json')
                f = open(lbl_path, 'r')
                dic = json.load(f)
                f.close()
                points = [[shape['points'][0][1] / oh * h, shape['points'][0][0] / ow * w] for shape in dic['shapes']]

                points_list.append(points)

                label_seq = [0 for _ in range(self.type_num)]
                label_seq[dic['type']] = 1
                label_seq_list.append(label_seq)

        augmentor = Augmentor(views=self.views)

        if self.augment:

            obj_lists, points_lists = augmentor(obj_list, points_list, self.augment)
            label_seq_lists = [label_seq_list for _ in range(self.augment)]

        else:

            obj_lists, points_lists, label_seq_lists = [obj_list], [points_list], [label_seq_list]

        points_lists = np.array(points_lists)
        points_lists = torch.tensor(points_lists)

        obj_lists = np.array(obj_lists)
        obj_lists = torch.tensor(obj_lists, dtype=torch.float32)
        obj_lists = torch.permute(obj_lists, (0, 1, 4, 2, 3))

        label_seq_lists = np.array(label_seq_lists)
        label_seq_lists = torch.tensor(label_seq_lists, dtype=torch.float32)

        return obj_lists, (points_lists, label_seq_lists)

    def __len__(self):

        return len(self.obj_paths)
