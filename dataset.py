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
            views: int = 1
    ):

        super().__init__()

        image_path = Path(dataset)

        if os.path.exists(Path(dataset) / 'images'):
            image_path = (Path(dataset) / 'images')

        self.image_size = imgsz
        self.mode = mode
        self.augment = augment
        self.views = views
        self.obj_paths_group = []

        view_count = 0
        obj_paths = []

        for obj_name in os.listdir(image_path):

            obj_path = image_path / obj_name
            obj_paths.append(obj_path)
            view_count += 1

            if view_count >= self.views:
                self.obj_paths_group.append(obj_paths)
                obj_paths = []
                view_count = 0

    def __getitem__(self, index: int):

        obj_paths = self.obj_paths_group[index]

        assert self.views == len(obj_paths)

        obj_list = []
        points_list = []

        for i in range(self.views):

            obj_path = obj_paths[i]
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

        augmentor = Augmentor(views=self.views)

        if self.augment:

            obj_lists, points_lists = augmentor(obj_list, points_list, self.augment)

        else:

            obj_lists, points_lists = [obj_list], [points_list]

        points_lists = np.array(points_lists)
        points_lists = torch.tensor(points_lists)

        obj_lists = np.array(obj_lists)
        obj_lists = torch.tensor(obj_lists, dtype=torch.float32)
        obj_lists = torch.permute(obj_lists, (0, 1, 4, 2, 3))

        return obj_lists, points_lists

    def __len__(self):

        return len(self.obj_paths_group)


class AnimalDataset(Dataset):

    def __init__(self, dataset: Union[str, Path], imgsz: list, mode: str, augment: int):

        super().__init__()

        image_path = Path(dataset) / 'images'
        self.image_size = imgsz
        self.mode = mode
        self.augment = augment
        self.obj_paths = []

        for obj_name in os.listdir(image_path):
            obj_path = image_path / obj_name
            self.obj_paths.append(obj_path)

        fin = open(Path(dataset) / 'keypoints.json', 'r')
        self.labels = json.load(fin)
        fin.close()

    def __getitem__(self, index: int):

        obj_path = self.obj_paths[index]
        obj = cv2.imread(obj_path.__str__())
        obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        ow, oh = obj.shape[1], obj.shape[0]
        w, h = self.image_size[0], self.image_size[1]
        obj = cv2.resize(obj, (w, h))

        if self.mode != 'test':

            keypoints = self.labels['annotations'][index]['keypoints']
            points = [[keypoint[1] / oh * h, keypoint[0] / ow * w] for keypoint in keypoints]

            if self.augment:

                augmentor = Augmentor()
                obj, points = augmentor(obj, points, self.augment)

            else:

                obj = [obj]

            target = torch.tensor(points)

        else:

            target = torch.tensor([])

        obj = torch.tensor(obj, dtype=torch.float32)
        obj = torch.permute(obj, (0, 3, 1, 2)) if self.mode != 'test' else torch.permute(obj, (2, 0, 1))

        return obj, target

    def __len__(self):

        return len(self.obj_paths)
