import os
from pathlib import Path
from typing import Union

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class KeyPointDataset(Dataset):

    def __init__(self, dataset: Union[str, Path], imgsz: list):
        super().__init__()
        self.image_size = imgsz
        self.obj_paths = []
        for obj_name in os.listdir(dataset):
            obj_path = Path(dataset) / obj_name
            self.obj_paths.append(obj_path)
        # TODO

    def __getitem__(self, index):
        obj_path = self.obj_paths[index]
        obj = Image.open(obj_path)
        w, h = self.image_size[0], self.image_size[1]
        obj = obj.resize((w, h))
        obj = obj.convert('RGB')
        obj = transforms.ToTensor()(obj)
        print(obj_path)
        return obj
        # TODO

    def __len__(self):
        # TODO
        return len(self.obj_paths)
