from pathlib import Path
from typing import Union
from torchvision import transforms
import os

from PIL import Image
from torch.utils.data import Dataset


class KeyPointDataset(Dataset):

    def __init__(self, dataset: Union[str, Path]):
        super().__init__()
        self.obj_paths = []
        for obj_name in os.listdir(dataset):
            obj_path = Path(dataset) / obj_name
            self.obj_paths.append(obj_path)
        # TODO

    def __getitem__(self, index):
        obj_path = self.obj_paths[index]
        obj = Image.open(obj_path)
        obj = obj.convert('RGB')
        obj = transforms.ToTensor()(obj)
        return obj
        # TODO

    def __len__(self):
        # TODO
        return len(self.obj_paths)
