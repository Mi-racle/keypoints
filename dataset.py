from pathlib import Path
from typing import Union

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class KeyPointDataset(Dataset):

    def __init__(self, dataset: Union[str, Path]):
        super().__init__()
        # TODO
        print(dataset)

    def __getitem__(self, index):
        # TODO
        pass

    def __len__(self):
        # TODO
        return 0
