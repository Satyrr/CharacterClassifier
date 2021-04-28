import math
import pickle
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset


class BatchIndexedDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        self.transform = transform
        self.x, self.y = pickle.load(open(data_path, 'rb'))

        batch_size, image_len = self.x.shape
        img_size = int(math.sqrt(image_len))

        self.x = self.x.reshape(batch_size, 1, img_size, img_size).astype(float)

        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

    def __getitem__(self, index: torch.LongTensor):
        x, y = self.x[index], self.y[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.y)
