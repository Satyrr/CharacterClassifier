from typing import Callable, Optional

from torch.utils.data import Dataset
import torch
import pickle


class BatchIndexedDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None):

        self.transform = transform
        self.x, self.y = pickle.load(open(data_path, 'rb'))

    def __getitem__(self, index: torch.LongTensor):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
