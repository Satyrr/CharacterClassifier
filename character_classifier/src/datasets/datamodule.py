import io
import os
import zipfile
from typing import Optional, Tuple

import requests
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from character_classifier.settings import PROJECT_DIR, DATA_URL
from character_classifier.src.datasets.dataset import BatchIndexedDataset


class CharactersDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = PROJECT_DIR / 'data',
            train_val_test_split: Tuple[int, int, int] = (22_634, 2_500, 5_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.data_path = self.data_dir / 'train.pkl'

        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transforms = transforms.Compose(
            [transforms.Normalize((0.1556,), (0.363,))]

        )

        # self.dims is returned when you call datamodule.size()
        self.dims = (1, 56, 56)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_path):
            download_link = DATA_URL
            r = requests.get(download_link)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        dataset = BatchIndexedDataset(self.data_path, transform=self.transforms)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset, self.train_val_test_split,
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
