from typing import Optional
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from sam.util import ConfigType

__all__ = ["DataModule"]

class DataModule(pl.LightningDataModule):
    def __init__(self, config: ConfigType, dataset_cls):
        super().__init__()

        self.config = config
        self.dataset_cls = dataset_cls
        self.datasets = {}

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_set = self.dataset_cls("train", self.config)
            self.val_set = self.dataset_cls("val", self.config)
        if stage == 'test' or stage is None:
            self.test_set = self.dataset_cls("test", self.config)

    def get_dataloader(self, dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_set, 
            self.config.data.train_batch_size,
            shuffle=True,
            num_workers=self.config.data.train_workers
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.val_set, 
            self.config.data.val_batch_size,
            shuffle=False,
            num_workers=self.config.data.val_workers
        )

    def test_dataloader(self):
        return self.get_dataloader(
            self.test_set, 
            self.config.data.test_batch_size,
            shuffle=False,
            num_workers=self.config.data.test_workers
        )