from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig

class MINTDatamodule(LightningDataModule):
    """
    TODO: Document me
    """

    def __init__(self, *, num_workers:int, prefetch_factor: int, batch_size: DictConfig,
                 train_dataset, valid_dataset, test_dataset):
        super().__init__()

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.batch_size = batch_size

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset

    def train_dataloader(self, shuffle=True):
        batch_size = self.batch_size.train
        num_workers = self.num_workers
        return DataLoader(
            self._train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True
        )

    def val_dataloader(self):
        batch_size = self.batch_size.train
        num_workers = self.num_workers
        return DataLoader(
            self._valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True
        )

    def test_dataloader(self):
        batch_size=self.batch_size.test
        num_workers = self.num_workers
        return DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.prefetch_factor,
            persistent_workers=True,
            drop_last=True
        )