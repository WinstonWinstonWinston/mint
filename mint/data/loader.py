from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig

from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader

def make_meta_collate(meta_keys):
    def collate_with_meta(data_list):
        # Take meta from the first element in the batch
        ref = data_list[0]
        meta = {k: getattr(ref, k) for k in meta_keys if hasattr(ref, k)}

        # Standard PyG batching, but excluding these meta keys from collation
        batch = Batch.from_data_list(data_list, exclude_keys=meta_keys)

        # Attach a single copy of each meta field to the batch
        for k, v in meta.items():
            setattr(batch, k, v)

        return batch

    return collate_with_meta


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

        self.collate_fn = make_meta_collate(train_dataset.meta_keys)

    def train_dataloader(self, shuffle=True):
        batch_size = self.batch_size.train
        num_workers = self.num_workers
        dl = DataLoader(
            self._train_dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True,
            collate_fn = self.collate_fn
        )
        return dl
    

    def val_dataloader(self):
        batch_size = self.batch_size.valid
        num_workers = self.num_workers
        dl =  DataLoader(
            self._valid_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=2,
            persistent_workers=True,
            drop_last=True,
            collate_fn = self.collate_fn
        )
        return dl

    def test_dataloader(self):
        batch_size=self.batch_size.test
        num_workers = self.num_workers
        dl =  DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.prefetch_factor,
            persistent_workers=True,
            drop_last=True,
            collate_fn = self.collate_fn
        )
        return dl