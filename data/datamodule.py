import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
from data.dataset import AgentFormerDataset


class AgentFormerDataModule(pl.LightningDataModule):
    def __init__(self, cfg, args):#batch_size):#
        super().__init__()
        self.cfg = cfg
        # self.batch_size = batch_size
        self.args = args

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def get_dataloader(self, mode):
        phase = 'testing' if 'val' in mode or 'test' in mode or 'sanity' in mode else 'training'
        ds = AgentFormerDataset(self.cfg, split=mode, phase=phase)
        shuffle = False if 'val' in mode or 'test' in mode or 'sanity' in mode else True
        dataloader = DataLoader(ds, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                pin_memory=True, collate_fn=ds.collate, shuffle=shuffle, drop_last=shuffle)
        return dataloader

    def train_dataloader(self):
        return self.get_dataloader('train')

    def val_dataloader(self):
        return self.get_dataloader('test')

    def test_dataloader(self):
        return self.get_dataloader('test')
