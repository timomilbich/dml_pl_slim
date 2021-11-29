from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from utils.auxiliaries import instantiate_from_config
import batchminer as bmine


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size//2
        self.wrap = wrap

        ## Gather dataset configs
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_datasampler = None
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_datasampler = None
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_datasampler = None

        ## Init datasets
        self._setup()

        ## Init dataloaders
        if train is not None:
            ## Add datasampler if required
            if "data_sampler" in self.dataset_configs["train"].keys():
                self.train_datasampler = self._add_datasampler(dataset="train")
            self.train_dataloader = self._train_dataloader

        if validation is not None:
            ## Add datasampler if required
            if "data_sampler" in self.dataset_configs["validation"].keys():
                self.val_datasampler = self._add_datasampler(dataset="validation")
            self.val_dataloader = self._val_dataloader

        if test is not None:
            ## Add datasampler if required
            if "data_sampler" in self.dataset_configs["test"].keys():
                self.test_datasampler = self._add_datasampler(dataset="test")
            self.test_dataloader = self._test_dataloader

    def _setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"],
                          batch_size=self.batch_size if self.train_datasampler is None else 1,
                          num_workers=self.num_workers,
                          batch_sampler=self.train_datasampler,
                          shuffle=self.train_datasampler is None)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size if self.val_datasampler is None else 1,
                          num_workers=self.num_workers,
                          batch_sampler=self.val_datasampler)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"],
                          batch_size=self.batch_size if self.test_datasampler is None else 1,
                          num_workers=self.num_workers,
                          batch_sampler=self.test_datasampler)

    def _add_datasampler(self, dataset):
        config_datasampler = self.dataset_configs[dataset]["data_sampler"]
        config_datasampler["params"]['batch_size'] = self.batch_size
        config_datasampler["params"]['image_dict'] = self.datasets[dataset].dataset.image_dict
        config_datasampler["params"]['image_list'] = self.datasets[dataset].dataset.image_list

        return instantiate_from_config(config_datasampler)



    # def prepare_data(self):
    #     for data_cfg in self.dataset_configs.values():
    #         instantiate_from_config(data_cfg)