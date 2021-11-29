import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random
import torch.distributed as dist


"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, image_dict, image_list, batch_size, samples_per_class=2, drop_last=False, num_replicas=None, rank=None):

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                num_replicas = dist.get_world_size()
            except:
                num_replicas = 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            try:
                rank = dist.get_rank()
            except:
                rank = 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        #####
        self.image_dict = image_dict
        self.image_list = image_list

        #####
        self.train_classes = list(self.image_dict.keys())

        ####
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.sampler_length_total = len(image_list) // batch_size
        self.sampler_length = self.sampler_length_total // self.num_replicas
        assert self.batch_size % self.samples_per_class == 0, '#Samples per class must divide batchsize!'

        self.name = 'class_random_sampler'
        self.requires_storage = False

        print(f"\nData sampler [{self.name}] initialized with rank=[{self.rank}/{self.num_replicas}] and sampler length=[{self.sampler_length}/{self.sampler_length_total}].\n")

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            train_draws = self.batch_size//self.samples_per_class

            for _ in range(train_draws):
                class_key = random.choice(self.train_classes)
                class_ix_list = [random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)]
                subset.extend(class_ix_list)

            yield subset

    def __len__(self):
        return self.sampler_length


    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
