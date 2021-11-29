import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
import random



"""======================================================"""
REQUIRES_STORAGE = False

###
class Sampler(torch.utils.data.sampler.Sampler):
    """
    Plugs into PyTorch Batchsampler Package.
    """
    def __init__(self, image_dict, image_list, batch_size, samples_per_class=2, internal_split=1):

        #####
        self.image_dict         = image_dict
        self.image_list         = image_list

        #####
        self.internal_split = internal_split
        self.use_meta_split = self.internal_split!=1
        self.classes        = list(self.image_dict.keys())
        self.tv_split       = int(len(self.classes)*self.internal_split)
        self.train_classes  = self.classes[:self.tv_split]
        self.val_classes    = self.classes[self.tv_split:]

        ####
        self.batch_size         = batch_size
        self.samples_per_class  = samples_per_class
        self.sampler_length     = len(image_list)//batch_size
        assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'

        self.name             = 'class_random_sampler'
        self.requires_storage = False

    def __iter__(self):
        for _ in range(self.sampler_length):
            subset = []
            ### Random Subset from Random classes
            if self.use_meta_split:
                train_draws = int((self.batch_size//self.samples_per_class)*self.internal_split)
                val_draws   = self.batch_size//self.samples_per_class-train_draws
            else:
                train_draws = self.batch_size//self.samples_per_class
                val_draws   = None

            for _ in range(train_draws):
                class_key = random.choice(self.train_classes)
                class_ix_list = [random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)]
                subset.extend(class_ix_list)

            if self.use_meta_split:
                for _ in range(val_draws):
                    class_key = random.choice(self.val_classes)
                    subset.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            yield subset

    def __len__(self):
        return self.sampler_length
