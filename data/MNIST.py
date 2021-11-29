import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import string
from typing import Callable
from torch.utils.data import Dataset

class MNISTDATA(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(
            self,
            root,
            train = True,
            transform = None,
            target_transform = None,
            img_dim = 28,
            ):

        super(MNISTDATA, self).__init__()

        self.train = train  # training set or test set
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_dim = img_dim
        self.root = root

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets = torch.load(os.path.join(self.root, data_file))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.unsqueeze(0)/255

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.root,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.root,
                                            self.test_file)))

