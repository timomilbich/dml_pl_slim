import os
import os.path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .basic_dml_dataset import BaseDataset
import pickle as pkl
import os


class DATA(Dataset):
    """`SOP dataset.

    Args:
        root (string): Root directory of data.
        train (bool, optional): Specifies type of data split (train or validation).
        arch (string, optional): Type of network architecture used for training, influences choice of transformations
            applied when sampling batches.

    """

    def __init__(
            self,
            root,
            train = True,
            arch = 'resnet50',
            ooDML_split_id=-1,
    ):

        super(DATA, self).__init__()

        self.train = train  # training set or test set
        self.root = "/export/home/karoth/Datasets/online_products/" if root is None else root
        self.n_classes = 11318 # number of train classes
        self.path_ooDML_splits = "/export/home/tmilbich/PycharmProjects/dml_pl/data/ooDML_splits/online_products_splits.pkl"

        if ooDML_split_id > -1:
            raise Exception('ooDML data splits are currently not implemented!')

        image_sourcepath = self.root + '/images'
        training_files = pd.read_table(self.root + '/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
        test_files = pd.read_table(self.root + '/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

        spi = np.array([(a, b) for a, b in zip(training_files['super_class_id'], training_files['class_id'])])
        super_dict = {}
        super_conversion = {}
        for i, (super_ix, class_ix, image_path) in enumerate(
                zip(training_files['super_class_id'], training_files['class_id'], training_files['path'])):
            if super_ix not in super_dict: super_dict[super_ix] = {}
            if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
            super_dict[super_ix][class_ix].append(image_sourcepath + '/' + image_path)
        train_image_dict = super_dict

        ####
        test_image_dict = {}
        train_image_dict_temp = {}
        super_train_image_dict = {}
        train_conversion = {}
        test_conversion = {}
        super_test_conversion = {}

        ## Create Training Dictionaries
        i = 0
        for super_ix, super_set in train_image_dict.items():
            super_ix -= 1
            counter = 0
            super_train_image_dict[super_ix] = []
            for class_ix, class_set in super_set.items():
                class_ix -= 1
                super_train_image_dict[super_ix].extend(class_set)
                train_image_dict_temp[class_ix] = class_set
                if class_ix not in train_conversion:
                    train_conversion[class_ix] = class_set[0].split('/')[-1].split('_')[0]
                    super_conversion[class_ix] = class_set[0].split('/')[-2]
                counter += 1
                i += 1
        train_image_dict = train_image_dict_temp

        ## Create Test Dictioniaries
        for class_ix, img_path in zip(test_files['class_id'], test_files['path']):
            class_ix = class_ix - 1
            if not class_ix in test_image_dict.keys():
                test_image_dict[class_ix] = []
            test_image_dict[class_ix].append(image_sourcepath + '/' + img_path)
            test_conversion[class_ix] = img_path.split('/')[-1].split('_')[0]
            super_test_conversion[class_ix] = img_path.split('/')[-2]

        ###
        if self.train:
            train_dataset = BaseDataset(train_image_dict, arch)
            train_dataset.conversion = train_conversion
            self.dataset = train_dataset
            print(f'DATASET:\ntype: SOP\nSetup: Train\n#Classes: {len(train_image_dict)}')
        else:
            test_dataset = BaseDataset(test_image_dict, arch, is_validation=True)
            test_dataset.conversion = test_conversion
            self.dataset = test_dataset
            print(f'DATASET:\ntype: SOP\nSetup: Val\n#Classes: {len(test_image_dict)}\n')

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


    def __len__(self):
        return len(self.dataset)


    def load_oodDML_split(self, split_id=1):
        split_dict = pkl.load(open(self.path_ooDML_splits, 'rb'))
        train_classes, test_classes, fid = split_dict[split_id]['train'], split_dict[split_id]['test'], split_dict[split_id]['fid']
        return train_classes, test_classes, fid

