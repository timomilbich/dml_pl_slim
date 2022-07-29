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

        # new nfs /export/home/pima/scratch/dataset/sop/
        self.train = train  # training set or test set
        self.root = "/export/home/tmilbich/Datasets/online_products/" if root is None else root
        self.n_classes = 11318 # number of train classes
        self.path_ooDML_splits = "/export/home/tmilbich/PycharmProjects/dml_pl/data/ooDML_splits/online_products_splits.pkl"
        image_sourcepath = self.root + '/images'
        training_files = pd.read_table(self.root + '/Info_Files/Ebay_train.txt', header=0, delimiter=' ')
        test_files = pd.read_table(self.root + '/Info_Files/Ebay_test.txt', header=0, delimiter=' ')

        # super_dict = {}
        super_conversion = {}
        class_dict = {}
        class_conversion = {}

        for i, (super_ix, class_ix, image_path) in enumerate(zip(training_files['super_class_id'], training_files['class_id'], training_files['path'])):
            # if super_ix not in super_dict: super_dict[super_ix] = {}
            # if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
            # super_dict[super_ix][class_ix].append(image_sourcepath + '/' + image_path)

            class_ix -= 1 # let class ids start with 0

            if class_ix not in class_dict: class_dict[class_ix] = []
            class_dict[class_ix].append(image_sourcepath + '/' + image_path)
            class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]

        for i, (super_ix, class_ix, image_path) in enumerate(zip(test_files['super_class_id'], test_files['class_id'], test_files['path'])):
            # if super_ix not in super_dict: super_dict[super_ix] = {}
            # if class_ix not in super_dict[super_ix]: super_dict[super_ix][class_ix] = []
            # super_dict[super_ix][class_ix].append(image_sourcepath + '/' + image_path)

            class_ix -= 1 # let class ids start with 0

            if class_ix not in class_dict: class_dict[class_ix] = []
            class_dict[class_ix].append(image_sourcepath + '/' + image_path)
            class_conversion[class_ix] = image_path.split('/')[-1].split('_')[0]

        classes = [val for key,val in class_conversion.items()]
        if ooDML_split_id == -1:
            ### Use the 11318 classes as training and the remaining classes as test data (official split)
            train, test = classes[:11318], classes[11318:]
            fid = -1
        else:
            ### load ooDML splits
            train, test, fid = self.load_oodDML_split(ooDML_split_id)

        train_image_dict = {key: item for key, item in class_dict.items() if str(class_conversion[key]) in train}
        test_image_dict = {key: item for key, item in class_dict.items() if str(class_conversion[key]) in test}
        train_conversion = {i: classname for i, classname in enumerate(train)}
        test_conversion = {i: classname for i, classname in enumerate(test)}

        ###
        if self.train:
            train_dataset = BaseDataset(train_image_dict, arch)
            train_dataset.conversion = train_conversion
            self.dataset = train_dataset
            print(f'Initializing Dataset:\n*** type: [SOP]\n ***Setup: [Train]\n*** #Classes: [{len(train_image_dict)}]')
            print(f'*** ooDML Data Split [{ooDML_split_id}] with FID: [{fid:.2f}]')
        else:
            test_dataset = BaseDataset(test_image_dict, arch, is_validation=True)
            test_dataset.conversion = test_conversion
            self.dataset = test_dataset
            print(f'Initializing Dataset:\n*** type: [SOP]\n*** Setup: [Val]\n*** #Classes: [{len(test_image_dict)}]\n')
            print(f'*** ooDML Data Split [{ooDML_split_id}] with FID: [{fid:.2f}]')


    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


    def __len__(self):
        return len(self.dataset)


    def load_oodDML_split(self, split_id=1):
        split_dict = pkl.load(open(self.path_ooDML_splits, 'rb'))
        train_classes, test_classes, fid = split_dict[split_id]['train'], split_dict[split_id]['test'], split_dict[split_id]['fid']
        return train_classes, test_classes, fid

