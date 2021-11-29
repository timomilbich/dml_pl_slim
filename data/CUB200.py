import os
import os.path
import torch
from torch.utils.data import Dataset
from .basic_dml_dataset import BaseDataset
import pickle as pkl
import os


class DATA(Dataset):
    """`CUB200 dataset.

    Args:
        root (string): Root directory of data.
        train (bool, optional): Specifies type of data split (train or validation).
        arch (string, optional): Type of network architecture used for training, influences choice of transformations
            applied when sampling batches.

    """

    def __init__(
            self,
            root,
            train=True,
            arch='resnet50',
            ooDML_split_id=-1,
            ):

        super(DATA, self).__init__()

        self.train = train  # training set or test set
        self.root = "/export/home/tmilbich/Datasets/cub200/" if root is None else root
        self.n_classes = 100
        self.path_ooDML_splits = "/export/home/tmilbich/PycharmProjects/dml_pl/data/ooDML_splits/cub200_splits.pkl"

        image_sourcepath = self.root + '/images'
        image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x],
                               key=lambda x: int(x.split('.')[0]))
        total_conversion = {int(x.split('.')[0]) - 1: x for x in image_classes}
        reverse_total_conversion = {val: key for key, val in total_conversion.items()}
        image_list = {int(key.split('.')[0]) - 1: sorted(
            [image_sourcepath + '/' + key + '/' + x for x in os.listdir(image_sourcepath + '/' + key) if '._' not in x])
                      for key in image_classes}
        image_list = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]
        image_list = [x for y in image_list for x in y]

        ### Dictionary of structure class:list_of_samples_with_said_class
        image_dict = {}
        for key, img_path in image_list:
            if not key in image_dict.keys():
                image_dict[key] = []
            image_dict[key].append(img_path)

        keys = sorted(list(image_dict.keys()))
        if ooDML_split_id == -1:
            ### Use the first half of the sorted data as training and the second half as test set
            train, test = keys[:len(keys) // 2], keys[len(keys) // 2:]
            fid = -1
        else:
            ### load ooDML splits
            train, test, fid = self.load_oodDML_split(ooDML_split_id)
            # convert to class_ids
            train = {reverse_total_conversion[key] for key in train if key in image_classes} # -1 to since classnames start with 001.xxx
            test = {reverse_total_conversion[key] for key in test if key in image_classes}

        ###
        train_image_dict = {key: image_dict[key] for key in train}
        test_image_dict = {key: image_dict[key] for key in test}

        ###
        if self.train:
            train_dataset = BaseDataset(train_image_dict, arch)
            self.dataset = train_dataset
            print(f'DATASET:\ntype: CUB200\nSetup: Train\n#Classes: {len(train_image_dict)}')
            print(f'ooDML Data Split [{ooDML_split_id}] with FID: [{fid:.2f}]')
        else:
            test_dataset = BaseDataset(test_image_dict, arch, is_validation=True)
            self.dataset = test_dataset
            print(f'DATASET:\ntype: CUB200\nSetup: Val\n#Classes: {len(test_image_dict)}')
            print(f'ooDML Data Split [{ooDML_split_id}] with FID: [{fid:.2f}]\n')

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


    def __len__(self):
        return len(self.dataset)

    def load_oodDML_split(self, split_id=1):
        split_dict = pkl.load(open(self.path_ooDML_splits, 'rb'))
        train_classes, test_classes, fid = split_dict[split_id]['train'], split_dict[split_id]['test'], split_dict[split_id]['fid']
        return train_classes, test_classes, fid

