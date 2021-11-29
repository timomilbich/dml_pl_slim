from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

"""==================================================================================================="""
################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseDataset(Dataset):
    def __init__(self, image_dict, arch, is_validation=False):
        self.is_validation = is_validation
        self.arch        = arch
        self.path_ooDML_splits = None

        #####
        self.image_dict = image_dict

        #####
        self.init_setup()

        #####
        if 'clip' in self.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])
        elif 'bninception' in self.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.502, 0.4588, 0.4078],std=[0.0039, 0.0039, 0.0039])
        elif 'vit' in self.arch:
            self.f_norm = normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
        else:
            self.f_norm = normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.crop_size = crop_im_size = 224 if 'googlenet' not in self.arch else 227

        #############
        self.normal_transform = []
        if not self.is_validation:
            self.normal_transform.extend([transforms.RandomResizedCrop(size=crop_im_size), transforms.RandomHorizontalFlip(0.5)])
        else:
            self.normal_transform.extend([transforms.Resize(256), transforms.CenterCrop(crop_im_size)])
        self.normal_transform.extend([transforms.ToTensor(), normalize])
        self.normal_transform = transforms.Compose(self.normal_transform)

    def init_setup(self):
        self.n_files       = np.sum([len(self.image_dict[key]) for key in self.image_dict.keys()])
        self.avail_classes = sorted(list(self.image_dict.keys()))

        counter = 0
        temp_image_dict = {}
        for i,key in enumerate(self.avail_classes):
            temp_image_dict[key] = []
            for path in self.image_dict[key]:
                temp_image_dict[key].append([path, counter])
                counter += 1

        self.image_dict = temp_image_dict
        self.image_list = [[(x[0],key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        self.image_paths = self.image_list

        self.is_init = True

    def ensure_3dim(self, img):
        if len(img.size)==2:
            img = img.convert('RGB')
        return img


    def __getitem__(self, idx):

        input_image = self.ensure_3dim(Image.open(self.image_list[idx][0]))
        im_a = self.normal_transform(input_image)
        if 'bninception' in self.arch:
            im_a = im_a[range(3)[::-1],:]
        return im_a, self.image_list[idx][-1], idx

    def __len__(self):
        return self.n_files
