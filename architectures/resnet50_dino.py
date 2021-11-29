"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
from torchvision import models as torchvision_models

"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.embed_dim = embed_dim

        # init model
        # self.model = torchvision_models.__dict__['resnet50'](num_classes=1000, pretrained=pretraining if pretraining=='imagenet' else None)
        self.model = torchvision_models.__dict__['resnet50']()
        feat_dim = self.model.fc.weight.shape[1]
        self.model.fc = nn.Identity()

        # load checkpoint
        path_pretrained = './architectures/pretrained_weights/dino_resnet50_pretrain_full_checkpoint.pth'
        state_dict = torch.load(path_pretrained, map_location="cpu")
        state_dict = state_dict['teacher']
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)

        self.name = self.arch

        if 'frozen' in self.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.fc = torch.nn.Linear(feat_dim, embed_dim)
        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x):
        z = self.model(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds': z}
