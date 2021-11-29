"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import timm.models.resnet as timm_resnet

def select_model(arch, pretrained=True, **kwargs):
    if 'wide_resnet50_2' in arch:
        return timm_resnet.wide_resnet50_2(pretrained=True if pretrained=='imagenet' else False)
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.embed_dim = embed_dim
        self.model = select_model(self.arch, pretraining)
        self.name = self.arch

        if 'frozen' in self.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.fc = torch.nn.Linear(self.model.num_features, embed_dim)
        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x, **kwargs):
        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = self.model.fc(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(x, dim=-1)

        return {'embeds': z}
