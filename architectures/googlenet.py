"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import torchvision.models as mod





"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, embed_dim):
        super(Network, self).__init__()

        self.arch  = arch
        self.model = mod.googlenet(pretrained=True)

        self.model.last_linear = torch.nn.Linear(self.model.fc.in_features, embed_dim)
        self.model.fc = self.model.last_linear

        self.name = self.arch

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x):
        x = self.model(x)
        if not 'normalize' in self.arch:
            return x
        return torch.nn.functional.normalize(x, dim=-1)
