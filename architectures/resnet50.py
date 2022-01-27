"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
from architectures.VQ import VectorQuantizer, MultiHeadVectorQuantizer
import argparse

"""============================================================="""

MAX_BLOCK_TO_QUANTIZE = 4
NUM_FEAT_PER_BLOCK = {
    4: 2048,
    3: 1024,
    2: 512,
    1: 256,
}

class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim, VQ, n_e=1000, beta=0.25, e_dim=1024, k_e=1, e_init='random_uniform', block_to_quantize=4):
        super(Network, self).__init__()

        self.arch = arch
        self.embed_dim = embed_dim
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=pretraining if pretraining=='imagenet' else None)
        self.name = self.arch
        self.VQ = VQ
        self.n_e = n_e
        self.beta = beta
        self.e_dim = e_dim
        self.k_e = k_e
        self.e_init = e_init
        self.block_to_quantize = block_to_quantize
        if self.block_to_quantize < MAX_BLOCK_TO_QUANTIZE:
            assert '1x1conv' not in self.arch, "Expected dimesionality of intermediate features must be kept."
            self.e_dim = NUM_FEAT_PER_BLOCK[self.block_to_quantize]
        elif self.block_to_quantize > MAX_BLOCK_TO_QUANTIZE:
            raise Exception('Attempting to quantize non-existent resnet block [Max. number is 4.]!')

        # Add Vector Quantization (Optionally)
        if self.VQ:
            if self.k_e == 1:
                self.VectorQuantizer = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)
            else:
                self.VectorQuantizer = MultiHeadVectorQuantizer(self.n_e, self.k_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)

        # Freeze all/part of the BatchNorm layers (Optionally)
        if 'frozenAll' in self.arch:
            self.freeze_all_batchnorm()
        elif 'frozenPart' in self.arch:
            self.freeze_and_remove_batchnorm()
        elif 'frozen' in self.arch:
            self.freeze_all_batchnorm()

        # Downsample final feature map (channel dim., Optionally)
        embed_in_features_dim = self.model.last_linear.in_features
        print(f'Initializing Architecture: [{self.arch}]\n*** embed_dims = [{self.embed_dim}]')
        if '1x1conv' in self.arch:
            assert self.e_dim > 0
            print(f'*** 1x1conv dimensionality reduction: [2048 -> {self.e_dim}]\n')
            embed_in_features_dim = self.e_dim
            self.conv_reduce = nn.Conv2d(in_channels=2048, out_channels=self.e_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_reduce = nn.Identity()

        # Add embedding layer
        self.model.last_linear = torch.nn.Linear(embed_in_features_dim, embed_dim)
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        # Select pooling strategy
        self.pool_base = torch.nn.AdaptiveAvgPool2d(1)
        self.pool_aux  = torch.nn.AdaptiveMaxPool2d(1) if 'double' in self.arch else None

    def forward(self, x, warmup=False, quantize=True, **kwargs):
        # 'quantize' argument is needed to turn off quantization for initial features
        # extracting before clustering initialization

        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for k, layerblock in enumerate(self.layer_blocks):
            x = layerblock(x)

            if (k+1) == self.block_to_quantize:
                x = self.conv_reduce(x) # if unspecified in init, nn.Identity() is used
                prepool_y = x

                # VQ features #########
                if self.VQ:
                    if quantize:
                        x, vq_loss, perplexity, cluster_use, vq_indices = self.VectorQuantizer(x)
                    else:
                        vq_loss = 0
                        perplexity = 0
                        cluster_use = 0
                        vq_indices = 0
                ##########

        if self.pool_aux is not None:
            y = self.pool_aux(x) + self.pool_base(x)
        else:
            y = self.pool_base(x)
        y = y.view(y.size(0),-1)

        if warmup:
            x,y,prepool_y = x.detach(), y.detach(), prepool_y.detach()

        z = self.model.last_linear(y)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        if self.VQ:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss, 'vq_perplexity': perplexity, 'vq_cluster_use': cluster_use, 'vq_indices': vq_indices}
        else:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}

    def freeze_and_remove_batchnorm(self):
        layers_to_remove = ['layer4.2', 'layer4.1']

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                if any(x in name for x in layers_to_remove): # remove final layer bn, since it is going to be quantized. Pretrained BN params may be off otherwise.
                    layer.reset_parameters()
                    layer.eval()
                    layer.train = lambda _: None
                    with torch.no_grad():
                        layer.weight.fill_(1.0)
                        layer.bias.zero_()
                else: # freeze BN layers
                    layer.eval()
                    layer.train = lambda _: None

    def freeze_all_batchnorm(self):
        for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
            module.eval()
            module.train = lambda _: None

