"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import pretrainedmodels as ptm
from architectures.VQ import VectorQuantizer, MultiHeadVectorQuantizer, FactorizedVectorQuantizer


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim, VQ, n_e = 1000, beta = 0.25, e_dim = 1024, k_e=1, e_init='random_uniform', e_dim_latent=32, **kwargs):
        super(Network, self).__init__()

        self.arch  = arch
        self.embed_dim = embed_dim
        self.name = self.arch
        self.VQ = VQ
        self.n_e = n_e
        self.beta = beta
        self.e_dim = e_dim
        self.e_dim_latent = e_dim_latent
        self.e_init = e_init
        self.k_e = k_e
        self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained=pretraining)

        # Add Vector Quantization (Optionally)
        if 'VQ_factorized' in self.VQ:
            self.VectorQuantizer = FactorizedVectorQuantizer(self.VQ, self.n_e, self.e_dim, self.e_dim_latent, self.beta, self.e_init)
        elif 'VQ_vanilla' in self.VQ:
                self.VectorQuantizer = VectorQuantizer(self.VQ, self.n_e, self.e_dim, self.beta, self.e_init)
        elif 'VQ_multihead' in self.VQ:
            self.VectorQuantizer = MultiHeadVectorQuantizer(self.VQ, self.n_e, self.k_e, self.e_dim, self.beta, self.e_init)
        else:
            self.VQ = False

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
            print(f'*** 1x1conv dimensionality reduction: [1024 -> {self.e_dim}]\n')
            embed_in_features_dim = self.e_dim
            self.conv_reduce = nn.Conv2d(in_channels=1024, out_channels=self.e_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_reduce = nn.Identity()

        # Add embedding layer
        self.model.last_linear = torch.nn.Linear(embed_in_features_dim, embed_dim)
        if '_he' in self.arch:
            torch.nn.init.kaiming_normal_(embed_in_features_dim, mode='fan_out')
            torch.nn.init.constant_(self.model.last_linear.bias, 0)

        # Select pooling strategy
        self.pool_base = F.avg_pool2d
        self.pool_aux  = F.max_pool2d if 'double' in self.arch else None

    def forward(self, x, warmup=False, quantize=True, **kwargs):
        # quantize argument is needed to turn off quantization for initial features
        # extracting before clustering initialization

        x = self.model.features(x)
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

        y = self.pool_base(x,kernel_size=x.shape[-1])
        # print('prepool y', prepool_y.shape)  112, 1024, 1, 1
        if self.pool_aux is not None:
            y += self.pool_aux(x, kernel_size=x.shape[-1])
        if 'lp2' in self.arch:
            y += F.lp_pool2d(x, 2, kernel_size=x.shape[-1])
        if 'lp3' in self.arch:
            y += F.lp_pool2d(x, 3, kernel_size=x.shape[-1])
        # print('y before view:', y.shape) = 112， 1024， 1， 1
        y = y.view(len(x),-1)
        if warmup:
            x,y,prepool_y = x.detach(), y.detach(), prepool_y.detach()

        z = self.model.last_linear(y)
        if 'normalize' in self.name:
            z = F.normalize(z, dim=-1)
        if self.VQ:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss, 'vq_perplexity': perplexity, 'vq_cluster_use': cluster_use, 'vq_indices': vq_indices}
        else:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}

    def freeze_and_remove_batchnorm(self):
        layers_to_remove = ['inception_5']

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
