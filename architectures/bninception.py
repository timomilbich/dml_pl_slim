"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import pretrainedmodels as ptm
from architectures.VQ import VectorQuantizer, MultiHeadVectorQuantizer


"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim, VQ, n_e = 1000, beta = 0.25, e_dim = 1024, k_e=1, e_init='random_uniform'):
        super(Network, self).__init__()

        self.arch  = arch
        self.embed_dim = embed_dim
        self.name = self.arch
        self.VQ = VQ
        self.n_e = n_e
        self.beta = beta
        self.e_dim = e_dim
        self.e_init = e_init
        self.k_e = k_e

        self.model = ptm.__dict__['bninception'](num_classes=1000, pretrained=pretraining)
        #self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, embed_dim)
        self.model.last_linear = torch.nn.Linear(self.e_dim, embed_dim)
        if self.VQ:
            if self.k_e == 1:
                self.VectorQuantizer = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.e_init)
            else:
                self.VectorQuantizer = MultiHeadVectorQuantizer(self.n_e, self.k_e, self.e_dim, self.beta, self.e_init)

        if '_he' in self.arch:
            torch.nn.init.kaiming_normal_(self.model.last_linear.weight, mode='fan_out')
            torch.nn.init.constant_(self.model.last_linear.bias, 0)

        if 'frozen' in self.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.pool_base = F.avg_pool2d
        self.pool_aux  = F.max_pool2d if 'double' in self.arch else None

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')

    def forward(self, x, warmup=False, quantize=True, **kwargs):
        # quantize argument is needed to turn off quantization for initial features
        # extracting before clustering initialization

        x = self.model.features(x)
        ###### 1*1 conv
        conv = nn.Conv2d(in_channels=x.shape[1], out_channels=self.e_dim, kernel_size=1, stride=1, padding=0).cuda()
        x = conv(x)
        ######
        prepool_y = x

        # VQ features #########
        if self.VQ:
            if quantize:
                x, vq_loss, perplexity, cluster_use = self.VectorQuantizer(x)
            else:
                vq_loss = 0
                perplexity = 0
                cluster_use = 0
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
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss, 'vq_perplexity': perplexity, 'vq_cluster_use': cluster_use}
        else:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}
