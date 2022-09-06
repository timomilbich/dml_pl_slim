"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
from torch import einsum
import torch.nn.functional as F
import pretrainedmodels as ptm
from einops import rearrange, repeat

from architectures.VQ import VectorQuantizer, MultiHeadVectorQuantizer, FactorizedVectorQuantizer
"""============================================================="""

MAX_BLOCK_TO_QUANTIZE = 4
NUM_FEAT_PER_BLOCK = {
    4: 2048,
    3: 1024,
    2: 512,
    1: 256,
}

class Network(torch.nn.Module):
    def __init__(self, arch='resnet50_frozen_normalize', pretraining='imagenet', embed_dim=512, VQ=None, n_e=2000, beta=0.25,
                 e_dim=2048, k_e=1, e_init='feature_clustering', block_to_quantize=4, ca_n_heads=1, ca_latent_dim=2048,
                 bn_layers=[4.2, 4.1], **kwargs):
        super(Network, self).__init__()

        self.arch = arch
        self.embed_dim = embed_dim
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained=pretraining if pretraining=='imagenet' else None)
        self.name = self.arch
        self.bn_layers = bn_layers
        self.RETRO = False

        self.ca_n_heads = ca_n_heads 
        self.ca_latent_dim = ca_latent_dim

        self.VQ = VQ
        self.e_init = e_init
        self.n_e = n_e
        self.beta = beta
        self.e_dim = e_dim
        self.k_e = k_e
        self.block_to_quantize = block_to_quantize

        if self.block_to_quantize < MAX_BLOCK_TO_QUANTIZE:
            assert '1x1conv' not in self.arch, "Expected dimesionality of intermediate features must be kept."
            self.e_dim = NUM_FEAT_PER_BLOCK[self.block_to_quantize]
        elif self.block_to_quantize > MAX_BLOCK_TO_QUANTIZE:
            raise Exception('Attempting to quantize non-existent resnet block [Max. number is 4.]!')

        # Add Vector Quantization (Optionally)
        if 'VQ_factorized' in self.VQ:
            self.VectorQuantizer = FactorizedVectorQuantizer(self.VQ, self.n_e, self.e_dim, self.e_dim_latent, self.beta, self.e_init, self.block_to_quantize)
        elif 'VQ_vanilla' in self.VQ:
                self.VectorQuantizer = VectorQuantizer(self.VQ, self.n_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)
        elif 'VQ_multihead' in self.VQ:
            self.VectorQuantizer = MultiHeadVectorQuantizer(self.VQ, self.n_e, self.k_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)
        else:
            self.VQ = False

        # Freeze all/part of the BatchNorm layers (Optionally)
        if 'frozenAll' in self.arch:
            self.freeze_all_batchnorm()
        elif 'frozenPart' in self.arch:
            self.freeze_and_remove_batchnorm(self.bn_layers)
        elif 'frozen_bn2ln' in self.arch:
            self.freeze_and_bn_to_ln()
        elif 'frozen' in self.arch:
            self.freeze_all_batchnorm()

        # Downsample final feature map (channel dim., Optionally)
        embed_in_features_dim = self.model.last_linear.in_features
        print(f'Initializing Architecture: [resnet50_ca]\n*** arch = [{self.arch}]\n*** embed_dims = [{self.embed_dim}]')
        if '1x1conv' in self.arch:
            assert self.e_dim > 0
            print(f'*** 1x1conv dimensionality reduction: [2048 -> {self.e_dim}]\n')
            embed_in_features_dim = self.e_dim
            self.conv_reduce = nn.Conv2d(in_channels=2048, out_channels=self.e_dim, kernel_size=1, stride=1, padding=0)
        else:
            print(f'\n')
            self.conv_reduce = nn.Identity()
            assert self.e_dim == 2048, "Without reduction, native feature dim of 2048 is expected!"

        # init cross attention
        print(f'Initializing Architecture: latent cross attention')
        self.ca_input_dim = self.e_dim

        assert self.ca_latent_dim % self.ca_n_heads == 0, "cross_att_head_dim is not integer!"
        self.ca_head_dim = int(self.ca_latent_dim / self.ca_n_heads)


        # TODO: this is the Q in the paper
        # by using nn.Parameter -> requires_grad and learnable
        # with N=1, we get rid of pooling layer
        self.latents = nn.Parameter(torch.randn(1, self.ca_latent_dim))

        # when ca_n_heads == 1, the latent_dim = inner_dim
        self.cross_attn = PreNorm(self.ca_latent_dim, Attention3(self.ca_latent_dim, self.ca_input_dim, heads=self.ca_n_heads, dim_head=self.ca_head_dim), context_dim=self.ca_input_dim, type_norm='ln')
        # self.cross_attn_ff = PreNorm(self.cross_att_latent_dim, FeedForward(self.cross_att_latent_dim))

        print(f'*** ca_input_dim = [{self.ca_input_dim}]')
        print(f'*** ca_latent_dim = [{self.ca_latent_dim}]')
        print(f'*** ca_n_heads = [{self.ca_n_heads}]')
        print(f'*** ca_head_dim = [{self.ca_head_dim}]\n')

        # Build base network
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        # Add embedding layer
        if 'direct' not in self.arch:
            self.model.last_linear = torch.nn.Linear(embed_in_features_dim, embed_dim)
        elif 'directLinear' in self.arch:
            self.model.last_linear = torch.nn.Linear(self.ca_latent_dim, embed_dim)
        else:
            assert self.ca_latent_dim == self.embed_dim, "make sure selected embed dim is maintained!"

    def _init_latent_parameters(self):
        with torch.no_grad():
            self.latents.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, warmup=False, quantize=True, **kwargs):
        # 'quantize' argument is needed to turn off quantization for initial features
        # extracting before clustering initialization

        # prepare latents (Q) for cross attention, and duplicate for b (batch) times.
        y = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        # apply base model
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for k, layerblock in enumerate(self.layer_blocks):
            x = layerblock(x)

            if (k+1) == self.block_to_quantize:
                x = self.conv_reduce(x) # if unspecified in init, nn.Identity() is used
                # prepool_y is the continuous feature here
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

        # stop here when we only need backbone tokens for initializing the codebook        
        if not quantize:
            return {'extra_embeds': prepool_y}
        
        if warmup:
            x = x.detach()

        # cross attention, x is the output from backbone + VQ
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = x.view(x.shape[0], -1, self.ca_input_dim)
        
        # the rearranged continuous feature
        x_con = rearrange(prepool_y, 'b c h w -> b h w c').contiguous()
        x_con = x_con.view(x_con.shape[0], -1, self.ca_input_dim)

        y = self.cross_attn(y, context_k=x, context_v=x_con) # + y #


        # map to embedding space
        if 'direct' in self.arch:
            z = torch.squeeze(y)
        else:
            z = self.model.last_linear(torch.squeeze(y))

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        if self.VQ:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss, 'vq_perplexity': perplexity, 'vq_cluster_use': cluster_use, 'vq_indices': vq_indices}
        else:
            return {'embeds': z, 'avg_features': y, 'features': x, 'extra_embeds': prepool_y}

    def freeze_and_remove_batchnorm(self, layers):
        # layers_to_remove = ['layer4.2', 'layer4.1']
        layers_to_remove = [f'layer{l}' for l in layers]

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

    def freeze_and_bn_to_ln(self):
        layers_to_remove = ['layer4.2', 'layer4.1']
        H = W = 7

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                if any(x in name for x in layers_to_remove): # remove final layer bn, since it is going to be quantized. Pretrained BN params may be off otherwise.
                    name_split = name.split('.')
                    self.model._modules[name_split[0]][int(name_split[1])]._modules[name_split[2]] = nn.LayerNorm([layer.num_features, H, W]) #todo: wrongly implemented?

                else: # freeze BN layers
                    layer.eval()
                    layer.train = lambda _: None


################################################# CROSS ATTENTION#################################

"""
The cross attention is adapted from https://github.com/lucidrains/perceiver-pytorch
"""


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None, type_norm='ln'):
        super().__init__()
        self.fn = fn
        self.type_norm = type_norm

        if type_norm == 'ln':
            self.norm = nn.LayerNorm(dim)
            self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
        elif type_norm == 'gn':
            self.norm = nn.GroupNorm(2, dim)
            self.norm_context = nn.GroupNorm(2, context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        
        # x here is basically the query for CA module
        if self.type_norm == 'ln':
            x = self.norm(x)
        elif self.type_norm == 'gn':
            x = x.permute(0, 2, 1)
            x = self.norm(x)
            x = x.permute(0, 2, 1)

        if exists(self.norm_context):
            context_k = kwargs['context_k']
            context_v = kwargs['context_v']

            if self.type_norm == 'ln':
                normed_context_k = self.norm_context(context_k)
                normed_context_v = self.norm_context(context_v)
            # elif self.type_norm == 'gn':
            #     normed_context = context.permute(0, 2, 1)
            #     normed_context = self.norm_context(normed_context)
            #     normed_context = normed_context.permute(0, 2, 1)
            kwargs.update(context_k=normed_context_k)
            kwargs.update(context_v=normed_context_v)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    # check GLU variants https://arxiv.org/pdf/2002.05202.pdf
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            # why *2, due to the GEGLU()
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        # self.dropout = nn.Dropout(dropout)
        # self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        # out = self.to_out(out)
        return out

class Attention2(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        # the sqrt(d)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        # self.dropout = nn.Dropout(dropout)
        # self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        # k = self.to_k(k)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        # v = self.to_v(v)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)

        return out



class Attention3(nn.Module):
    ''' The attention class to merge continuous and vq features, 
    the only difference from Attention2 is that it takes 2 contexts 
    instead of one, one direct from backbone one from vq.
    '''
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        # the sqrt(d)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)


    def forward(self, x, context_k=None, context_v=None):
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        context_k = default(context_k, x)
        context_v = default(context_v, x)
        k = self.to_k(context_k)
        v = self.to_v(context_v)

        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # out =  self.to_out(out)

        return out
