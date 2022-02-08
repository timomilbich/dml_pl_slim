"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import torch, torch.nn as nn
import pretrainedmodels as ptm
from architectures.VQ import VectorQuantizer, MultiHeadVectorQuantizer, FactorizedVectorQuantizer
import argparse
from einops import rearrange

"""============================================================="""

MAX_BLOCK_TO_QUANTIZE = 4
NUM_FEAT_PER_BLOCK = {
    4: 2048,
    3: 1024,
    2: 512,
    1: 256,
}

class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim, VQ, n_e=1000, beta=0.25, e_dim=2048, k_e=1, e_init='random_uniform', block_to_quantize=4):
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
        # if self.block_to_quantize < MAX_BLOCK_TO_QUANTIZE:
        #     assert '1x1conv' not in self.arch, "Expected dimesionality of intermediate features must be kept."
        #     self.e_dim = NUM_FEAT_PER_BLOCK[self.block_to_quantize]
        # elif self.block_to_quantize > MAX_BLOCK_TO_QUANTIZE:
        #     raise Exception('Attempting to quantize non-existent resnet block [Max. number is 4.]!')
        if self.block_to_quantize != MAX_BLOCK_TO_QUANTIZE:
            raise Exception('Currently only the quantization of the final ResNet block is supported!')

        # Add Vector Quantization (Optionally)
        if 'VQ_factorized' in self.VQ:
            self.VectorQuantizer = FactorizedVectorQuantizer(self.n_e, self.e_dim, self.e_dim_latent, self.beta, self.e_init, self.block_to_quantize)
        elif 'VQ_vanilla' in self.VQ:
                self.VectorQuantizer = VectorQuantizer(self.n_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)
        elif 'VQ_multihead' in self.VQ:
            self.VectorQuantizer = MultiHeadVectorQuantizer(self.VQ, self.n_e, self.k_e, self.e_dim, self.beta, self.e_init, self.block_to_quantize)
        else:
            self.VectorQuantizer = None
            self.VQ = False
            self.k_e = 1
            self.e_dim = 2048

        print(f'Initializing Architecture: backbone [{self.arch}]')
        # Freeze all/part of the BatchNorm layers (Optionally)
        if 'frozenAll' in self.arch:
            self.freeze_all_batchnorm()
        elif 'frozenPart' in self.arch:
            self.freeze_and_remove_batchnorm()
        elif 'frozen' in self.arch:
            self.freeze_all_batchnorm()

        # Build ResNet backbone
        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        # Downsample final feature map (channel dim., Optionally)
        if '1x1conv' in self.arch:
            assert self.e_dim > 0
            print(f'*** 1x1conv dimensionality reduction: [2048 -> {self.e_dim}]')
            self.conv_reduce = nn.Conv2d(in_channels=2048, out_channels=self.e_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_reduce = nn.Identity()

        # Add Transformer blocks
        print(f'Initializing Architecture: transformer')
        self.n_tokens = 7 * 7 * self.k_e # current feature map resolution is 7 x 7
        self.e_dim_seg = self.e_dim / self.k_e
        assert self.e_dim % self.k_e  == 0, "Assert feature dim is dividable by codeword dim."
        self.e_dim_seg = int(self.e_dim_seg)
        self.embed_dim_head = self.e_dim_seg
        self.transf_depth = 2
        self.transf_num_heads = 1
        self.transf_mlp_ratio = 2.

        print(f'*** e_dim_seg = [{self.e_dim_seg}]')
        print(f'*** embed_dim_head = [{self.embed_dim_head}]')
        print(f'*** n_tokens = [{self.n_tokens}]')
        print(f'*** depth = [{self.transf_depth}]')
        print(f'*** num_heads = [{self.transf_num_heads}]')
        print(f'*** mlp_ratio = [{self.transf_mlp_ratio}]')

        self.model.transformer = CustomTransformer(
            num_tokens=self.n_tokens, embed_dim=self.e_dim_seg, embed_dim_head=self.e_dim_seg, depth=self.transf_depth,
            num_heads=self.transf_num_heads, mlp_ratio=self.transf_mlp_ratio,
        )

    def forward(self, x, quantize=True, warmup=False, **kwargs):
        # 'quantize' argument is needed to turn off quantization for initial features
        # extracting before clustering initialization

        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for k, layerblock in enumerate(self.layer_blocks):
            x = layerblock(x)

            if (k+1) == self.block_to_quantize:

                if warmup:
                    x = x.detach()

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

        if self.VectorQuantizer:
            x = self.VectorQuantizer.split_feature_map(x)
        else:
            x = rearrange(x, 'b c h w -> b h w c').contiguous()
            x = x.view(-1, self.embed_dim_seg)
        z = self.model.transformer(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        if self.VQ:
            return {'embeds': z, 'features': x, 'extra_embeds': prepool_y, 'vq_loss': vq_loss, 'vq_perplexity': perplexity, 'vq_cluster_use': cluster_use, 'vq_indices': vq_indices}
        else:
            return {'embeds': z, 'features': x, 'extra_embeds': prepool_y}

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

######################################################################################################
############################################## TRANSFORMER STUFF #####################################
######################################################################################################

from functools import partial
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, Mlp

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "Embedding dimensions not divisible by num. heads!"
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        # self.mlp = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class CustomTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(self, num_tokens=128, embed_dim=768, depth=1, num_heads=8, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='', embed_dim_head=-1):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.embed_dim_head = embed_dim_head
        self.num_tokens = num_tokens
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim)) # +1 for positional embedding
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(self.embed_dim, self.embed_dim) if embed_dim_head > 0 else nn.Identity()
        # self.head = nn.Identity()

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def forward_features(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        #todo: use pooling over sequence dim, instead of using CLS token

        return x[:, 0] # return only class token embedding

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


