from functools import partial
import os
import torch, torch.nn as nn

import timm.models.vision_transformer as timm_vit
from timm.models.vision_transformer import _create_vision_transformer, _cfg


def select_model(arch, pretrained=True, **kwargs):
    if 'vit_base_patch32_224' in arch:
        return timm_vit.vit_base_patch32_224_in21k(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_base_patch16_224_in21k' in arch:
        return timm_vit.vit_base_patch16_224_in21k(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_base_patch16_224' in arch:
        return timm_vit.vit_base_patch16_224(pretrained=pretrained, **kwargs), 768 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_small_patch32_224' in arch:
        return timm_vit.vit_small_patch32_224(pretrained=pretrained, **kwargs), 384 # CHECK IF NEW TIMM PIP IS AVAILABLE TO USE NORMAL IMAGENET
    elif 'vit_small_patch16_224_in21k' in arch:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
        cfg_custom = _cfg(url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz', num_classes=21843),
        model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, default_cfg=cfg_custom[0], **model_kwargs)
        return model, 384
    elif 'vit_small_patch16_224' in arch:
        model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
        cfg_custom = _cfg(url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz')
        model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, default_cfg=cfg_custom, **model_kwargs)
        return model, 384
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')


"""============================================================="""

class Ensemble_pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, x):

        # aggregate by pooling
        return self.pool(x.permute(0, 2, 1)).squeeze()

class Ensemble_addition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):

        # aggregate by addition
        return torch.sum(x, 1)

class Ensemble_attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.W_local = torch.nn.Parameter(torch.rand([196, dim, dim]))


    def forward(self, x):
        embed_global = x[:, 0]
        embeds_local = x[:, 1::]

        # compute attention weighting
        global_embed_proj = torch.matmul(embed_global, self.W_local).permute(1, 0, 2) # batchsize x n_embed-1 x dim
        attn = torch.matmul(global_embed_proj, embeds_local.transpose(1, 2)) * self.scale
        attn = torch.diagonal(attn, 0, 1, 2)
        attn = attn.softmax(dim=-1)

        # aggregate
        attn = attn.unsqueeze(-1).repeat(1, 1, self.head_dim)
        embed = embed_global + torch.sum(embeds_local * attn, 1)

        # # attention
        # B, N, C = embeds_local.shape
        # qkv = self.qkv(embeds_local).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        #
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        #
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return embed

        return x

class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.name = self.arch
        self.vit, embed_dim_model = select_model(arch, pretrained=True if pretraining=='imagenet' else False)
        self.embed_dim = embed_dim if not embed_dim == -1 else embed_dim_model

        if 'att_ens' in self.arch:
            self.embed = Ensemble_attention(embed_dim_model, num_heads=1)
        elif 'add_ens' in self.arch:
            self.embed = Ensemble_addition()
        elif 'pool_ens' in self.arch:
            self.embed = Ensemble_pool()
        else:
            # self.embed = torch.nn.Linear(embed_dim_model, embed_dim) if embed_dim > 0 else nn.Identity() ## no activation function befor linear layer!
            self.embed = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(embed_dim_model, embed_dim)) if embed_dim > 0 else nn.Identity()

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)

        # apply layernorm if required (standard is TRUE)
        if 'nolayernorm' not in self.arch:
            x = self.vit.norm(x)

        # directly use final cls-token embedding
        if not 'ens' in self.arch:
            x = x[:, 0]
        z = self.embed(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds': z}
