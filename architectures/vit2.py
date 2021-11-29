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

class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.name = self.arch
        self.vit, embed_dim_model = select_model(arch, pretrained=True if pretraining=='imagenet' else False)
        self.embed_dim = embed_dim if not embed_dim == -1 else embed_dim_model
        self.depth = 12 if not 'large' in self.arch else 24

        self.blocks = [*self.vit.blocks.children()]

        # self.embed = torch.nn.Linear(embed_dim_model, embed_dim) if embed_dim > 0 else nn.Identity() ## no activation function befor linear layer!
        self.embed = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(embed_dim_model, embed_dim)) if embed_dim > 0 else nn.Identity()

        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    # def forward(self, x):
    #     x = self.vit.patch_embed(x)
    #     cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    #     if self.vit.dist_token is None:
    #         x = torch.cat((cls_token, x), dim=1)
    #     else:
    #         x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    #     x = self.vit.pos_drop(x + self.vit.pos_embed)
    #     x = self.vit.blocks(x)
    #
    #     # apply layernorm if required (standard is TRUE)
    #     if 'nolayernorm' not in self.arch:
    #         x = self.vit.norm(x)
    #
    #     # directly use final cls-token embedding
    #     if not 'ens' in self.arch:
    #         x = x[:, 0]
    #     z = self.embed(x)
    #
    #     if 'normalize' in self.arch:
    #         z = torch.nn.functional.normalize(z, dim=-1)
    #
    #     return {'embeds': z}


    def forward(self, x, embed_depth=10):
        # x = self.forward_features(x)
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # evaluate blocks individually
        for i, block in enumerate(self.blocks):
            x = block(x)

            if i == embed_depth-1 and embed_depth < 12:
                # return {'embeds': x[:, 0]}
                break

        # # apply layernorm if required (standard is TRUE)
        # if 'nolayernorm' not in self.arch:
        #     x = self.vit.norm(x)

        # directly use final cls-token embedding
        x = x[:, 0]
        z = self.embed(x)

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds': z}