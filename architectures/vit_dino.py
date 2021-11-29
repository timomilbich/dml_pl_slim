import torch, torch.nn as nn
import architectures.vit_dino_aux as vit_dino


def select_model(arch, pretrained=True, **kwargs):
    if 'vit_small_patch8_224_dino' in arch:
        model = vit_dino.vit_small(patch_size=8)
        path_pretrained = './architectures/pretrained_weights/dino_deitsmall8_pretrain_full_checkpoint.pth'
        embed_dim = 384
    elif 'vit_small_patch16_224_dino' in arch:
        model = vit_dino.vit_small(patch_size=16)
        path_pretrained = './architectures/pretrained_weights/dino_deitsmall16_pretrain_full_checkpoint.pth'
        embed_dim = 384
    else:
        raise NotImplemented(f'Architecture {arch} has not been found.')

    if pretrained:
        checkpoint = torch.load(path_pretrained, map_location="cpu")
        state_dict = checkpoint["teacher"]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    return model, embed_dim



"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, arch, pretraining, embed_dim):
        super(Network, self).__init__()

        self.arch = arch
        self.name = self.arch
        self.vit, embed_dim_model = select_model(arch, pretrained=True if pretraining=='imagenet' else False)
        self.embed_dim = embed_dim if not embed_dim == -1 else embed_dim_model

        # self.embed = torch.nn.Linear(embed_dim_model, embed_dim) if embed_dim > 0 else nn.Identity() ## no activation function befor linear layer!
        self.embed = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(embed_dim_model, embed_dim)) if embed_dim > 0 else nn.Identity()
        print(f'ARCHITECTURE:\ntype: {self.arch}\nembed_dims: {self.embed_dim}\n')


    def forward(self, x):
        x = self.vit.prepare_tokens(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        z = x[:, 0]

        if 'normalize' in self.arch:
            z = torch.nn.functional.normalize(z, dim=-1)

        return {'embeds': z}