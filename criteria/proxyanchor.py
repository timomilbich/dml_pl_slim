import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_LOGGING    = False


class Criterion(torch.nn.Module):
    def __init__(self, opt):
        """
        Args:
            opt: Namespace containing all relevant parameters.
        """
        super(Criterion, self).__init__()
        self.num_proxies        = opt.n_classes
        self.embed_dim          = opt.embed_dim

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        self.REQUIRES_LOGGING = REQUIRES_LOGGING

        ####
        self.embed_div          = opt.loss_proxyanchor_div
        self.proxies            = torch.nn.Parameter(torch.randn(self.num_proxies, self.embed_dim)/self.embed_div)
        self.lr   = opt.lr * opt.loss_proxyanchor_lrmulti

        self.class_idxs         = torch.arange(self.num_proxies)

        self.name           = 'proxyanchor'

        self.sphereradius = opt.loss_proxyanchor_sphereradius
        self.alpha        = opt.loss_proxyanchor_alpha
        self.delta        = opt.loss_proxyanchor_delta

        self.var = opt.loss_proxyanchor_var

    def prep(self, thing):
        return self.sphereradius*torch.nn.functional.normalize(thing, dim=1)

    def forward(self, batch, labels, aux_batch=None):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        loss = 0

        ###
        bs          = len(batch)
        batch       = self.prep(batch)
        batch_dict  = {}

        ###
        labels       = labels.unsqueeze(1)
        bsame_labels = labels.T == labels
        bdiff_labels = labels.T != labels
        diff_labels  = torch.arange(len(self.proxies)).unsqueeze(1) != labels.T

        ###
        proxies     = self.prep(self.proxies)
        pos_proxies = proxies[labels.squeeze(-1)]

        ###
        pos_s    = batch.mm(pos_proxies.T)
        pos_s    = -self.alpha*(pos_s-self.delta)
        max_v    = pos_s.max(dim=0,keepdim=True)[0]
        pos_s    = pos_s - max_v
        pos_s    = torch.nn.Softplus()(torch.log(torch.sum(torch.exp(pos_s)*bsame_labels.to(torch.float).T.to(pos_s.device), dim=0)).view(-1)+max_v.view(-1)).mean()

        neg_s    = batch.mm(proxies.T)
        neg_s    = self.alpha*(neg_s+self.delta)
        max_v    = neg_s.max(dim=0,keepdim=True)[0]
        neg_s    = neg_s - max_v
        neg_s    = torch.nn.Softplus()(torch.log(torch.sum(torch.exp(neg_s)*diff_labels.to(torch.float).T.to(neg_s.device), dim=0)).view(-1)+max_v.view(-1)).mean()

        loss  = pos_s + neg_s

        return loss
