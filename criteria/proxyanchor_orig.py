import numpy as np
import batchminer

import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.nb_classes     = opt.n_classes
        self.sz_embed       = opt.embed_dim
        self.mrg            = opt.loss_proxyanchor_delta
        self.alpha          = opt.loss_proxyanchor_alpha
        self.name           = 'proxyanchor_orig'

        self.lr   = opt.lr * opt.loss_proxyanchor_lrmulti

        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def binarize(self, T, nb_classes):
        T = T.cpu().numpy()
        import sklearn.preprocessing
        T = sklearn.preprocessing.label_binarize(
            T, classes=range(0, nb_classes)
        )
        T = torch.FloatTensor(T).cuda()
        return T

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, X, T):
        P = self.proxies

        P_one_hot = self.binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        dists = F.linear(self.l2_norm(X), self.l2_norm(P))  # Calcluate cosine similarity
        pos_exp = torch.exp(-self.alpha * (dists - self.mrg))
        neg_exp = torch.exp(self.alpha * (dists + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss