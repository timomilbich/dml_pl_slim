import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS = ['npair']
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False

### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
        """
        super(Criterion, self).__init__()
        self.pars = opt
        self.l2_weight = opt.loss_npair_l2
        self.batchminer = batchminer

        self.name           = 'npair'

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        anchors, positives, negatives = self.batchminer(batch, labels)

        ##
        loss  = 0
        if 'bninception' in self.pars.arch:
            ### NPair does not allow for a logsumexp - notation, hence the clamping to avoid overflow!
            batch = batch/4
        for anchor, positive, negative_set in zip(anchors, positives, negatives):
            a_embs, p_embs, n_embs = batch[anchor:anchor+1], batch[positive:positive+1], batch[negative_set]
            inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
            inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
            loss  = loss + torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1))/len(anchors)
            loss  = loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))/len(anchors)



        # a_embs, p_embs, n_embs = batch[anchors], batch[positives], batch[negatives]
        # inner_sum = a_embs[:,None,:].bmm((n_embs - p_embs[:,None,:]).permute(0,2,1))
        # inner_sum = inner_sum.view(inner_sum.shape[0], inner_sum.shape[-1])
        #
        # loss      = torch.mean(torch.log(torch.sum(torch.exp(inner_sum), dim=1) + 1))
        # loss      = loss + self.l2_weight*torch.mean(torch.norm(batch, p=2, dim=1))
        # from IPython import embed; embed()

        return loss
