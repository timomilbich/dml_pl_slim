import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer


"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = True
REQUIRES_OPTIM      = False
### MarginLoss with trainable class separation margin beta. Runs on Mini-batches as well.
class Criterion(torch.nn.Module):
    def __init__(self, opt, batchminer):
        """
        Args:
            margin:             Triplet Margin.
            nu:                 Regularisation Parameter for beta values if they are learned.
            beta:               Class-Margin values.
            n_classes:          Number of different classes during training.
        """
        super(Criterion, self).__init__()
        self.n_classes          = opt.n_classes

        self.temp  = opt.loss_ep_temp
        self.name  = 'easy positive'

        self.batchminer = batchminer

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM


    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """
        sampled_triplets, distances = self.batchminer(batch, labels, return_distances=True)
        a_list, p_list, n_list = [], [], []

        loss, count = 0, 0
        for i in range(len(labels)):
            a,_,n  = sampled_triplets[i]
            neg    = np.array(labels!=labels[i]); pos = np.array(labels==labels[i])
            pos[i] = 0

            if np.sum(pos)>1:
                pos_idxs = np.where(pos)[0]
                p        = pos_idxs[np.argmin(distances[i][pos])]
                loss    += -F.log_softmax(torch.stack([self.similarity(batch[a],batch[p]),self.similarity(batch[a],batch[n])]).unsqueeze(0)/self.temp)[:,0]
                count   += 1
        loss = loss/count

        return loss

    def similarity(self,x,y, mode='euclidean'):
        if mode=='euclidean':
            return 1-torch.dist(x,y,p=2)/2
        elif mode=='cosine':
            return x.T.mm(y)
