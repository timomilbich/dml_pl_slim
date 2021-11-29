import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import batchminer

"""================================================================================================="""
ALLOWED_MINING_OPS  = list(batchminer.BATCHMINING_METHODS.keys())
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True
REQUIRES_LOGGING    = False

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
        self.batchminer = batchminer
        self.name  = 'simsiam'
        self.criterion = nn.CosineSimilarity(dim=1)
        self.lr = opt.pred_lr
        self.n_warmup_iterations = opt.n_warmup_iterations
        self.iter_counter = 0

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.BatchNorm1d(opt.embed_dim),
                                       nn.Linear(opt.embed_dim, opt.pred_dim, bias=False),
                                       nn.BatchNorm1d(opt.pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(opt.pred_dim, opt.embed_dim)) # output layer

        ####
        self.ALLOWED_MINING_OPS  = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM      = REQUIRES_OPTIM
        self.REQUIRES_LOGGING = REQUIRES_LOGGING

    def forward(self, batch, labels, **kwargs):
        """
        Args:
            batch:   torch.Tensor: Input of embeddings with size (BS x DIM)
            labels: nparray/list: For each element of the batch assigns a class [0,...,C-1], shape: (BS x 1)
        """

        ids_pos = []
        bs, _ = batch.shape
        for i in range(bs):
            pos = (labels == labels[i]).cpu().numpy() # represents second view

            # Sample positives randomly
            if np.sum(pos) > 0:
                if np.sum(pos) > 1: pos[i] = 0 # exclude anchor itself
                ids_pos.append(np.random.choice(np.where(pos)[0]))
            else:
                raise Exception("Anchor without positive detected!")

        positives = batch[ids_pos, :]

        # create predictions
        self.iter_counter += 1
        if self.iter_counter <= self.n_warmup_iterations:
            p1 = self.predictor(batch.detach())  # NxC
            p2 = self.predictor(positives.detach())  # NxC
        else:
            p1 = self.predictor(batch)  # NxC
            p2 = self.predictor(positives)  # NxC
        # p1 = torch.nn.functional.normalize(p1, dim=-1)
        # p2 = torch.nn.functional.normalize(p2, dim=-1)

        # compute loss
        loss = -(self.criterion(p1, positives.detach()).mean() + self.criterion(p2, batch.detach()).mean()) * 0.5

        return loss
