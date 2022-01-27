### Standard DML criteria
from criteria import triplet, margin, proxynca, npair, simsiam
from criteria import lifted, contrastive, softmax, ep
from criteria import angular, snr, histogram, arcface, proxyanchor, proxyanchor_orig, oproxy
from criteria import softtriplet, multisimilarity, quadruplet
### Basic Libs
import copy
import argparse

#####
losses = {'triplet': triplet,
          'margin': margin,
          'proxynca': proxynca,
          'npair': npair,
          'angular': angular,
          'contrastive': contrastive,
          'lifted': lifted,
          'snr': snr,
          'ep': ep,
          'oproxy': oproxy,
          'multisimilarity': multisimilarity,
          'histogram': histogram,
          'softmax': softmax,
          'softtriplet': softtriplet,
          'arcface': arcface,
          'proxyanchor': proxyanchor,
          'quadruplet': quadruplet,
          'proxyanchor_orig': proxyanchor_orig,
          'simsiam': simsiam,
          }

"""================================================================================================="""
def select(name, batchminer, **kwargs):

    if name not in losses: raise NotImplementedError('Loss {} not implemented!'.format(name))
    print(f"\nInitializing Optimization objective:\ntype: [{name}]")

    loss_lib = losses[name]
    if loss_lib.REQUIRES_BATCHMINER:
        if batchminer is None:
            raise Exception('Loss {} requires one of the following batch mining methods: {}'.format(name, loss_lib.ALLOWED_MINING_OPS))
        else:
            if batchminer.name not in loss_lib.ALLOWED_MINING_OPS:
                raise Exception('{}-mining not allowed for {}-loss!'.format(batchminer.name, name))

    opt = argparse.Namespace(**kwargs)
    loss_par_dict = {'opt': opt}
    if loss_lib.REQUIRES_BATCHMINER:
        loss_par_dict['batchminer'] = batchminer
        print(f"*** Batchmining enabled: [{batchminer.name}]\n")
    else:
        loss_par_dict['batchminer'] = None
        print(f"*** Batchmining disabled.\n")

    criterion = loss_lib.Criterion(**loss_par_dict)

    return criterion

def add_criterion_optim_params(criterion, to_optim):
    loss_lib = losses[criterion.name]
    if loss_lib.REQUIRES_OPTIM:
        if hasattr(criterion,'optim_dict_list') and criterion.optim_dict_list is not None:
            to_optim += criterion.optim_dict_list
        else:
            to_optim    += [{'params':criterion.parameters(), 'lr':criterion.lr}]

    return to_optim
