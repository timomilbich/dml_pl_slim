"""
    Vector Quantizer taken from the VQ-GAN github.
    see https://github.com/CompVis/taming-transformers/blob/31216490efe8ae3604efbf9f1531ff5c70bd446a/taming/modules/vqvae/quantize.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
import faiss
from copy import  deepcopy


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, vq_arch, n_e, e_dim, beta, e_init='random_uniform', block_to_quantize=-1, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy
        self.e_init = e_init
        self.k_e = 1
        self.vq_arch = vq_arch
        self.block_to_quantize = block_to_quantize

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if self.e_init == 'random_uniform':
            self.embedding.weight.data.uniform_(-100.0 / self.n_e, 100.0 / self.n_e)

        print(f'Initializeing VQ [VectorQuantization]')
        print(f'*** n_e = [{self.n_e}]')
        print(f'*** e_dim = [{self.e_dim}]')
        print(f'*** e_init = [{self.e_init}]')
        print(f'*** block_to_quantize = [{self.block_to_quantize}]')
        print(f'*** beta = [{self.beta}]\n')

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding.weight(min_encoding_indices)
        z_q = z_q.view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        perplexity, cluster_use = self.measure_perplexity(min_encoding_indices, self.n_e)

        return z_q, loss, perplexity, cluster_use, min_encoding_indices


    def init_codebook_by_clustering(self, features, evaluate_on_gpu=True, n_max=100000):

        ### Prepare features
        features = features.astype(np.float32)

        ### select samples to use
        idx_to_use = np.random.choice(features.shape[0], np.min([features.shape[0], n_max]), replace=False)
        features = features[idx_to_use, :]

        ### Init faiss
        faiss.omp_set_num_threads(20)
        res = None
        torch.cuda.empty_cache()
        if evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        ### Set CPU Cluster index
        cluster_idx = faiss.IndexFlatL2(features.shape[-1])
        if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
        kmeans = faiss.Clustering(features.shape[-1], self.n_e)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(features, cluster_idx)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(self.n_e, features.shape[-1])

        ### Init codebook
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(deepcopy(centroids)).float(), freeze=False)

        ### empty cache on GPU
        torch.cuda.empty_cache()

    def measure_perplexity(self, predicted_indices, n_embed):  # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use


class MultiHeadVectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, vq_arch, n_e, k_e, e_dim, beta, e_init='random_uniform', block_to_quantize=-1, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.k_e = k_e
        self.e_dim = e_dim
        self.vq_arch = vq_arch

        self.e_dim_seg = self.e_dim / self.k_e
        assert self.e_dim % self.k_e  == 0, "Assert feature dim is dividable by codeword dim."
        self.e_dim_seg = int(self.e_dim_seg)

        self.beta = beta
        self.legacy = legacy
        self.e_init = e_init
        self.block_to_quantize = block_to_quantize

        self.embedding = nn.Embedding(self.n_e, self.e_dim_seg)
        if self.e_init == 'random_uniform':
            self.embedding.weight.data.uniform_(-100.0 / self.n_e, 100.0 / self.n_e)

        print(f'Initializeing VQ [MultiHeadVectorQuantization]')
        print(f'*** n_e = [{self.n_e}]')
        print(f'*** e_dim = [{self.e_dim}]')
        print(f'*** k_e = [{self.k_e}]')
        print(f'*** e_dim_seg = [{self.e_dim_seg}]')
        print(f'*** e_init = [{self.e_init}]')
        print(f'*** block_to_quantize = [{self.block_to_quantize}]')
        print(f'*** beta = [{self.beta}]\n')

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_shape = z.shape
        z = torch.chunk(z, self.k_e, -1)

        all_z_sub_q = []
        losses = []
        all_min_encoding_indices = []
        for k, z_sub in enumerate(z):

            z_sub = z_sub.view(-1, self.e_dim_seg)

            d = torch.sum(z_sub ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_sub, rearrange(self.embedding.weight, 'n d -> d n'))

            min_encoding_indices = torch.argmin(d, dim=1)
            z_sub_q = self.embedding(min_encoding_indices)

            z_sub_q = z_sub_q.view(z_shape[0], z_shape[1], z_shape[2], -1)
            z_sub = z_sub.view(z_shape[0], z_shape[1], z_shape[2], -1)

            all_z_sub_q.append(z_sub_q)
            all_min_encoding_indices.append(min_encoding_indices)

            # compute loss for embedding
            if not self.legacy:
                loss_sub = self.beta * torch.mean((z_sub_q.detach() - z) ** 2) + \
                       torch.mean((z_sub_q - z_sub.detach()) ** 2)
            else:
                loss_sub = torch.mean((z_sub_q.detach() - z_sub) ** 2) + self.beta * \
                       torch.mean((z_sub_q - z_sub.detach()) ** 2)
            losses.append(loss_sub)

        z_q = torch.cat(all_z_sub_q, dim=-1)
        z = torch.cat(z, dim=-1)
        loss = torch.stack(losses).mean()
        min_encoding_indices = torch.cat(all_min_encoding_indices)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        perplexity, cluster_use = self.measure_perplexity(min_encoding_indices, self.n_e)

        return z_q, loss, perplexity, cluster_use, min_encoding_indices


    def split_feature_map(self, features):
        z = rearrange(features, 'b c h w -> b h w c').contiguous()
        z = torch.reshape(z, (z.shape[0], z.shape[1], z.shape[2], self.k_e, self.e_dim_seg))

        z = z.view(z.shape[0], -1, self.e_dim_seg)

        return z


    def init_codebook_by_clustering(self, features, evaluate_on_gpu=True, n_max=100000):

        n_feat_tot = features.shape[0]

        ### Prepare features
        features = features.astype(np.float32)

        ### select samples to use
        idx_to_use = np.random.choice(features.shape[0], np.min([features.shape[0], n_max]), replace=False)
        features = features[idx_to_use, :]

        print(f'Kmeans clustering: [n_feat={features.shape[0]}] [n_feat_tot={n_feat_tot}] [n_centroids={self.n_e}] [dim_centroid={features.shape[-1]}]')

        ### Init faiss
        faiss.omp_set_num_threads(20)
        res = None
        torch.cuda.empty_cache()
        if evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        ### Set CPU Cluster index
        cluster_idx = faiss.IndexFlatL2(features.shape[-1])
        if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
        kmeans = faiss.Clustering(features.shape[-1], self.n_e)
        kmeans.niter = 20
        kmeans.min_points_per_centroid = 1
        kmeans.max_points_per_centroid = 1000000000

        ### Train Kmeans
        kmeans.train(features, cluster_idx)
        centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(self.n_e, features.shape[-1])

        ### Init codebook
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(deepcopy(centroids)).float(), freeze=False)

        ### empty cache on GPU
        torch.cuda.empty_cache()

    #TODO disable for the moment to save computation, as it creates huge workload
    def measure_perplexity(self, predicted_indices, n_embed):  # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        # encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
        # avg_probs = encodings.mean(0)
        # perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        # cluster_use = torch.sum(avg_probs > 0)

        perplexity = torch.zeros(1)
        cluster_use = torch.zeros(1)
        return perplexity, cluster_use


class FactorizedVectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, vq_arch, n_e, e_dim, e_dim_latent, beta, e_init='random_uniform', block_to_quantize=-1, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.e_dim_latent = e_dim_latent
        self.beta = beta
        self.legacy = legacy
        self.e_init = e_init
        self.block_to_quantize = block_to_quantize
        self.vq_arch = vq_arch

        # factorization projectors
        self.proj_down = torch.nn.Linear(self.e_dim, self.e_dim_latent)
        self.proj_up = torch.nn.Linear(self.e_dim_latent, self.e_dim)

        self.embedding = nn.Embedding(self.n_e, self.e_dim_latent)
        if self.e_init == 'random_uniform':
            self.embedding.weight.data.uniform_(-100.0 / self.n_e, 100.0 / self.n_e)
        elif self.e_init == 'random_gaussian':
            self.embedding.weight.data.normal_(-100.0 / self.n_e, 100.0 / self.n_e)

        print(f'Initializeing VQ [VectorQuantization]')
        print(f'*** n_e = [{self.n_e}]')
        print(f'*** e_dim = [{self.e_dim}]')
        print(f'*** e_dim_latent = [{self.e_dim_latent}]')
        print(f'*** e_init = [{self.e_init}]')
        print(f'*** block_to_quantize = [{self.block_to_quantize}]')
        print(f'*** beta = [{self.beta}]\n')

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_latent = z.view(-1, self.e_dim)

        #project into factorization space
        z_latent = self.proj_down(z_latent)

        d = torch.sum(z_latent ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_latent, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_latent_q = self.embedding(min_encoding_indices)

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * torch.mean((z_latent_q.detach() - z_latent) ** 2) + \
                   torch.mean((z_latent_q - z_latent.detach()) ** 2)
        else:
            loss = torch.mean((z_latent_q.detach() - z_latent) ** 2) + self.beta * \
                   torch.mean((z_latent_q - z_latent.detach()) ** 2)

        # preserve gradients
        z_latent_q = z_latent + (z_latent_q - z_latent).detach()

        # back-project into embedding space
        z_q = self.proj_up(z_latent_q).view(z.shape)

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        perplexity, cluster_use = self.measure_perplexity(min_encoding_indices, self.n_e)

        return z_q, loss, perplexity, cluster_use, min_encoding_indices

    def measure_perplexity(self, predicted_indices, n_embed):  # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
        encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
        avg_probs = encodings.mean(0)
        perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
        cluster_use = torch.sum(avg_probs > 0)
        return perplexity, cluster_use
