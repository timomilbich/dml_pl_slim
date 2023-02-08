import numpy as np
import faiss
import torch
from tqdm import tqdm
import time
import copy

from metrics import select

class MetricComputer():
    def __init__(self, metric_names, n_classes, evaluate_on_gpu, num_workers):
        self.n_classes       = n_classes
        self.evaluate_on_gpu = evaluate_on_gpu
        self.metric_names    = metric_names
        self.num_workers = num_workers
        self.list_of_metrics = [select(metricname) for metricname in metric_names]
        self.requires        = [metric.requires for metric in self.list_of_metrics]
        self.requires        = list(set([x for y in self.requires for x in y]))

    def compute_standard(self, features, target_labels, device):

        ###
        target_labels = np.hstack(target_labels.cpu().detach().numpy()).reshape(-1,1)
        features = features.cpu().detach().numpy().astype(np.float32)
        computed_metrics = dict()

        ### Init faiss
        faiss.omp_set_num_threads(self.num_workers)
        res = None
        torch.cuda.empty_cache()
        if self.evaluate_on_gpu:
            res = faiss.StandardGpuResources()

        if 'kmeans' in self.requires or 'kmeans_nearest' in self.requires:
            ### Set CPU Cluster index
            cluster_idx = faiss.IndexFlatL2(features.shape[-1])
            if res is not None: cluster_idx = faiss.index_cpu_to_gpu(res, 0, cluster_idx)
            kmeans            = faiss.Clustering(features.shape[-1], self.n_classes)
            kmeans.niter = 20
            kmeans.min_points_per_centroid = 1
            kmeans.max_points_per_centroid = 1000000000
            ### Train Kmeans
            kmeans.train(features, cluster_idx)
            centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(self.n_classes, features.shape[-1])


        if 'kmeans_nearest' in self.requires:
            faiss_search_index = faiss.IndexFlatL2(centroids.shape[-1])
            if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
            faiss_search_index.add(centroids)
            _, computed_cluster_labels = faiss_search_index.search(features, 1)

        if 'nearest_features' in self.requires:
            faiss_search_index  = faiss.IndexFlatL2(features.shape[-1])
            if res is not None: faiss_search_index = faiss.index_cpu_to_gpu(res, 0, faiss_search_index)
            faiss_search_index.add(features)

            max_kval            = np.max([int(x.split('@')[-1]) for x in self.metric_names if 'recall' in x])
            _, k_closest_points = faiss_search_index.search(features, int(max_kval+1))
            k_closest_classes   = target_labels.reshape(-1)[k_closest_points[:,1:]]

        ###
        if self.evaluate_on_gpu:
            features = torch.from_numpy(features).to(device)

        start = time.time()
        for metric in self.list_of_metrics:
            input_dict = {}
            if 'features' in metric.requires:         input_dict['features'] = features
            if 'target_labels' in metric.requires:    input_dict['target_labels'] = target_labels
            if 'kmeans' in metric.requires:           input_dict['centroids'] = centroids
            if 'kmeans_nearest' in metric.requires:   input_dict['computed_cluster_labels'] = computed_cluster_labels
            if 'nearest_features' in metric.requires: input_dict['k_closest_classes'] = k_closest_classes
            computed_metrics[metric.name] = metric(**input_dict)

        torch.cuda.empty_cache()

        return computed_metrics