import os
import pdb
import multiprocessing

import numpy as np
import pandas as pd
import networkx as nx

from numpy import linalg as LA

from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans


class HybridSampling:

    @staticmethod
    def feat_prop(graph, size, pool, features):
        g = graph
        A = np.array(nx.adjacency_matrix(g).todense())
        D = np.diag(np.array(A).sum(1))
        I = np.identity(len(g))
        S = np.sqrt(I + D) @ (I + A) @ np.sqrt(I + D)
        S = S ** 2
        
        def my_dist(a, b):
            return LA.norm(a - b)

        SX = S @ features

        p_ind = np.where(np.in1d(np.array(features.index), pool))[0]
        SX = SX.loc[p_ind]

        # arxiv version uses K-Medoids: https://arxiv.org/abs/1910.07567
        # sg_dists = pairwise_distances(SX, metric=my_dist, n_jobs=-1)
        # kmedoids = KMedoids(n_clusters=size, metric='precomputed').fit(sg_dists)
        # kmedoids = KMedoids(n_clusters=size, init='k-medoids++').fit(SX)
        # node_indices = np.array(SX.index)[kmedoids.medoid_indices_]
        # return list(features.index[node_indices])


        # NeurIPS workshop version uses K-Means: https://grlearning.github.io/papers/46.pdf
        p_features = features.loc[pool]

        num_jobs = min(16, multiprocessing.cpu_count())
        kmeans = KMeans(n_clusters=size, random_state=0, n_jobs=num_jobs).fit(SX)
        # node_indices = pairwise_distances_argmin(kmeans.cluster_centers_, p_df_features.values)
        node_indices = []
        for i in range(size):
            cc = kmeans.cluster_centers_[i]
            dists = pairwise_distances(cc.reshape(1, -1), p_features.values)
            sorted_indices = list(np.argsort(dists)[0])
            for si in sorted_indices:
                if si in node_indices:
                    continue
                node_indices.append(si)
                break
        
        selected = np.array(p_features.index)[node_indices]

        assert(len(selected) == len(set(selected)))

        return selected