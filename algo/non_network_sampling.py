import os
import pdb
import multiprocessing

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.cluster import KMeans


class NonNetworkSampling:

    @staticmethod
    def random(graph, size, pool):
        return np.random.choice(pool, size, replace = False)


    @staticmethod
    def kmeans(graph, size, pool, k, features):
        num_jobs = min(16, multiprocessing.cpu_count())
        kmeans = KMeans(n_clusters=k, random_state=0, n_jobs=num_jobs).fit(features.loc[pool])

        clusters = {}
        for k_ in range(k):
            clusters[k_] = np.array(pool)[(kmeans.labels_ == k_).nonzero()[0]]

        selection = []
        while size > 0:
            for k_ in range(k):
                if len(clusters[k_]) == 0:
                    continue
                v = np.random.choice(clusters[k_], 1, replace = False)[0]
                clusters[k_] = np.delete(clusters[k_], np.argwhere(clusters[k_]==v))
                selection.append(v)
                size -= 1
        
        return selection
        