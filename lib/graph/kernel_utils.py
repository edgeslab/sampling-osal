import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import scale
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel


def infer_gamma(A):

    if np.max(np.max(A)) <= 1e-5:
        return 0.0

    A = scale(A)
    dist_matrix = euclidean_distances(A, A, None, squared=True)
    dist_vector = dist_matrix[np.nonzero(np.tril(dist_matrix))]
    dist_median = np.median(dist_vector)
    return dist_median


def base_kernel(A, s=1.):
    """ Compute radial basis function kernel.

    Parameters:
        A -- Feature matrix.
        s -- Scale parameter (positive float, 1.0 by default).
        
    Return:
        K -- Radial basis function kernel matrix.

    Source: https://github.com/gzampieri/Scuba/blob/master/compute_kernel.py
    """

    gamma = infer_gamma(A)
    K = rbf_kernel(A, None, gamma*s)
    return K


def relational_kernel(V, A, samples=[]):
    K = base_kernel(V)
    D = np.diag(1 / np.array(A).sum(1))
    if samples:
        Ds = D[samples, :][:, samples]
        return Ds @ A[samples, :] @ K @ A[:, samples] @ Ds
    return D @ A @ K @ A @ D


def rk_clustering(graph, df_features, pool, num_clusters):
    g = graph
    V = df_features.values
    A = np.array(nx.adjacency_matrix(g).todense())

    p_ind = np.in1d(pool, np.array(df_features.index)).nonzero()[0]
    k_sim = relational_kernel(V, A, samples=list(p_ind))
    # kmedoids = KMedoids(n_clusters=B-len(L), init='k-medoids++', metric='precomputed').fit(sg_dists)

    sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(k_sim)

    cluster_centers = []
    for i in range(num_clusters):
        c_sim = k_sim[sc.labels_ == i, :][:, sc.labels_ == i]
        c_center = np.argmax(c_sim.mean(axis=0))
        c_center = np.nonzero(sc.labels_ == i)[0][c_center]
        cluster_centers.append(np.array(df_features.index)[c_center])

    return cluster_centers