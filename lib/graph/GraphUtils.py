import pdb
import logging
import operator
import queue as que

import numpy as np
import networkx as nx

from scipy.linalg import eigh
from sklearn.cluster import SpectralClustering

from .kernel_utils import *



def gen_class_probs(C, CB):
    mu = 1.0/C
    class_probs = np.random.normal(loc=mu, scale=CB * mu, size=C)
    offset = np.min(class_probs)
    if offset < 0:
        class_probs += -offset + 0.01      # offset to remove negative probs
    class_probs /= np.sum(class_probs)

    one = np.where(class_probs > 0.9)
    if one[0].shape[0] != 0:
        return gen_class_probs(C, CB)

    logging.info("CP: %s, STD: %0.3f" % (' '.join(["%0.3f" % cp for cp in class_probs]), np.std(class_probs)))
    return class_probs


def build_feature_generator(numAttr, numNodes, dists):
    generators = []
    dists_a = np.random.choice(dists, numAttr)
    for i in range(len(dists_a)):
        if dists_a[i] == 'binomial':
            n, p = 1, np.random.random_sample()
            gen = (np.random.binomial, {'n': n, 'p': p, 'size': numNodes})
            generators.append(gen)
        elif dists_a[i] == 'normal':
            mu, sigma = i * 100, 20
            gen = (np.random.normal, {'loc': mu, 'scale': sigma, 'size': numNodes})
            generators.append(gen)
        elif dists_a[i] == 'beta':
            a, b = np.random.random_sample(), np.random.random_sample()
            gen = (np.random.beta, {'a': a, 'b': b, 'size': numNodes})
            generators.append(gen)
            
    return generators


def gen_graph(C=3, CB=0.0, N=15, D=1.0, H=1.0, A=10, dists=['binomial'], seed=42):
    np.random.seed(seed)
    
    eps = 1e-3
    CB = 1 - CB
    CB = 1e-3 if abs(CB - 0) < 1e-3 else CB

    G = nx.Graph()
    V = range(10)
    E = []
    maxE = (N * (N - 1) / 2) * 1.0

    classes = [c for c in range(C)]
    class_probs = gen_class_probs(C, CB)
    # class_probs = [0.5, 0.5]

    L = list(np.random.choice(classes, size=len(V), p=class_probs))
    
    density = len(E) / maxE
    while len(V) < N or density < D:
        all_indices = set(range(len(V)))
        maxV = (len(V) * (len(V) - 1) / 2)
        
        u = np.random.choice(V, 1)[0]
        ulabel = L[V.index(u)]
        
        same = False
        if np.random.random_sample() <= H:
            same = True

        if len(V) == N or ((np.random.random_sample() <= D) and (len(E) < maxV)):
            #Add link to existing node
            u_indices = set([i for i, x in enumerate(L) if x == ulabel])

            attempt = 0
            v = -1
            while True:
                candidate_indices = u_indices if same else (all_indices - u_indices)
                if len(candidate_indices) == 0:
                    break

                v = np.random.choice(list(candidate_indices), 1)[0]
                
                if attempt > len(V)/2:
                    break

                attempt += 1
                if v == u:
                    continue

                if ((u, v) in E) or ((v, u) in E):
                    continue
                
                E.append((u, v))
                break

        else:   #Add link to new node
            v = len(V)
            vlabel = -1

            if same:
                vlabel = ulabel
            else:
                while True:
                    c = np.random.choice(classes, size=1, p=class_probs)[0]
                    if c != ulabel:
                        vlabel = c
                        break
            
            V.append(v)
            L.append(vlabel)
            E.append((u,v))                
        
        density = len(E) / maxE
        
    G.add_nodes_from(V)
    G.add_edges_from(E)
    nx.set_node_attributes(G, 'label', dict(zip(V, L)))
    
    attr_matrix = np.zeros(shape=(len(V), A), dtype=float)
    labels = np.array(L)
    
    for c in classes:
        mask = labels == c
        feat_gen = build_feature_generator(A, np.sum(mask), dists)
        temp = attr_matrix[mask]

        for i in range(A):
            gen = feat_gen[i][0]
            params = feat_gen[i][1]
            temp[:, i] = gen(**params)

        attr_matrix[mask] = temp
        
    features = {}
    for i in range(len(attr_matrix)):
        features[i] = attr_matrix[i]
    
    class_dict = dict(zip(classes, range(len(classes))))
    logging.info(nx.info(G))

    return G, class_dict, features



def sample(g):
    lcg = max(nx.connected_component_subgraphs(g), key=len)
    return lcg


def cluster_features(g, features, num_clusters = 8):
    import pdb

    node_ids = np.asarray(g.nodes())
    allpoints = np.zeros(shape=(node_ids.shape[0], len(features[node_ids[0]]) ), dtype=np.float)
    for i, f in enumerate(node_ids):
        allpoints[i] = np.asarray(features[f])

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++').fit(allpoints)
    labels = kmeans.labels_

    clusters = []
    for ci in range(num_clusters):
        node_indices = np.where(labels == ci)[0]
        c = g.subgraph(node_ids[node_indices])
        clusters.append(c)

    return clusters


# Clustering by [Newman'06]
def modularity_cluster(g, cap = 200):
    '''
    Cluster the graph based on Newman's algorithm.
    M. E. J. Newman, "Modularity and community structure in networks",
    Proc. Natl. Acad. Sci. USA, vol. 103, pp. 8577-8582, 2006.

    Useful Links:
        - Modularity matrix: https://goo.gl/AQ7ZUC
        - Complex number in Eigenvalue issue: https://goo.gl/APvxRn
        - Benchmark Eigenvalues with Matlab: https://goo.gl/3giVRU
    '''

    if(g.number_of_nodes() < cap):
        return []

    mm = nx.modularity_matrix(g)
    e = eigh(mm, eigvals_only=True)

    pmask = np.array([i > 0 for i in e])
    nmask = np.array(list(map(operator.not_, pmask)))

    indicies = np.array(g.nodes())
    ep = indicies[pmask]
    en = indicies[nmask]

    if(len(ep) < cap or len(en) < cap):
        return [g]

    gpart1 = g.subgraph(ep)
    gpart2 = g.subgraph(en)

    # logging.debug("cluster: (%d, %d) -> (%d, %d) (%d, %d)" % (g.number_of_nodes(), g.number_of_edges(), gpart1.number_of_nodes(), gpart1.number_of_edges(), gpart2.number_of_nodes(), gpart2.number_of_edges()))

    return modularity_cluster(gpart1, cap) + modularity_cluster(gpart2, cap)


def rk_cluster(g, df_features, num_clusters):
    features = df_features.values
    A = np.array(nx.adjacency_matrix(g).todense())
    sg_dists = relational_kernel(features, A)
    sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(sg_dists)

    clusters = []
    for i in range(num_clusters):
        ci = np.nonzero(sc.labels_ == i)[0]
        nodes = np.array(df_features.index)[ci]
        clusters.append(g.subgraph(nodes).copy())

    return clusters


def labels_stat(L, graph):
    g = graph.subgraph(L)

    # d = g.degree(L).values()
    d = dict(g.degree(L)).values()
    avgd = float(sum(d))/len(d)

    labels = nx.get_node_attributes(g,'label').values()

    c_hist = {}
    for c in labels:
        if c in c_hist:
            c_hist[c] += 1
        else:
            c_hist[c] = 1

    class_dist = np.array(list(c_hist.values()), dtype=float)
    class_dist /= np.sum(class_dist)
    logging.debug(class_dist)
    logging.debug("avg degree: %0.04f, class STD: %0.02f" % (avgd, np.std(class_dist)))
    
    return np.std(list(c_hist.values()))


def homophily(graph):
    labels = nx.get_node_attributes(graph,'label')
    # print labels[0]
    c = 0.0
    for e in graph.edges():
        if labels[e[0]] == labels[e[1]]:
            c += 1
    return c / len(graph.edges())


def density(graph):
    N = graph.number_of_nodes()
    maxE = (N * (N - 1) / 2) * 1.0
    return graph.number_of_edges() / maxE


def normalize_dict_features(dict_features):
    keys = dict_features.keys()
    vals = dict_features.values()
    vals_array = np.asarray(vals)
    for i in range(vals_array.shape[1]):
        vals_array[:, i] /= np.sum(vals_array[:, i])
    return dict(zip(keys, vals_array.tolist()))


def build_struct_features(graph):
    from datetime import datetime
    start = datetime.now()

    struct_feature = {}
    feature_labels = []

    for k in graph.node:
        struct_feature[k] = []

    def add_struct_feature(feature, label):
        for k in feature:
            struct_feature[k].append(feature[k])
        feature_labels.append(label)

    add_struct_feature(graph.degree(), 'degree')
    add_struct_feature(nx.degree_centrality(graph), 'degree c')
    add_struct_feature(nx.betweenness_centrality(graph), 'between c')
    add_struct_feature(nx.closeness_centrality(graph), 'close c')
    add_struct_feature(nx.load_centrality(graph), 'load c')
    add_struct_feature(nx.triangles(graph), 'trianlges')
    # add_struct_feature(nx.pagerank(graph))
    # add_struct_feature(nx.clustering(graph))
    # add_struct_feature(nx.average_clustering(graph))

    # citeseer
    #  H: [ 0.10371596  0.10868468  0.17090806  0.15738537  0.14351598  0.12823917 0.18755079]
    #  A: [ 0.06851971  0.08760618  0.14262233  0.27602311  0.12954667  0.22816851 0.0675135 ]

    # cora
    # H: [ 0.11170605  0.12219257  0.15195672  0.16988706  0.14448671  0.14731139 0.15245949]
    # A: [ 0.09526744  0.06283002  0.16778552  0.17890458  0.17042309  0.22219445 0.1025949 ]



    elapsed = (datetime.now() - start).seconds
    logging.debug('build_struct_features completed in %d seconds' % elapsed)

    return struct_feature, feature_labels



def edge_stats(G, L):
    import pdb
    lg = G.subgraph(L)
    E = list(lg.edges_iter())
    ncc = nx.number_connected_components(lg)
    print('num connected component:', ncc)

    if not E:
        print('average edge betweenness: NaN')
        return

    # H: 40
    # A: 87

    eb = nx.edge_betweenness_centrality(G)
    avg_eb = 0.0

    for e in eb:
        # pdb.set_trace()
        u, v = e[0], e[1]
        if (u in L) and (v in L):
            avg_eb += eb[e]
    if len(eb) > 0:
        avg_eb /= len(E)
    # avg_eb = sum(eb.values()) / len(eb)
    print('average edge betweenness:', avg_eb)

    # H: 0.0084219858156
    # A: 0.000338915470494


    # H:
    # [ 0.03  0.16  0.43  0.12  0.19  0.07]
    # avg degree: 2.3200, class STD: 0.13

    # A:
    # [ 0.04  0.06  0.23  0.22  0.32  0.13]
    # avg degree: 0.2600, class STD: 0.10


def dump_graph_data(graph, train_set, labeled_nodes, file_edge, file_node):
    # pdb.set_trace()

    nx.write_edgelist(graph, path=file_edge, delimiter=',')
    
    is_labeled = dict(zip(graph.nodes(), [0] * graph.number_of_nodes()))
    for n in labeled_nodes:
        is_labeled[n] = 1

    class_labels = nx.get_node_attributes(graph, 'label')
    class_map = {}
    for i,v in enumerate(set(class_labels.values())):
        class_map[v] = i

    is_train = dict(zip(graph.nodes(), [0] * graph.number_of_nodes()))
    for n in train_set:
        is_train[n] = 1

    with open(file_node, 'w') as fn:
        for n in graph.nodes():
            fn.write('%s,%d,%d,%d\n' % (n, class_map[ class_labels[n] ], is_labeled[n], is_train[n]))



def get_neighborhood_nodes(graph, node, hops=1):
    N = []
    h = 1
    preds = [node]
    while h <= hops:
        nn = []
        for n in preds:
            nn += graph.neighbors(n)

        nn = filter(lambda x: x not in N, nn)
        preds = nn
        N += nn
        h += 1
    
    return N


def get_neighborhood_labels(graph, node, hops=1):
    N = []
    h = 1
    preds = [node]
    while h <= hops:
        nn = []
        for n in preds:
            nn += graph.neighbors(n)
            
        # nn = filter(lambda x: x not in N, nn)
        nn = list(set(nn) - set(N))
        preds = nn
        N += nn
        h += 1

    sg = graph.subgraph(N)

    if sg.number_of_edges() == 0:
        return N, np.zeros(shape=(len(N), len(N)), dtype=int)


    return N, np.asarray(nx.adjacency_matrix(graph.subgraph(N)).todense(), dtype=int)



def random_walk(G, v, walkLength=10):
    walk = [v]
    walkLength -= 1
    for i in range(walkLength):
        nb = G.neighbors(v)
        if (nb is None) or len(nb) == 0:
            return walk
        v = np.random.choice(nb, 1)[0]
        try_count = len(nb)
        while (v in walk) and try_count >= 0:
            v = np.random.choice(nb, 1)[0]
            try_count -= 1
        if v in walk:
            return walk
        walk.append(v)
    return walk 

def random_walk_unique(G, v, walkLength=10):
    walk = random_walk(G, v, walkLength)
    retry = 3
    while (len(walk) != walkLength) and retry >= 0 :
        walk = random_walk(G, v, walkLength)
        retry -= 1

    if retry < 0:
        return None

    return walk




def get_neighborhood_random(graph, seed_node, hops=1, max_nodes=1):
    neighbors = get_neighborhood_nodes(graph, seed_node, hops)

    if (not neighbors) or (len(neighbors) == 0):
        return None, None

    cap = min(max_nodes, len(neighbors))
    random_nodes = np.random.choice(neighbors, cap)

    sg = graph.subgraph(random_nodes)
    sg = sample(sg)
    random_neighbors = list(sg.node)

    if sg.number_of_edges() == 0:
        return random_neighbors, np.zeros(shape=(len(random_neighbors), len(random_neighbors)), dtype=int)

    return random_neighbors, np.asarray(nx.adjacency_matrix(sg).todense(), dtype=int)


# def snowball_sampling(g, seed, P, maxsize=50):
#     """this function returns a set of nodes equal to maxsize from g that are 
#     collected from around seed node via snownball sampling"""

#     if g.number_of_nodes() < maxsize:
#         return set()

#     subgraph = set([seed])
    
#     q = que.Queue()
#     q.put(seed)
#     mark = {seed : True}
#     while not q.empty():
#         v = q.get()
#         for node in g.neighbors(v):
#             if node in mark:
#                 continue
#             if len(subgraph) < maxsize:
#                 q.put(node)
#                 if node in P:
#                     subgraph.add(node)
#                 mark[node] = True
#             else :
#                 return subgraph

#     return subgraph



# def forest_fire_sampling(g, seed, p, pool, maxsize=50):
#     """this function returns a set of nodes from pool equal to maxsize from g that are 
#     collected from around seed node via forest fire sampling with visit probability = p"""

#     if g.number_of_nodes() < maxsize:
#         return set()

#     subgraph = set([seed])
    
#     q = que.Queue()
#     q.put(seed)
#     mark = {seed : True}
#     while not q.empty():
#         v = q.get()
#         for node in g.neighbors(v):
#             if np.random.rand() > p:
#                 continue

#             if node in mark:
#                 continue
#             if len(subgraph) < maxsize:
#                 q.put(node)
#                 if node in pool:
#                     subgraph.add(node)
#                 mark[node] = True
#             else :
#                 return subgraph

#     return subgraph