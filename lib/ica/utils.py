from .classifiers_nx import LocalClassifier
from .classifiers_nx import RelationalClassifier
from .classifiers_nx import ICA
from .classifiers_nx import CountAggregator, ProportionalAggregator, ExistAggregator

import numpy as np
from math import log

import sys
import pdb
sys.path.append('../../')


def pick_aggregator(agg, domain_labels, directed):
    if agg == 'count':
        aggregator = CountAggregator(domain_labels, directed)
    if agg == 'prop':
        aggregator = ProportionalAggregator(domain_labels, directed)
    if agg =='exist':
        aggregator = ExistAggregator(domain_labels, directed)
    return aggregator

def create_map(graph, indices, co_predict = None):
    conditional_map = {}
    node_ids = list(graph.nodes())
    gnodes = graph.node

    if co_predict is not None:
        for i in range(len(node_ids)):
            node = node_ids[i]
            label = co_predict[i]
            conditional_map[node] = label

    for i in indices:
        conditional_map[i] = gnodes[i]['label']

    return conditional_map


# TODO: Move this method to appropriate location
def label_entropy(labels):
    ''' Computes entropy of label distribution. '''
    n_labels = float(len(labels))
    
    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    
    # Compute standard entropy.
    for i in probs:
        if i > 0:
            ent -= i * log(i, n_classes)

    return ent



# Not used
def split_train_test(data, test_ratio):
    data = np.array(data)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]



def draw_pca_scatter_plot(x, y, algoname):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.ylabel('component 2')
    plt.xlabel('component 1')
    
    plt.title('PCA plot for ' + algoname)

    plt.savefig('images/' + algoname + '.png', format="PNG")



def node_stats(L, jumps, features):
    import pdb
    LS = np.array(L)
    jumps = np.array(jumps)
    good_nodes_i = jumps > np.percentile(jumps, 80)
    good_nodes = LS[good_nodes_i]
    # bad_nodes_i = np.invert(good_nodes_i)
    bad_nodes_i = jumps < 0.0
    bad_nodes = LS[bad_nodes_i]

    if len(good_nodes) <= 1:
        print('not enough good points!')
        pdb.set_trace()
        return

    if len(bad_nodes) <= 1:
        print('not enough bad points!')
        pdb.set_trace()
        return

    def gen_points(nodes):
        filtered_features = []
        for i in range(nodes.shape[0]):
            filtered_features.append(features[ nodes[i] ])

        filtered_features = np.array(filtered_features)

        means = np.zeros(filtered_features.shape[1])
        stds = np.zeros(filtered_features.shape[1])
        for i in range(filtered_features.shape[1]):
            means[i] = np.mean(filtered_features[:, i])
            stds[i] = np.std(filtered_features[:, i])
            stds[i] = np.clip(stds[i], 0, means[i])
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(filtered_features.T)
        pca_points = pca.components_.T
        # return filtered_features, means, stds
        return filtered_features, pca_points, means, stds


    print('%d good nodes' % len(good_nodes))
    good_points, good_pca_points, good_means, good_stds = gen_points(good_nodes)

    print('%d bad nodes' % len(bad_nodes))
    bad_points, bad_pca_points, bad_means, bad_stds = gen_points(bad_nodes)

    stats = {}
    stats['good'] = (good_points, good_pca_points, good_means, good_stds)
    stats['bad'] = (bad_points, bad_pca_points, bad_means, bad_stds)

    return stats

    