import pdb
import os
import itertools
import logging
import networkx as nx
import numpy as np
import pandas as pd


PROPERTY_NAME 		= "name"
PROPERTY_LABEL 		= "label"


def loadCSV(nodes_file = "nodes.csv", edges_file = "edges.csv", features_file = "features.csv"):
    nodes = pd.read_csv(nodes_file, sep=',', header=None, names=[PROPERTY_NAME, PROPERTY_LABEL])
    nodes['label'] = nodes['label'].astype(str)
    
    edges = pd.read_csv(edges_file, sep=',', header=None, names=['a', 'b'])
    edges = edges.loc[edges['a'].isin(nodes[PROPERTY_NAME]) & edges['b'].isin(nodes[PROPERTY_NAME])]

    # G = nx.from_pandas_dataframe(edges, 'a', 'b')
    G = nx.from_pandas_edgelist(edges, 'a', 'b')

    logging.info("Graph Loaded: %s", nx.info(G))

    labels_dict = nodes.set_index(PROPERTY_NAME).T.to_dict('records')[0]
    # nx.set_node_attributes(G, PROPERTY_LABEL, labels_dict)
    nx.set_node_attributes(G, labels_dict, PROPERTY_LABEL)

    features = pd.read_csv(features_file, sep=',', header=None, index_col=0, low_memory=False).T
    num_feat = features.shape[0]
    # cap_feat = int(num_feat * feat_thresh)
    # cap_feat = int(feat_thresh)

    # feat_indices = np.random.choice(range(num_feat), cap_feat, replace=False)
    # features = features.iloc[feat_indices]
    # logging.info('Feature dimension: %d', features.shape[0])

    unique_labels = nodes.label.unique()
    class_labels_dict = dict(zip(unique_labels, range(len(unique_labels))))

    features_dict = features.to_dict('list')

    return G, class_labels_dict, features_dict