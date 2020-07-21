from copy import copy
from datetime import datetime

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import networkx as nx
import numpy as np


from lib.ica.utils import *
from lib.graph.GraphUtils import *


class RelationalClassifier:
    
    def __init__(self, data, sampling, sampling_params):
        self.data = data
        self.name = 'RC'
        self.train_graph = None
        self.features = {}
        self.sampling = sampling
        self.sparams = sampling_params


    def cap_features(self, feat_indices):
        for k,v in self.data.feature_matrix.items():
            self.features[k] = operator.itemgetter(*feat_indices)(v)

        self.build_pandas_features(self.data.graph, self.features)

    
    def reduce_dimensions(self, df_features, num_dimensions=100):
        return df_features

    
    def build_pandas_features(self, g, feature_dict):
        num_features = len(list(feature_dict.values())[0])
        feature_names = ["w_{}".format(ii) for ii in range(num_features)]
        column_names = feature_names + ["label"]

        features = {}
        for f in column_names:
            features[f] = []

        for s in g.nodes():
            for i in range(num_features):
                features["w_{}".format(i)].append(feature_dict[s][i])
            features['label'].append(g.nodes()[s]['label'])

        F = pd.DataFrame(data=features, index=g.nodes())
        self.df_features = F[feature_names]
        self.df_targets = F[['label']].astype(str)

        self.df_features = self.reduce_dimensions(self.df_features)


    def select_seed_nodes(self, g, train_set, k):
        P = list(train_set)	# Pool of unlabeled nodes

        nClasses = 0
        while nClasses < 2:
            L = list(np.random.choice(P, k, replace = False))
            nClasses = len(set([g.node[v]['label'] for v in L]))

        for v in L:
            P.remove(v)

        return L, P


    def select_nodes(self, L, P, B, params):
        params_dict = {
            'k'         :   params['al_batch'],
            'features'  :   self.df_features
        }

        for p in self.sparams:
            self.sparams[p] = params_dict[p] if self.sparams[p] is None else self.sparams[p]

        selection = self.sampling(graph=self.data.graph, size=B-len(L), pool=P, **self.sparams)
        
        for v in selection:
            L.append(v)
            P.remove(v)


    def execute(self, params, train_set, test_set):
        raise NotImplementedError