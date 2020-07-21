import os
import pdb
import sys
import math
import logging

from copy import copy
from abc import ABCMeta
from random import random 
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from lib.ica import classifiers_nx as cnx
from lib.ica.utils import *
from lib.graph.GraphUtils import *



class CC_ALFNET:
    __metaclass__ = ABCMeta

    def __init__(self, data):
        self.data = data
        # self.train_graph = None
        self.name = 'CC_ALFNET'
        self.features = {}


    def cap_features(self, feat_indices):
        for k,v in self.data.feature_matrix.items():
            self.features[k] = operator.itemgetter(*feat_indices)(v)
        
        self.build_pandas_features(self.data.graph, self.features)


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
        self.df_targets = F[['label']]


    def disagreement(self, CO, CC, Ci, L, P, isSS = False):
        '''
            Calculate the disageement score between CO and CC for cluster Ci given labeled pool L
            Return the disagreement score
        '''
        # g = self.train_graph
        g = self.data.graph
        FEATURE_MATRIX = self.features
        CLASS_LABELS = self.data.class_labels

        if Ci.number_of_nodes() == 0:
            return 0.0

        nodes = list(Ci.nodes())
        co_predict = CO.predict(g, nodes, FEATURE_MATRIX)

        co_ss_predict = None
        if isSS:
            co_ss_predict = CO.predict(g, g.nodes(), FEATURE_MATRIX)

        conditional_node_to_label_map = create_map(g, L, co_ss_predict)
        cc_predict = CC.predict(g, nodes, conditional_node_to_label_map, FEATURE_MATRIX)

        # TODO: optimize this
        majority_dict = {}
        majority_count = 0
        majority_label = ''
        for i in nodes:
            if i in L:
                l = Ci.node[i]['label']
                if l in majority_dict:
                    majority_dict[l] += 1
                else:
                    majority_dict[l] = 1
                
                if majority_dict[l] > majority_count:
                    majority_count = majority_dict[l]
                    majority_label = l
        
        dscore = 0.0
        for i in range(len(nodes)):

            Di = []
            if majority_label != '':
                Di.append( CLASS_LABELS[majority_label] )

            if nodes[i] in L:
                continue
            
            Di.append( CLASS_LABELS[co_predict[i]] )
            Di.append( CLASS_LABELS[cc_predict[i]] )

            #TODO: look for better alternatives
            dscore += label_entropy(Di)

        return dscore


    def train_clf(self, CO, CC, L, isSS = False):
        '''
            Train CO and CC classifier with updated labeled pool L
            Return new trained models
        '''

        # g = self.train_graph
        g = self.data.graph
        FEATURE_MATRIX = self.features

        CO.fit(g, L, FEATURE_MATRIX)

        ss_predict = None
        if isSS:
            ss_predict = CO.predict(g, g.nodes(), FEATURE_MATRIX)
        
        conditional_node_to_label_map = create_map(g, L, ss_predict)
        CC.fit(g, L, FEATURE_MATRIX, conditional_node_to_label_map)

        return CO, CC


    def test_accuracy(self, CO, clf, test_set, L, clf_type = "CC", isSS = False):
        
        g = self.data.graph
        FEATURE_MATRIX = self.features

        y_predict = None

        if clf_type == "CC":
            ss_predict = None
            if isSS:
                ss_predict = CO.predict(g, g.nodes(), FEATURE_MATRIX)
            conditional_node_to_label_map = create_map(g, L, ss_predict)
            y_predict = clf.predict(g, test_set, conditional_node_to_label_map, FEATURE_MATRIX)
        elif clf_type == "CO":
            y_predict = clf.predict(g, test_set, FEATURE_MATRIX)

        y_true = list(map(lambda x: g.node[x]['label'], test_set))
        cc_accuracy = accuracy_score(y_true, y_predict)

        return f1_score(y_true, y_predict, average='micro', labels=np.unique(y_predict))
        # return cc_accuracy


    def get_neighborhood_clusters(self, graph, k, P, C, nhops):
        #update predicted labels
        pg = graph.subgraph(P)
        if len(pg.node.keys()) != len(P):
            pdb.set_trace()

        all_neighbors = []
        # select nodes based on predicted labels
        for k_ in range(k): 
            v = np.random.choice(P, 1, replace = False)[0]
            neighbors, A = get_neighborhood_labels(pg, v, hops=nhops)

            if (not neighbors) or (np.sum(A) == 0):
                pass
            else:
                all_neighbors += neighbors

        all_neighbors = list(set(all_neighbors))

        CN = [[] for i in range(len(C))]
        for node in all_neighbors:
            for i in range(len(C)):
                if node in C[i]:
                    CN[i].append(node)

        for i in range(len(C)):
            CN[i] = graph.subgraph(CN[i])

        return CN


    def select_seed_nodes(self, g, train_set, k):
        P = list(train_set)	# Pool of unlabeled nodes

        nClasses = 0
        while nClasses < 2:
            L = list(np.random.choice(P, k, replace = False))
            nClasses = len(set([g.node[v]['label'] for v in L]))

        for v in L:
            P.remove(v)

        return L, P


    def get_clusters(self, g, num_clusters):
        return modularity_cluster(g, 0.8 * (g.number_of_nodes() / num_clusters))


    def build_CC(self, scikit_clf):
        co_params = {'class_weight' : 'balanced', 'solver' : 'lbfgs', 'multi_class' : 'auto'}
        CO = cnx.LocalClassifier(scikit_clf, **co_params)

        CO_copy = cnx.LocalClassifier(scikit_clf, **co_params)
        Aggr = pick_aggregator('count', list(self.data.class_labels.keys()), False)
        RC = cnx.RelationalClassifier(scikit_clf, Aggr, True, **co_params)
        CC = cnx.ICA(CO_copy, RC)

        return CO, CC


    def execute(self, params, train_set, test_set):
        '''
        g	= Graph
        CO 	= Content only classifier
        CC 	= COllective classifier
        k	= Batch size
        B 	= Budget
        P	= Pool of unlabeled examples
        '''

        CO, CC = self.build_CC(params['scikit_clf'])
        k = params['al_batch']
        B = params['al_budget']
        B_batch = params['budget_step']
        num_clusters = 8 #params['NC']

        # g = self.data.graph
        g = self.data.graph.copy()
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)
        # self.train_graph = g
        # train_nodes = list(g.nodes)

        logging.debug("Running %s with: Batch size %d, Budget %d, Vertices: %d, Edges: %d" % (self.name, k, B, g.number_of_nodes(), g.number_of_edges()))
        
        C = self.get_clusters(g, num_clusters)

        logging.info("Entering the main loop of %s" % (self.name))
        logging.debug("\t%-9s -> %-55s | %-6s | %-6s" % ("Budget(%)", "Disagreement scores", "Acc CO", "Acc CC"))
        logging.debug("\t" + "=" * 86)

        cc_acc = []
        co_acc = []
        timing = []
        jumps = []
        x_acc = []
        isSS = True

        def test_result(CO, CC, L):
            a_cc = self.test_accuracy(CO, CC, test_set, L, "CC", isSS)
            a_co = self.test_accuracy(CO, CO, test_set, L, "CO")
            return a_cc, a_co


        def append_result(a_cc, a_co, L):
            timing.append((datetime.now() - start).total_seconds()/60.0)
            cc_acc.append(a_cc)
            co_acc.append(a_co)
            x_acc.append(len(L))

        start = datetime.now()

        B_ = B_batch
        while B_ <= B:
            score = [0] * len(C)
            L, P = self.select_seed_nodes(g, train_set, k)
            CO, CC = self.train_clf(CO, CC, L, isSS)

            while len(L) < B_:

                for i in range(len(C)):
                    score[i] = self.disagreement(CO, CC, C[i], L, P, isSS)

                #TODO: exp on sort order
                Ck = self.get_sorted_clusters(C, score, k)
                
                for Ci in Ck:
                    nodes = Ci.nodes()
                    times = int(k / len(Ck))
                    for j in range(times):
                        v = np.random.choice(P, 1)[0]
                        #TODO: select node based on disagreement score
                        L.append(v)
                        P.remove(v)

                CO, CC = self.train_clf(CO, CC, L, isSS)

            a_cc, a_co = test_result(CO, CC, L)

            ds = (' '.join(['%6.2f' % s for s in score]))
            logging.debug("\t%9.2f -> %s | %6.4f | %6.4f" % (len(L), ds, a_co, a_cc))

            timing.append((datetime.now() - start).total_seconds())
            cc_acc.append(a_cc)
            co_acc.append(a_co)
            x_acc.append(B_)

            B_ += B_batch

        accuracies = (np.array(x_acc), np.array(cc_acc), np.array(co_acc))
        return accuracies, timing, None


    def get_sorted_clusters(self, C, score, k):
        return [x for _,x in sorted(zip(score,C), key=lambda x: x[0], reverse=True)][: min(k, len(C))]


        
