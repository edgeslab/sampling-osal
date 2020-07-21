from copy import copy
from datetime import datetime

import networkx as nx
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from lib.ica import classifiers_nx as cnx
from lib.ica.utils import pick_aggregator

from lib.ica.utils import *
from lib.graph.GraphUtils import *
from .relational_classifier import RelationalClassifier


class wvRN(RelationalClassifier):
    
    def __init__(self, data, sampling, sampling_params):
        RelationalClassifier.__init__(self, data, sampling, sampling_params)
        self.name = 'wvRN'


    def wvrn_infer(self, g, L, class_priors, node):
        NN = list(g.neighbors(node))
        class_counts = {c:0 for c in self.df_targets['label'].unique()}
        for n in NN:
            if n in L:
                n_cls = g.node[n]['label']
                class_counts[n_cls] += 1
            else:
                for c in class_counts:
                    class_counts[c] += 0 if c not in class_priors else class_priors[c]
        class_sorted = [k for k, v in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)]
        
        return class_sorted[0]


    def test_accuracy(self, test_set, L):
        g = self.data.graph

        known_labels = np.array([g.node[v]['label'] for v in L])
        classes, cnt = np.unique(known_labels, return_counts=True)
        class_priors = dict(zip(classes, cnt / np.sum(cnt)))

        y_predict = []
        for node in test_set:
            pred = self.wvrn_infer(g, L, class_priors, node)
            y_predict.append(pred)

        y_true = list(map(lambda x: g.node[x]['label'], test_set))
        cc_accuracy = accuracy_score(y_true, y_predict)
        
        return f1_score(y_true, y_predict, average='micro', labels=np.unique(y_predict))


    def build_CC(self, scikit_clf):
        co_params = {'class_weight' : 'balanced', 'solver' : 'lbfgs', 'multi_class' : 'auto'}
        CO = cnx.LocalClassifier(scikit_clf, **co_params)

        CO_copy = cnx.LocalClassifier(scikit_clf, **co_params)
        Aggr = pick_aggregator('count', list(self.data.class_labels.keys()), False)
        RC = cnx.RelationalClassifier(scikit_clf, Aggr, True, **co_params)
        CC = cnx.ICA(CO_copy, RC)

        return CO, CC


    def execute(self, params, train_set, test_set):
        FEATURE_MATRIX = self.features
        
        g = self.data.graph.copy()
        # print 'subgraph:', nx.info(g)
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)
    
        CO, CC = self.build_CC(params['scikit_clf'])
        k = params['al_batch']
        B = params['al_budget']
        B_batch = params['budget_step']

        logging.info("Running algorithm: %s-%s" % (self.name, self.sampling.__name__))

        cc_acc = []
        x_acc = []
        timing = []

        B_ = B_batch
        while B_ <= B:
            L, P = self.select_seed_nodes(g, train_set, k)
            
            start = datetime.now()
            self.select_nodes(L, P, B_, params)
            timing.append((datetime.now() - start).total_seconds())

            a_cc = self.test_accuracy(test_set, L)
            
            logging.debug("\t%9.2f -> | %6.4f" % (B_, a_cc))
            cc_acc.append(a_cc)
            x_acc.append(B_)

            B_ += B_batch

        accuracies = (np.array(x_acc), np.array(cc_acc), np.array(cc_acc), )
        return accuracies, timing, None