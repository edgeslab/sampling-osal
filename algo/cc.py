from copy import copy
from datetime import datetime

import numpy as np
import networkx as nx

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from lib.ica import classifiers_nx as cnx
from lib.ica.utils import pick_aggregator

from lib.ica.utils import *
from lib.graph.GraphUtils import *
from .relational_classifier import RelationalClassifier


class CC(RelationalClassifier):
    
    def __init__(self, data, sampling, sampling_params):
        RelationalClassifier.__init__(self, data, sampling, sampling_params)
        self.name = 'CC'


    def train_clf(self, g, CO, CC, L):
        '''
            Train CO and CC classifier with updated labeled pool L
            Return new trained models
        '''
        
        FEATURE_MATRIX = self.features
        
        # For Semi-supervised
        CO.fit(g, L, FEATURE_MATRIX)

        # For ICA
        ss_predict = CO.predict(g, g.nodes(), FEATURE_MATRIX)
        # print ss_predict
        conditional_node_to_label_map = create_map(g, L, ss_predict)
        CC.fit(g, L, FEATURE_MATRIX, conditional_node_to_label_map)
        
        return CO, CC


    def test_accuracy(self, CO, clf, test_set, L, clf_type = "CC"):
        g = self.data.graph
        FEATURE_MATRIX = self.features
        
        y_predict = None

        if clf_type == "CC":
            ss_predict = CO.predict(g, g.nodes(), FEATURE_MATRIX)
            conditional_node_to_label_map = create_map(g, L, ss_predict)
            y_predict = clf.predict(g, test_set, conditional_node_to_label_map, FEATURE_MATRIX)
        elif clf_type == "CO":
            y_predict = clf.predict(g, test_set, FEATURE_MATRIX)

        y_true = list(map(lambda x: g.node[x]['label'], test_set))
        cc_accuracy = accuracy_score(y_true, y_predict)
        
        return f1_score(y_true, y_predict, average='micro', labels=np.unique(y_predict))
        # return cc_accuracy

    
    def extract_subgraph_nodes(self, graph, k, P, nhops):
        pg = graph.subgraph(P)
        
        all_neighbors = []
        for k_ in range(k): 
            v = np.random.choice(P, 1, replace = False)[0]
            neighbors, A = get_neighborhood_labels(pg, v, hops=nhops)

            if not neighbors:
                all_neighbors += [v]
            else:
                all_neighbors += neighbors
        all_neighbors = list(set(all_neighbors))

        return all_neighbors


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

        temp_co = copy(CO)
        cc_acc = []
        co_acc = []
        x_acc = []
        jumps = []
        timing = []

    
        B_ = B_batch
        while B_ <= B:
            L, P = self.select_seed_nodes(g, train_set, k)
            
            start = datetime.now()
            self.select_nodes(L, P, B_, params)
            timing.append((datetime.now() - start).total_seconds())

            CO, CC = self.train_clf(g, CO, CC, L)
            a_cc = self.test_accuracy(CO, CC, test_set, L)
            
            logging.debug("\t%9.2f -> | %6.4f" % (B_, a_cc))
            cc_acc.append(a_cc)
            x_acc.append(B_)

            B_ += B_batch

        accuracies = (np.array(x_acc), np.array(cc_acc), np.array(cc_acc), )
        return accuracies, timing, None