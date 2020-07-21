import os
import operator, logging, pdb
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
import stellargraph as sg

from numpy import linalg as LA


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids

from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from stellargraph.layer import GCN
from sklearn import preprocessing, feature_extraction, model_selection

from lib.ica.utils import *
from lib.graph.GraphUtils import *
# from algo.keras_utils import macro_f1
from .relational_classifier import RelationalClassifier

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class GNN(RelationalClassifier):
    
    def __init__(self, data, sampling, sampling_params):
        RelationalClassifier.__init__(self, data, sampling, sampling_params)

        self.data.graph.remove_nodes_from(nx.isolates(self.data.graph))

        class_labels = nx.get_node_attributes(self.data.graph, "label")
        nx.set_node_attributes(self.data.graph, "paper", "tag")


    def cap_features(self, feat_indices):
        for k,v in self.data.feature_matrix.items():
            self.features[k] = operator.itemgetter(*feat_indices)(v)

        self.build_sgc_features(self.data.graph, self.features)


    def reduce_dimensions(self, df_features, num_dimensions=100):
        return df_features
        

    def build_sgc_features(self, g, feature_dict):
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

        self.SG = sg.StellarGraph(g, node_features=self.df_features, node_type_name='tag')
        self.generator = self.build_generator()

        target_encoding = feature_extraction.DictVectorizer(sparse=False)
        self.target_encoding = target_encoding.fit(self.df_targets.to_dict("records"))

    
    def build_generator(self):
        NotImplementedError


    def weighted_categorical_crossentropy(self, weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy
        
        Variables:
            weights: numpy array of shape (C,) where C is the number of classes
        
        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        
        source: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        """
        
        weights = K.variable(weights)
            
        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss
        
        return loss


    def train_clf(self, graph, L):
        raise NotImplementedError


    def predict(self, model, test_gen):
        raise NotImplementedError


    def test_accuracy(self, model, test_set):
        test_targets = self.target_encoding.transform(self.df_targets.loc[test_set].to_dict("records"))
        test_gen = self.generator.flow(test_set, test_targets)

        test_metrics = model.evaluate_generator(test_gen)

        y_true = test_targets.argmax(axis=1)
        y_predict = self.predict(model, test_gen).argmax(axis=1)
        
        # return test_metrics[1]
        return f1_score(y_true, y_predict, average='micro', labels=np.unique(y_predict))


    def execute(self, params, train_set, test_set):
        k = params['al_batch']
        B = params['al_budget']
        B_batch = params['budget_step']

        g = self.data.graph.copy()
        isolates = list(nx.isolates(g))
        g.remove_nodes_from(isolates)

        logging.info("Running algorithm: %s-%s" % (self.name, self.sampling.__name__))
        logging.debug("\t%-9s -> | %-6s" % ("Budget", "Acc"))
        logging.debug("\t" + "=" * 40)

        x_acc = []
        cc_acc = []
        co_acc = []
        timing = []
        jumps = []
        g_temp = g

        B_ = B_batch
        while B_ <= B:
            L, P = self.select_seed_nodes(g, train_set, k)

            start = datetime.now()
            self.select_nodes(L, P, B_, params)
            timing.append((datetime.now() - start).total_seconds())

            model = self.train_clf(g, L)
            a_cc = self.test_accuracy(model, test_set)
            
            logging.debug("\t%9.2f -> | %6.4f" % (B_, a_cc))
            cc_acc.append(a_cc)
            co_acc.append(a_cc)
            x_acc.append(B_)

            B_ += B_batch

        logging.debug("\t" + "=" * 40)

		# draw_acc_plot(x_acc, cc_acc, co_acc, isSS)
        accuracies = (np.array(x_acc), np.array(cc_acc), np.array(co_acc))
        return accuracies, timing, None 