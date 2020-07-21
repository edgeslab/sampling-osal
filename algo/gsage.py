import os
import operator, logging, pdb
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, losses, metrics, Model, regularizers

from stellargraph.layer import GraphSAGE
from stellargraph.mapper import GraphSAGENodeGenerator

from .gnn import GNN


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class GSAGE(GNN):
    
    def __init__(self, data, sampling, sampling_params):
        GNN.__init__(self, data, sampling, sampling_params)
        self.name = 'GSAGE'


    def build_generator(self):
        batch_size = 50
        num_samples = [10, 5]
        return GraphSAGENodeGenerator(self.SG, batch_size, num_samples)


    def predict(self, model, test_gen):
        return model.predict(test_gen)


    def train_clf(self, graph, L):
        '''
			Train GraphSage model with updated labeled pool L
			Return new trained model
		'''
        train_targets = self.target_encoding.transform(self.df_targets.loc[L].to_dict("records"))
        train_gen = self.generator.flow(L, train_targets)

        gsage = GraphSAGE(
            layer_sizes=[32, 32],
            generator=self.generator,
            bias=True,
            dropout=0.5
        )

        x_inp, x_out = gsage.build()
        predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

        class_support = dict(Counter(self.df_targets.loc[L]["label"]))
        classes = sorted(self.data.class_labels)
        counts = [class_support[c] if c in class_support else 0 for c in classes ]
        weights = np.array(counts) / np.sum(counts)
        weighted_loss = self.weighted_categorical_crossentropy(weights)

        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.2),
            # loss=losses.categorical_crossentropy,
            loss=weighted_loss,
            metrics=["acc"],
        )

        # if not os.path.isdir("model_logs"):
        #     os.makedirs("model_logs")
        # es_callback = EarlyStopping(
        #     monitor="acc", patience=50
        # )  # patience is the number of epochs to wait before early stopping in case of no further improvement
        # mc_callback = ModelCheckpoint(
        #     "model_logs/best_model.h5", monitor="acc", save_best_only=True, save_weights_only=True
        # )

        history = model.fit_generator(
            train_gen,
            epochs=50,
            verbose=0,
            shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
            # callbacks=[es_callback, mc_callback],
        )

        # model.load_weights("model_logs/best_model.h5")
        
        return model
