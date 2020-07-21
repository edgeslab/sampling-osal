import logging
import lib.graph.nx_csv as nxcsv

from lib.graph.GraphUtils import *

class GraphData:

    def __init__(self, graph_csv_loader=nxcsv, node_file=None, edge_file=None, feature_file=None, synth_params={}):
        self.graph_csv_loader = graph_csv_loader
        
        graph_data = None
        if synth_params:
            graph_data = gen_graph(**synth_params)
        else:
            graph_data = graph_csv_loader.loadCSV(node_file, edge_file, feature_file)
        
        self.graph, self.class_labels, self.feature_matrix = graph_data
        
        self.num_feat = 0
        for k in self.feature_matrix:
            self.num_feat = len(self.feature_matrix[k])
            break
        logging.info('Feature dimension: %d', self.num_feat)

        self.graph = sample(self.graph)
        logging.info("Filtering complete: %d vertices and %d edges" % (self.graph.number_of_nodes(), self.graph.number_of_edges()))