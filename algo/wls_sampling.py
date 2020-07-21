import pdb
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

from lib.graph.palette_wl import *
from lib.graph.GraphUtils import *


class WLS_Sampling:
    
    @staticmethod
    def wls(graph, size, pool, k, nhops):
        P = deepcopy(pool)

        selection = []
        while len(selection) < size:
            pg = graph.subgraph(P)
            assert(len(pg.node.keys()) == len(P))

            for k_ in range(k): 
                v = np.random.choice(P, 1, replace = False)[0]
                neighbors, A = get_neighborhood_labels(pg, v, nhops)

                sel_node = None
                if not neighbors:
                    sel_node = v
                elif np.sum(A) == 0:
                    sel_node = np.random.choice(neighbors, 1, replace = False)[0]
                else:
                    wl_labels = palette_wl(A)
                    l = np.argsort(wl_labels)
                    sel_node = neighbors[l[0]]
                
                selection.append(sel_node)
                P.remove(sel_node)
                pg = graph.subgraph(P)

        return selection

