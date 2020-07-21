import os
import pdb
import queue as que
from copy import deepcopy

import numpy as np
import pandas as pd
import networkx as nx

from lib.graph.GraphUtils import modularity_cluster


class NetworkSampling:

    @staticmethod
    def degree_centrality(graph, size, pool, descending=True):
        dc = nx.degree_centrality(graph)
        dc = sorted(dc.items(), key=lambda x: x[1], reverse=descending)
        
        i = 0
        selection = []
        while size > 0:
            v = dc[i][0]
            
            if v not in pool:
                i += 1
                continue
            
            selection.append(v)
            size -= 1
            i += 1

        return selection


    @staticmethod
    def clustering_coefficient(graph, size, pool, descending=True):
        ct = nx.clustering(graph, nodes=pool)
        ct = sorted(ct.items(), key=lambda x: x[1], reverse=True)
        
        i = 0
        selection = []
        while size > 0:
            v = ct[i][0]
            
            if v not in pool:
                i += 1
                continue
            
            selection.append(v)
            size -= 1
            i += 1

        return selection


    @staticmethod
    def edge_sampling(graph, size, pool, descending=True):
        P = deepcopy(pool)
        edges = list(graph.edges())
        
        selection = []
        while size > 0:
            i = np.random.choice(range(len(edges)), 1)[0]
            u,v = edges[i]
            
            if u in P:
                selection.append(u)
                P.remove(u)
                size -= 1

            if v in P:
                selection.append(v)
                P.remove(v)
                size -= 1

        return selection


    @staticmethod
    def snowball_sampling(graph, size, pool):
        """this function returns a set of nodes of size 'size' from 'graph' that are 
        collected from around seed node via snownball sampling"""

        if graph.number_of_nodes() < size:
            return set()

        seed = np.random.choice(pool, 1)[0]
        selection = set([seed])
        
        q = que.Queue()
        q.put(seed)
        mark = {seed : True}
        while not q.empty():
            v = q.get()
            for node in graph.neighbors(v):
                if node in mark:
                    continue
                if len(selection) < size:
                    q.put(node)
                    if node in pool:
                        selection.add(node)
                    mark[node] = True
                else :
                    return selection

        return selection


    @staticmethod
    def forest_fire_sampling(graph, size, pool, p=0.7):
        """this function returns a set of nodes from pool equal to size from graph that are 
        collected from around seed node via forest fire sampling with visit probability = p"""

        if graph.number_of_nodes() < size:
            return set()

        selection = set([])
        while len(selection) < size:
            seed = np.random.choice(pool, 1)[0]
            selection = set([seed])
            
            q = que.Queue()
            q.put(seed)
            mark = {seed : True}
            while not q.empty():
                v = q.get()
                for node in graph.neighbors(v):
                    if np.random.rand() > p:
                        continue

                    if node in mark:
                        continue
                    if len(selection) < size:
                        q.put(node)
                        if node in pool:
                            selection.add(node)
                        mark[node] = True
                    else :
                        return selection

        return selection
        

    @staticmethod
    def modularity_clustering(graph, size, pool, k=8):
        clusters = modularity_cluster(graph, 0.8 * (graph.number_of_nodes() / k))
        cmap = {i: clusters[i].nodes() for i in range(len(clusters))}

        for i in range(len(clusters)):
            cmap[i] = np.array(list(set(cmap[i]).intersection(pool)))

        selection = []
        while size > 0:
            for k_ in range(k):
                if len(cmap[k_]) == 0:
                    continue

                v = np.random.choice(cmap[k_], 1, replace = False)[0]
                cmap[k_] = np.delete(cmap[k_], np.argwhere(cmap[k_]==v))
                selection.append(v)
                size -= 1

        return selection