from GraphADT import GraphADT
import networkx as nx
import itertools
import logging
import os

PROPERTY_NAME 		= "name"
PROPERTY_LABEL 		= "label"
PROPERTY_FEATURE 	= "feature"

def loadCSV(nodes="nodes.csv", edges="edges.csv", features="features.csv", g = None):
	NodeList = {}
	EdgeList = []
	FeatureList = {}


	with open(nodes) as f:
		for line in f:
			ll = line.strip().split(' ')
			NodeList[ll[0]] = ll[1]

	logging.info("Node file: %s, #nodes: %d" % (nodes, len(NodeList)))

	with open(edges) as f:
		for line in f:
			ll = line.strip().split('\t')
			EdgeList.append((ll[0], ll[1]))

	logging.info("Edge file: %s, #edges: %d" % (edges, len(EdgeList)))

	with open(features) as f:
		for line in f:
			ll = line.strip().split(' ')
			FeatureList[ll[0]] = [bool(i == '1') for i in ll[1:]]

	logging.info("Feature file: %s, features dimension: %d" % (edges, len(FeatureList[FeatureList.keys()[0]])))



	ClassLabels = {}
	NodeClasses = {}

	classIndex = -1
	for k,v in NodeList.iteritems():
		if ClassLabels.has_key(v) == False:
			classIndex = classIndex + 1
			ClassLabels[v] = classIndex
		
		NodeClasses[k] = ClassLabels[v]

	logging.info("Number of class labels: %d" % len(ClassLabels))
	logging.info(sorted(enumerate(ClassLabels), key=lambda x:x[0]))

	N = len(NodeList)

	NodeIndex = {}
	FeatureIndex = {}

	index = 0
	for k,v in NodeList.iteritems():
		g.add_node(index, {'name' : k, 'label' : v})
		NodeIndex[k] = index
		index = index + 1

	for k,v in FeatureList.iteritems():
		index = NodeIndex[k]
		g.set_node_property(index, 'feature', v)
		FeatureIndex[index] = v

	for edge in EdgeList:
		if NodeIndex.has_key(edge[0]) == False or NodeIndex.has_key(edge[1]) == False:
			continue
		
		g.add_edge(NodeIndex[edge[0]], NodeIndex[edge[1]])

	return NodeIndex, ClassLabels, FeatureIndex