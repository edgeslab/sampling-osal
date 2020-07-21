from GraphADT import GraphADT
import networkx as nx

class GraphNX(GraphADT):

    def __init__(self, origin = None):
        GraphADT.__init__(self, origin)

    def create(self):
        return nx.Graph()


    def add_node(self, index, properties = None):
        if properties is not None:
            self.graph.add_node(index, **properties)
        else:
            self.graph.add_node(index)


    def add_edge(self, src_index, dest_index, properties = None):
        if properties is not None:
            self.graph.add_edge(src_index, dest_index, **properties)
        else:
            self.graph.add_edge(src_index, dest_index)


    def num_nodes(self):
        return self.graph.number_of_nodes()


    def num_edges(self):
        return self.graph.number_of_edges()


    def neighbors(self, node_index):
        return self.graph.neighbors(node_index)


    def largest_component(self):
        return GraphNX(max(nx.connected_component_subgraphs(self.graph), key=len))


    def set_node_property(self, index, prop_name, prop_value):
        self.graph.node[index][prop_name] = prop_value

# Test
# from GraphNX import GraphNX
# g = GraphNX()
# g.add_node(1)
# g.add_node(2)
# g.add_node(3)
# g.add_edge(1,2)
# g.num_nodes()
# g.neighbors(1)
# l = g.largest_component()
# l.num_nodes()
# g.set_node_property(1, 'name', 'hati')