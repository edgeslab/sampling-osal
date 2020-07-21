from abc import ABCMeta, abstractmethod

class GraphADT:
    __metaclass__ = ABCMeta

    def __init__(self, origin = None):
        if origin is not None:
            self.graph = origin
        else:
            self.graph = self.create()

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def add_node(self, index, properties = None):
        pass

    @abstractmethod
    def add_edge(self, src_index, dest_index, properties = None):
        pass

    @abstractmethod
    def num_nodes(self):
        pass

    @abstractmethod
    def num_edges(self):
        pass

    @abstractmethod
    def neighbors(self, node_index):
        pass

    @abstractmethod
    def largest_component(self):
        pass

    @abstractmethod
    def set_node_property(self, index, prop_name, prop_value):
        pass