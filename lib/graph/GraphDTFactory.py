from GraphNX import GraphNX

class GDTFactory:
    @staticmethod
    def get_graph_dt(gdt_type):
        if gdt_type == "networkx":
            return GraphNX
