import networkx as nx

from decentralizepy.graphs.Graph import Graph


class Star(Graph):
    """
    The class for generating a Star topology
    Adapted from ./Regular.py

    """

    def __init__(self, n_procs):
        """
        Constructor. Generates a Ring graph

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph

        """
        super().__init__(n_procs)
        G = nx.star_graph(n_procs - 1)
        adj = G.adjacency()
        for i, l in adj:
            self.adj_list[i] = set()  # new set
            for k in l:
                self.adj_list[i].add(k)
        if not nx.is_connected(G):
            self.connect_graph()
