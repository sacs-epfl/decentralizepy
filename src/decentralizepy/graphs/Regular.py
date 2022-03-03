import networkx as nx

from decentralizepy.graphs.Graph import Graph


class Regular(Graph):
    """
    The class for generating a Regular topology

    """

    def __init__(self, n_procs, degree, seed=None):
        """
        Constructor. Generates a Ring graph

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph
        degree : int
            Neighbors of each node

        """
        super().__init__(n_procs)
        G = nx.random_regular_graph(degree, n_procs, seed)
        adj = G.adjacency()
        for i, l in adj:
            self.adj_list[i] = set()  # new set
            for k in l:
                self.adj_list[i].add(k)
        if not nx.is_connected(G):
            self.connect_graph()
