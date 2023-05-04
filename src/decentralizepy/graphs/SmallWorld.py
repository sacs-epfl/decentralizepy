import smallworld

from decentralizepy.graphs.Graph import Graph


class SmallWorld(Graph):
    """
    The class for generating a SmallWorld topology Graph

    Adapted from https://gitlab.epfl.ch/sacs/ml-rawdatasharing/dnn-recommender/-/blob/master/topologies.py

    """

    def __init__(self, n_procs, k_over_2, beta):
        """
        Constructor. Generates a random connected SmallWorld graph

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph
        k_over_2 : int
            k_over_2 config for smallworld
        beta : float
            beta config for smallworld. β = 1 is truly equal to the Erdős-Rényi network model

        """
        super().__init__(n_procs)
        G = smallworld.get_smallworld_graph(self.n_procs, k_over_2, beta)
        for edge in list(G.edges):
            node1 = edge[0]
            node2 = edge[1]
            self.adj_list[node1].add(node2)
            self.adj_list[node2].add(node1)

        self.connect_graph()
