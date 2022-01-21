from decentralizepy.graphs.Graph import Graph


class FullyConnected(Graph):
    """
    The class for generating a Fully Connected Graph Topology

    """

    def __init__(self, n_procs):
        """
        Constructor. Generates a Fully Connected graph

        Parameters
        ----------
        n_procs : int
            total number of nodes in the graph

        """
        super().__init__(n_procs)
        for node in range(n_procs):
            neighbors = set([x for x in range(n_procs) if x != node])
            self.adj_list[node] = neighbors
