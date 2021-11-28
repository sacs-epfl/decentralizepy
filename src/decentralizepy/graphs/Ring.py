from decentralizepy.graphs.Graph import Graph


class Ring(Graph):
    """
    The class for generating a Ring topology
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
        self.connect_graph()
