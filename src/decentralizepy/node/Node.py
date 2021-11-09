class Node:
    """
    This class defines the node (entity that performs learning, sharing and communication).
    """

    def __init__(self, rank, mapping, graph, options):
        """
        Constructor
        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        dataset : decentralizepy.datasets class
            The class whose object will be instantiated to init the dataset
        """
        self.rank = rank
        self.graph = graph
        self.mapping = mapping
        self.options = options

    def __get_item__(self, i):
        """
        Function to get the item with index i.
        Parameters
        ----------
        i : int
            Index
        Returns
        -------
        2-tuple
            A tuple of the ith data sample and it's corresponding label
        """
        return self.x[i], self.y[i]
