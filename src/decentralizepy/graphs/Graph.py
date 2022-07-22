import networkx as nx
import numpy as np


class Graph:
    """
    This class defines the graph topology.
    Adapted from https://gitlab.epfl.ch/sacs/ml-rawdatasharing/dnn-recommender/-/blob/master/api.py
    """

    def __init__(self, n_procs=None):
        """
        Constructor

        Parameters
        ----------
        n_procs : int, optional
            Number of processes in the graph, if already known

        """
        if n_procs != None:
            self.n_procs = n_procs
            self.adj_list = [set() for i in range(self.n_procs)]

    def get_all_nodes(self):
        return [i for i in range(self.n_procs)]

    def __insert_adj__(self, node, neighbours):
        """
        Inserts `neighbours` into the adjacency list of `node`

        Parameters
        ----------
        node : int
            The vertex in question
        neighbours : list(int)
            A list of neighbours of the `node`

        """
        self.adj_list[node].update(neighbours)

    def __insert_edge__(self, x, y):
        """
        Inserts edge `x -> y` into the graph

        Parameters
        ----------
        x : int
            The source vertex
        y : int
            The destination vertex

        """
        self.adj_list[x].add(y)
        self.adj_list[y].add(x)

    def read_graph_from_file(self, file, type="edges", force_connect=False):
        """
        Reads the graph from a given file

        Parameters
        ----------
        file : str
            path to the file
        type : str
            `edges` or `adjacency`
        force_connect : bool, optional
            Should the graph be force-connected using a ring

        Returns
        -------
        int
            Number of processes, read from the first line of the file

        Raises
        ------
        ValueError
            If the type is not either `edges` or `adjacency`

        """

        with open(file, "r") as inf:
            self.n_procs = int(inf.readline().strip())
            self.adj_list = [set() for i in range(self.n_procs)]

            lines = inf.readlines()
            if type == "edges":
                for line in lines:
                    x, y = map(int, line.strip().split())
                    self.__insert_edge__(x, y)
            elif type == "adjacency":
                node_id = 0
                for line in lines:
                    neighbours = map(int, line.strip().split())
                    self.__insert_adj__(node_id, neighbours)
                    node_id += 1
            else:
                raise ValueError("type must be from {edges, adjacency}!")

        if force_connect:
            self.connect_graph()

        return self.n_procs

    def write_graph_to_file(self, file, type="edges"):
        """
        Writes graph to file

        Parameters
        ----------
        file : str
            File path
        type : str
            One of {"edges", "adjacency"}. Writes the corresponding format.

        """
        with open(file, "w") as of:
            of.write(str(self.n_procs) + "\n")
            if type == "edges":
                for node, adj in enumerate(self.adj_list):
                    for neighbor in adj:
                        of.write("{} {}".format(node, neighbor) + "\n")
            elif type == "adjacency":
                for adj in self.adj_list:
                    of.write(str(*adj) + "\n")
            else:
                raise ValueError("type must be from {edges, adjacency}!")

    def connect_graph(self):
        """
        Connects the graph using a Ring

        """
        for node in range(self.n_procs):
            self.adj_list[node].add((node + 1) % self.n_procs)
            self.adj_list[node].add((node - 1) % self.n_procs)

    def neighbors(self, uid):
        """
        Gives the neighbors of a node

        Parameters
        ----------
        uid : int
            globally unique identifier of the node

        Returns
        -------
        set(int)
            a set of neighbours

        """
        return self.adj_list[uid]

    def centr(self):
        my_adj = {x: list(adj) for x, adj in enumerate(self.adj_list)}
        nxGraph = nx.Graph(my_adj)
        a = nx.to_numpy_matrix(nxGraph)
        self.averaging_weights = np.ones((self.n_procs, self.n_procs), dtype=float)
        centrality = nx.betweenness_centrality(nxGraph)
        for i in range(len(centrality)):
            centrality[i] += 0.01
        for i in range(self.averaging_weights.shape[0]):
            s = 0
            for j in range(self.averaging_weights.shape[0]):
                self.averaging_weights[i, j] = 1.0 / centrality[j]
                s += self.averaging_weights[i, j]
            for j in range(self.averaging_weights.shape[0]):
                self.averaging_weights[i, j] = self.averaging_weights[i, j] / s
        return self.averaging_weights
