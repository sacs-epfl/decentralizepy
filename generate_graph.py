from decentralizepy.graphs.Regular import Regular
from decentralizepy.graphs.FullyConnected import FullyConnected
from decentralizepy.graphs.Ring import Ring
from decentralizepy.graphs.SmallWorld import SmallWorld
from decentralizepy.graphs.Star import Star
import getopt, sys

if __name__ == "__main__":
    """
    Script to generate a graph file.

    Usage
    -----
    python generate_graph.py -g <graph_type> -n <num_nodes> -s <seed> -d <degree> -k <k_over_2> -b <beta> -f <file_name> -a

    Parameters
    ----------
    graph_type : str
        One of {"Regular", "FullyConnected", "Ring", "SmallWorld", "Star"}
    num_nodes : int
        Number of nodes in the graph
    seed : int, optional
        Seed for random number generator
    degree : int, optional
        Degree of the graph
    k_over_2 : int, optional
        Parameter for smallworld
    beta : float, optional
        Parameter for smallworld
    file_name : str, optional
        Name of the file to write the graph to
    a : flag, optional
        If set, the graph is written in adjacency list format, otherwise in edge list format
    h : flag, optional
        Prints this help message

    """
    __doc__ = "Usage: python3 generate_graph.py -g <graph_type> -n <num_nodes> -s <seed> -d <degree> -k <k_over_2> -b <beta> -f <file_name> -a -h"
    assert len(sys.argv) >= 2, __doc__
    argumentList = sys.argv[1:]

    options = "hg:n:s:d:k:b:f:a"

    long_options = ["graph=", "nodes=", "seed=", "degree=", "kover2=", "beta=", "file=", "adjacency", "help"]

    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)

        graph_type = None
        num_nodes = None
        seed = None
        degree = None
        k_over_2 = None
        beta = None
        file_name = None
        type_adjacency = "edges"

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--help"):
                print(__doc__)
                exit(0)
            elif currentArgument in ("-g", "--graph"):
                graph_type = currentValue
            elif currentArgument in ("-n", "--nodes"):
                num_nodes = int(currentValue)
            elif currentArgument in ("-s", "--seed"):
                seed = int(currentValue)
            elif currentArgument in ("-d", "--degree"):
                degree = int(currentValue)
            elif currentArgument in ("-k", "--kover2"):
                k_over_2 = int(currentValue)
            elif currentArgument in ("-b", "--beta"):
                beta = float(currentValue)
            elif currentArgument in ("-f", "--file"):
                file_name = currentValue
            elif currentArgument in ("-a", "--adjacency"):
                type_adjacency = "adjacency"
    
        if graph_type == 'Regular':
            g = Regular(num_nodes, degree, seed)
        elif graph_type == 'FullyConnected':
            g = FullyConnected(num_nodes)
        elif graph_type == 'Ring':
            g = Ring(num_nodes)
        elif graph_type == 'SmallWorld':
            g = SmallWorld(num_nodes, k_over_2, beta)
        elif graph_type == 'Star':
            g = Star(num_nodes)
        else:
            raise ValueError("Invalid graph type: " + graph_type)
        

        if file_name is not None:
            g.write_graph_to_file(file_name, type=type_adjacency)
        else:
            raise ValueError("No file name. " + __doc__)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)