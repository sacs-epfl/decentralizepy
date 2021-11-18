from decentralizepy.node.Node import Node
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from torch import multiprocessing as mp
import logging

from localconfig import LocalConfig

def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items('DATASET')))
    return config


if __name__ == "__main__":   
    config = read_ini("config.ini")
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    g = Graph()
    g.read_graph_from_file("graph.adj", "adjacency")
    l = Linear(1, 6)

    mp.spawn(fn = Node, nprocs = 6, args=[0,l,g,my_config,20,"results",logging.DEBUG])
