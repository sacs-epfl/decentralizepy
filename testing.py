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
    g.read_graph_from_file("36_nodes.edges", "edges")
    n_machines = 3
    procs_per_machine = 12
    l = Linear(n_machines, procs_per_machine)

    mp.spawn(fn = Node, nprocs = procs_per_machine, args=[0,l,g,my_config,20,"results",logging.DEBUG])
