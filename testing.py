import argparse
import logging

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.Node import Node


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    config = read_ini("config.ini")
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    parser = argparse.ArgumentParser()
    parser.add_argument("-mid", "--machine_id", type=int, default=0)
    parser.add_argument("-ps", "--procs_per_machine", type=int, default=1)
    parser.add_argument("-ms", "--machines", type=int, default=1)

    args = parser.parse_args()

    g = Graph()
    g.read_graph_from_file("36_nodes.edges", "edges")
    n_machines = args.machines
    procs_per_machine = args.procs_per_machine
    l = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    mp.spawn(
        fn=Node,
        nprocs=procs_per_machine,
        args=[m_id, l, g, my_config, 20, "results", logging.DEBUG],
    )
