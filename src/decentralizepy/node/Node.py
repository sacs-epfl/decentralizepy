from torch import utils, optim
import importlib


def __remove_keys__(d, keys_to_remove):
    return {key: d[key] for key in d if key not in keys_to_remove}

class Node:
    """
    This class defines the node (entity that performs learning, sharing and communication).
    """

    def __init__(self, rank, mapping, graph, config, *args):
        """
        Constructor
        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        n_procs_local : int
            Number of processes on current machine
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
        args : optional
            Other arguments
        """
        self.rank = rank
        self.graph = graph
        self.mapping = mapping
        
        dataset_configs = dict(config.items("DATASET"))
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        
        dataset_params = __remove_keys__(dataset_configs, ["dataset_package", "dataset_class", "model_class"])
        self.dataset =  dataset_class(rank, **dataset_params)
        self.trainset = self.dataset.get_trainset()

        model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = model_class()

        optimizer_configs = dict(config.items("OPTIMIZER_PARAMS"))
        optimizer_module = importlib.import_module(optimizer_configs["optimizer_package"])
        optimizer_class = getattr(optimizer_module, optimizer_configs["optimizer_class"])
        
        optimizer_params = __remove_keys__(optimizer_configs, ["optimizer_package", "optimizer_class"])
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        self.run()

    def train_step(self):
        """
        The training step
        """
        for epoch in self.epochs_per_round: # Instantiate this variable
            for data, target in self.trainset: 
                # Perform training step
                raise NotImplementedError




    def run(self):
        """
        The learning loop.
        """
        while True:
            # Receive data

            # Learn


            # Send data
            raise NotImplementedError

