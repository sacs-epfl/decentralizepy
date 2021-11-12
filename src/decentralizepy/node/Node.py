import logging
import os

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy import utils

from torch import optim
import importlib


class Node:
    """
    This class defines the node (entity that performs learning, sharing and communication).
    """
    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations = 1,
        log_dir=".",
        log_level=logging.INFO,
        *args
    ):
        """
        Constructor
        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
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
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments
        """
        log_file = os.path.join(log_dir, str(rank) + ".log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
            level=log_level,
            force=True,
        )

        logging.info("Started process.")

        self.rank = rank
        self.graph = graph
        self.mapping = mapping

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))
        
        dataset_configs = config["DATASET"]
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        dataset_params = utils.remove_keys(dataset_configs, ["dataset_package", "dataset_class", "model_class"])
        self.dataset =  dataset_class(rank, **dataset_params)

        logging.info("Dataset instantiation complete.")

        model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = model_class()

        optimizer_configs = config["OPTIMIZER_PARAMS"]
        optimizer_module = importlib.import_module(optimizer_configs["optimizer_package"])
        optimizer_class = getattr(optimizer_module, optimizer_configs["optimizer_class"])
        optimizer_params = utils.remove_keys(optimizer_configs, ["optimizer_package", "optimizer_class"])
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        train_configs = config["TRAIN_PARAMS"]
        train_module = importlib.import_module(train_configs["training_package"])
        train_class = getattr(train_module, train_configs["training_class"])

        loss_package = importlib.import_module(train_configs["loss_package"])
        loss = getattr(loss_package, train_configs["loss"])

        train_params = utils.remove_keys(train_configs, ["training_package", "training_class", "loss", "loss_package"])
        self.trainer = train_class(self.model, self.optimizer, loss, **train_params)

        for iteration in range(iterations):
            logging.info("Starting training iteration: %d", iteration)
            self.trainer.train(self.dataset)