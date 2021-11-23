import importlib
import logging
import os

from decentralizepy import utils
from decentralizepy.communication.Communication import Communication
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping


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
        iterations=1,
        log_dir=".",
        log_level=logging.INFO,
        test_after=5,
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
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

        dataset_configs = config["DATASET"]
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        dataset_params = utils.remove_keys(
            dataset_configs, ["dataset_package", "dataset_class", "model_class"]
        )
        self.dataset = dataset_class(rank, **dataset_params)

        logging.info("Dataset instantiation complete.")

        model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = model_class()

        optimizer_configs = config["OPTIMIZER_PARAMS"]
        optimizer_module = importlib.import_module(
            optimizer_configs["optimizer_package"]
        )
        optimizer_class = getattr(
            optimizer_module, optimizer_configs["optimizer_class"]
        )
        optimizer_params = utils.remove_keys(
            optimizer_configs, ["optimizer_package", "optimizer_class"]
        )
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

        train_configs = config["TRAIN_PARAMS"]
        train_module = importlib.import_module(train_configs["training_package"])
        train_class = getattr(train_module, train_configs["training_class"])

        loss_package = importlib.import_module(train_configs["loss_package"])
        if "loss_class" in train_configs.keys():
            loss_class = getattr(loss_package, train_configs["loss_class"])
            loss = loss_class()
        else:
            loss = getattr(loss_package, train_configs["loss"])

        train_params = utils.remove_keys(
            train_configs,
            [
                "training_package",
                "training_class",
                "loss",
                "loss_package",
                "loss_class",
            ],
        )
        self.trainer = train_class(self.model, self.optimizer, loss, **train_params)

        comm_configs = config["COMMUNICATION"]
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )
        self.communication.connect_neighbors(self.graph.neighbors(self.uid))

        sharing_configs = config["SHARING"]
        sharing_package = importlib.import_module(sharing_configs["sharing_package"])
        sharing_class = getattr(sharing_package, sharing_configs["sharing_class"])
        self.sharing = sharing_class(
            self.rank,
            self.machine_id,
            self.communication,
            self.mapping,
            self.graph,
            self.model,
            self.dataset,
        )

        self.testset = self.dataset.get_testset()
        rounds_to_test = test_after

        for iteration in range(iterations):
            logging.info("Starting training iteration: %d", iteration)
            self.trainer.train(self.dataset)

            self.sharing.step()
            self.optimizer = optimizer_class(
                self.model.parameters(), **optimizer_params
            )  # Reset optimizer state
            self.trainer.reset_optimizer(self.optimizer)

            rounds_to_test -= 1

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = test_after
                self.dataset.test(self.model)

        self.communication.disconnect_neighbors()
