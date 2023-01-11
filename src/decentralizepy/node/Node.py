import importlib
import logging
import math
import os
from collections import deque

import torch

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping


class Node:
    """
    This class defines the node (entity that performs learning, sharing and communication).

    """

    def connect_neighbor(self, neighbor):
        """
        Connects given neighbor. Sends HELLO.

        """
        logging.info("Sending connection request to {}".format(neighbor))
        self.communication.init_connection(neighbor)
        self.communication.send(neighbor, {"HELLO": self.uid, "CHANNEL": "CONNECT"})

    def receive_channel(self, channel, block=True):
        if channel not in self.message_queue:
            self.message_queue[channel] = deque()

        if len(self.message_queue[channel]) > 0:
            return self.message_queue[channel].popleft()
        else:
            x = self.communication.receive(block=block)
            if x == None:
                assert not block
                return None
            sender, recv = x

            logging.info(
                "Received some message from {} with CHANNEL: {}".format(
                    sender, recv["CHANNEL"]
                )
            )
            assert "CHANNEL" in recv
            while recv["CHANNEL"] != channel:
                if recv["CHANNEL"] not in self.message_queue:
                    self.message_queue[recv["CHANNEL"]] = deque()
                self.message_queue[recv["CHANNEL"]].append((sender, recv))
                x = self.communication.receive(block=block)
                if x == None:
                    assert not block
                    return None
                sender, recv = x
                logging.info(
                    "Received some message from {} with CHANNEL: {}".format(
                        sender, recv["CHANNEL"]
                    )
                )
            return (sender, recv)

    def receive_hello(self):
        return self.receive_channel("CONNECT")

    def wait_for_hello(self, neighbor):
        """
        Waits for HELLO.
        Caches any data received while waiting for HELLOs.

        Raises
        ------
        RuntimeError
            If received BYE while waiting for HELLO

        """
        while neighbor not in self.barrier:
            logging.info("Waiting HELLO from {}".format(neighbor))
            sender, _ = self.receive_hello()
            logging.info("Received HELLO from {}".format(sender))
            self.barrier.add(sender)

    def connect_neighbors(self):
        """
        Connects all neighbors. Sends HELLO. Waits for HELLO.
        Caches any data received while waiting for HELLOs.

        Raises
        ------
        RuntimeError
            If received BYE while waiting for HELLO

        """
        logging.info("Sending connection request to all neighbors")
        wait_acknowledgements = []
        for neighbor in self.my_neighbors:
            if not self.communication.already_connected(neighbor):
                self.connect_neighbor(neighbor)
                wait_acknowledgements.append(neighbor)

        for neighbor in wait_acknowledgements:
            self.wait_for_hello(neighbor)

    def receive_disconnect(self):
        return self.receive_channel("DISCONNECT")

    def disconnect_neighbors(self):
        """
        Disconnects all neighbors.

        Raises
        ------
        RuntimeError
            If received another message while waiting for BYEs

        """
        if not self.sent_disconnections:
            logging.info("Disconnecting neighbors")
            for uid in self.barrier:
                self.communication.send(uid, {"BYE": self.uid, "CHANNEL": "DISCONNECT"})
            self.sent_disconnections = True
            while len(self.barrier):
                sender, _ = self.receive_disconnect()
                self.barrier.remove(sender)

    def init_log(self, log_dir, rank, log_level, force=True):
        """
        Instantiate Logging.

        Parameters
        ----------
        log_dir : str
            Logging directory
        rank : rank : int
            Rank of process local to the machine
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        force : bool
            Argument to logging.basicConfig()

        """
        log_file = os.path.join(log_dir, str(rank) + ".log")
        logging.basicConfig(
            filename=log_file,
            format="[%(asctime)s][%(module)s][%(levelname)s] %(message)s",
            level=log_level,
            force=force,
        )

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.iterations = iterations
        self.sent_disconnections = False

        logging.info("Rank: %d", self.rank)
        logging.info("type(graph): %s", str(type(self.rank)))
        logging.info("type(mapping): %s", str(type(self.mapping)))

    def init_dataset_model(self, dataset_configs):
        """
        Instantiate dataset and model from config.

        Parameters
        ----------
        dataset_configs : dict
            Python dict containing dataset config params

        """
        dataset_module = importlib.import_module(dataset_configs["dataset_package"])
        self.dataset_class = getattr(dataset_module, dataset_configs["dataset_class"])
        random_seed = (
            dataset_configs["random_seed"] if "random_seed" in dataset_configs else 97
        )
        torch.manual_seed(random_seed)
        self.dataset_params = utils.remove_keys(
            dataset_configs,
            ["dataset_package", "dataset_class", "model_class", "random_seed"],
        )
        self.dataset = self.dataset_class(
            self.rank, self.machine_id, self.mapping, **self.dataset_params
        )

        logging.info("Dataset instantiation complete.")

        self.model_class = getattr(dataset_module, dataset_configs["model_class"])
        self.model = self.model_class()

    def init_optimizer(self, optimizer_configs):
        """
        Instantiate optimizer from config.

        Parameters
        ----------
        optimizer_configs : dict
            Python dict containing optimizer config params

        """
        optimizer_module = importlib.import_module(
            optimizer_configs["optimizer_package"]
        )
        self.optimizer_class = getattr(
            optimizer_module, optimizer_configs["optimizer_class"]
        )
        self.optimizer_params = utils.remove_keys(
            optimizer_configs, ["optimizer_package", "optimizer_class"]
        )
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_params
        )

    def init_trainer(self, train_configs):
        """
        Instantiate training module and loss from config.

        Parameters
        ----------
        train_configs : dict
            Python dict containing training config params

        """
        train_module = importlib.import_module(train_configs["training_package"])
        train_class = getattr(train_module, train_configs["training_class"])

        loss_package = importlib.import_module(train_configs["loss_package"])
        if "loss_class" in train_configs.keys():
            loss_class = getattr(loss_package, train_configs["loss_class"])
            self.loss = loss_class()
        else:
            self.loss = getattr(loss_package, train_configs["loss"])

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
        self.trainer = train_class(
            self.rank,
            self.machine_id,
            self.mapping,
            self.model,
            self.optimizer,
            self.loss,
            self.log_dir,
            **train_params
        )

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def init_sharing(self, sharing_configs):
        """
        Instantiate sharing module from config.

        Parameters
        ----------
        sharing_configs : dict
            Python dict containing sharing config params

        """
        sharing_package = importlib.import_module(sharing_configs["sharing_package"])
        sharing_class = getattr(sharing_package, sharing_configs["sharing_class"])
        sharing_params = utils.remove_keys(
            sharing_configs, ["sharing_package", "sharing_class"]
        )
        self.sharing = sharing_class(
            self.rank,
            self.machine_id,
            self.communication,
            self.mapping,
            self.graph,
            self.model,
            self.dataset,
            self.log_dir,
            **sharing_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        log_level=logging.INFO,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])

    def run(self):
        """
        Start the decentralized learning

        """
        raise NotImplementedError

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
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        # self.instantiate(
        #     rank,
        #     machine_id,
        #     mapping,
        #     graph,
        #     config,
        #     iterations,
        #     log_dir,
        #     log_level,
        #     *args
        # )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
