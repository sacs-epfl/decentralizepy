import importlib
import json
import logging
import math
import os
from collections import deque

import torch

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node


class STCClient(Node):
    """
    This class defines the node for federated learning with gradient sharing

    """

    def run(self):
        """
        Start the decentralized learning

        """
        sender, data = None, None
        iteration = 0
        while len(self.barrier):
            self.sharing.process_received(data)

            self.sharing.communication_round += 1

            logging.info(
                "Received worker request at node {}, global iteration {}, local round {}".format(
                    self.uid, iteration, self.participated
                )
            )

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            # Perform iteration
            logging.info("Starting training iteration")
            self.trainer.train(self.dataset)

            logging.info("Finished training iteration")

            # Send update to server
            to_send = self.sharing.get_data_to_send()
            to_send["CHANNEL"] = "WORKER_GRADIENTS"
            self.communication.send(self.server_uid, to_send)

            if self.participated:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

            self.participated += 1

            sender, data = self.receive_channel("WORKER_REQUEST")

            if "BYE" in data:
                logging.debug("Received {} from {}".format("BYE", sender))
                self.barrier.remove(sender)
                break

            iteration = data["iteration"]
            del data["iteration"]
            del data["CHANNEL"]

        logging.info("Server disconnected. Process complete!")

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
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
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

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

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
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
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
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
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()

        self.participated = 0

        self.init_sharing(config["SHARING"])

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        server_uid=-1,
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
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        server_uid: int
            The server's uid
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.get_local_procs_count()), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )

        self.message_queue["PEERS"] = deque()

        self.server_uid = server_uid
        self.my_neighbors = {self.server_uid}
        self.connect_neighbors()

        self.run()
