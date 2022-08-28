import importlib
import logging
import os
from collections import deque

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node


class ParameterServer(Node):
    """
    This class defines the parameter serving service

    """

    def init_log(self, log_dir, log_level, force=True):
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
        log_file = os.path.join(log_dir, "ParameterServer.log")
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
        comm_params = utils.remove_keys(
            comm_configs, ["comm_package", "comm_class"])
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

        self.init_log(log_dir, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
        )

        self.message_queue = dict()

        self.barrier = set()

        self.peer_deques = dict()

        self.init_dataset_model(config["DATASET"])
        self.init_comm(config["COMMUNICATION"])
        self.my_neighbors = self.graph.get_all_nodes()
        self.connect_neighbors()
        self.init_sharing(config["SHARING"])

    def receive_server_request(self):
        return self.receive_channel("SERVER_REQUEST")

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (k not in self.peer_deques) or len(self.peer_deques[k]) == 0:
                return False
        return True

    def run(self):
        """
        Start the parameter-serving service.

        """
        for iteration in range(self.iterations):
            self.iteration = iteration
            # reset deques after each iteration
            self.peer_deques = dict()

            while not self.received_from_all():
                sender, data = self.receive_channel("DPSGD")
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()
                self.peer_deques[sender].append(data)

            logging.info("Received from everybody")

            averaging_deque = dict()
            total = dict()
            for neighbor in self.my_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            for i, n in enumerate(averaging_deque):
                data = averaging_deque[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                data = self.sharing.deserialized_model(data)
                for key, value in data.items():
                    if key in total:
                        total[key] += value
                    else:
                        total[key] = value

            for key, value in total.items():
                total[key] = total[key] / len(averaging_deque)

            to_send = total
            to_send["CHANNEL"] = "GRADS"

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

        while len(self.barrier):
            sender, data = self.receive_server_request()
            if "BYE" in data:
                logging.debug("Received {} from {}".format("BYE", sender))
                self.barrier.remove(sender)

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
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        args : optional
            Other arguments

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            log_level,
            *args
        )

        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            log_level,
            *args
        )

        self.run()

        logging.info("Parameter Server exiting")
