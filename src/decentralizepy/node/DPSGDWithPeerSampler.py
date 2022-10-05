import logging
import math
import os
from collections import deque

import torch

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDNode import DPSGDNode


class DPSGDWithPeerSampler(DPSGDNode):
    """
    This class defines the node for DPSGD

    """

    def receive_neighbors(self):
        return self.receive_channel("PEERS")[1]["NEIGHBORS"]

    def get_neighbors(self, node=None):
        logging.info("Requesting neighbors from the peer sampler.")
        self.communication.send(
            self.peer_sampler_uid,
            {
                "REQUEST_NEIGHBORS": self.uid,
                "iteration": self.iteration,
                "CHANNEL": "SERVER_REQUEST",
            },
        )
        my_neighbors = self.receive_neighbors()
        logging.info("Neighbors this round: {}".format(my_neighbors))
        return my_neighbors

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
        peer_sampler_uid=-1,
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
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
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

        self.peer_sampler_uid = peer_sampler_uid
        self.connect_neighbor(self.peer_sampler_uid)
        self.wait_for_hello(self.peer_sampler_uid)

        self.run()

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
            self.communication.send(
                self.peer_sampler_uid, {"BYE": self.uid, "CHANNEL": "SERVER_REQUEST"}
            )
            self.sent_disconnections = True

            self.barrier.remove(self.peer_sampler_uid)

            while len(self.barrier):
                sender, _ = self.receive_disconnect()
                self.barrier.remove(sender)
