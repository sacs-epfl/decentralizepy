import logging
import math
import os
import queue
from random import Random
from threading import Lock, Thread

import numpy as np
import torch
from numpy.linalg import norm

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.OverlayNode import OverlayNode


class KNN(OverlayNode):
    """
    This class defines the node for KNN Learning Node

    """

    def similarityMetric(self, candidate):
        """
        Lower is better

        """
        logging.debug("A: {}".format(self.othersInfo[self.uid]))
        logging.debug("B: {}".format(self.othersInfo[candidate]))
        A = np.array(self.othersInfo[self.uid])
        B = np.array(self.othersInfo[candidate])
        return np.abs(np.dot(A, B) / (norm(A) * norm(B)))

    def get_most_similar(self, candidates, to_keep=4):
        if len(candidates) <= to_keep:
            return candidates

        cur_candidates = dict()
        for i in candidates:
            simil = round(self.similarityMetric(i), 3)
            if simil not in cur_candidates:
                cur_candidates[simil] = []
            cur_candidates[simil].append(i)

        similarity_scores = list(cur_candidates.keys())
        similarity_scores.sort()

        left_to_keep = to_keep
        return_result = set()
        for i in similarity_scores:
            if left_to_keep >= len(cur_candidates[i]):
                return_result.update(cur_candidates[i])
                left_to_keep -= len(cur_candidates[i])
            elif left_to_keep > 0:
                return_result.update(
                    list(self.rng.sample(cur_candidates[i], left_to_keep))
                )
                left_to_keep = 0
                break
            else:
                break

        return return_result

    def create_message_to_send(
        self,
        channel="KNNConstr",
        boolean_flags=[],
        add_my_info=False,
        add_neighbor_info=False,
    ):
        message = {"CHANNEL": channel, "KNNRound": self.knn_round}
        for x in boolean_flags:
            message[x] = True
        if add_my_info:
            message[self.uid] = self.othersInfo[self.uid]
        if add_neighbor_info:
            for neighbors in self.out_edges:
                if neighbors in self.othersInfo:
                    message[neighbors] = self.othersInfo[neighbors]
        return message

    def receive_KNN_message(self):
        return self.receive_channel("KNNConstr", block=False)

    def process_init_receive(self, message):
        self.mutex.acquire()
        if "RESPONSE" in message[1]:
            self.num_initializations += 1
        else:
            self.communication.send(
                message[0],
                self.create_message_to_send(
                    boolean_flags=["INIT", "RESPONSE"], add_my_info=True
                ),
            )
        x = (
            message[0],
            utils.remove_keys(message[1], ["RESPONSE", "INIT", "CHANNEL", "KNNRound"]),
        )
        self.othersInfo.update(x[1])
        self.mutex.release()

    def remove_meta_from_message(self, message):
        return (
            message[0],
            utils.remove_keys(message[1], ["RESPONSE", "INIT", "CHANNEL", "KNNRound"]),
        )

    def process_candidates_without_lock(self, current_candidates, message):
        if not self.exit_receiver:
            message = (
                message[0],
                utils.remove_keys(
                    message[1], ["CHANNEL", "RESPONSE", "INIT", "KNNRound"]
                ),
            )
            self.othersInfo.update(message[1])
            new_candidates = set(message[1].keys())
            current_candidates = current_candidates.union(new_candidates)
            if self.uid in current_candidates:
                current_candidates.remove(self.uid)
            self.out_edges = self.get_most_similar(current_candidates, to_keep=self.K)

    def send_response(self, message, add_neighbor_info=False, process_candidates=False):
        self.mutex.acquire()
        logging.debug("Responding to {}".format(message[0]))
        self.communication.send(
            message[0],
            self.create_message_to_send(
                boolean_flags=["RESPONSE"],
                add_my_info=True,
                add_neighbor_info=add_neighbor_info,
            ),
        )
        if process_candidates:
            self.process_candidates_without_lock(set(self.out_edges), message)
        self.mutex.release()

    def receiver_thread(self):
        knnBYEs = set()
        self.num_initializations = 0
        waiting_queue = queue.Queue()
        while True:
            if len(knnBYEs) == self.mapping.get_n_procs() - 1:
                self.mutex.acquire()
                if self.exit_receiver:
                    self.mutex.release()
                    logging.debug("Exiting thread")
                    return
                self.mutex.release()

            if self.num_initializations < self.initial_neighbors:
                x = self.receive_KNN_message()
                if x == None:
                    continue
                elif "INIT" in x[1]:
                    self.process_init_receive(x)
                else:
                    waiting_queue.put(x)
            else:
                logging.debug("Waiting for messages")
                if waiting_queue.empty():
                    x = self.receive_KNN_message()
                    if x == None:
                        continue
                else:
                    x = waiting_queue.get()

                if "INIT" in x[1]:
                    logging.debug("A past INIT Message received from {}".format(x[0]))
                    self.process_init_receive(x)
                elif "RESPONSE" in x[1]:
                    logging.debug(
                        "A response message received from {} from KNNRound {}".format(
                            x[0], x[1]["KNNRound"]
                        )
                    )
                    x = self.remove_meta_from_message(x)
                    self.responseQueue.put(x)
                elif "RANDOM_DISCOVERY" in x[1]:
                    logging.debug(
                        "A Random Discovery message received from {} from KNNRound {}".format(
                            x[0], x[1]["KNNRound"]
                        )
                    )
                    self.send_response(
                        x, add_neighbor_info=False, process_candidates=False
                    )
                elif "KNNBYE" in x[1]:
                    self.mutex.acquire()
                    knnBYEs.add(x[0])
                    logging.debug("{} KNN Byes received".format(knnBYEs))
                    if self.uid in x[1]["CLOSE"]:
                        self.in_edges.add(x[0])
                    self.mutex.release()
                else:
                    logging.debug(
                        "A KNN sharing message received from {} from KNNRound {}".format(
                            x[0], x[1]["KNNRound"]
                        )
                    )
                    self.send_response(
                        x, add_neighbor_info=True, process_candidates=True
                    )

    def build_topology(self, rounds=60, random_nodes=4):
        self.knn_round = 0
        self.exit_receiver = False
        t = Thread(target=self.receiver_thread)

        t.start()

        # Initializations : Send my dataset info to others

        self.mutex.acquire()
        initial_KNN_message = self.create_message_to_send(
            boolean_flags=["INIT"], add_my_info=True
        )
        for x in self.out_edges:
            self.communication.send(x, initial_KNN_message)
        self.mutex.release()

        for round in range(rounds):
            self.knn_round = round
            logging.debug("Starting KNN Round {}".format(round))
            self.mutex.acquire()
            rand_neighbor = self.rng.choice(list(self.out_edges))
            logging.debug("Random neighbor: {}".format(rand_neighbor))
            self.communication.send(
                rand_neighbor,
                self.create_message_to_send(add_my_info=True, add_neighbor_info=True),
            )
            self.mutex.release()

            logging.debug("Waiting for knn response from {}".format(rand_neighbor))

            response = self.responseQueue.get(block=True)

            logging.debug("Got response from random neighbor")

            self.mutex.acquire()
            random_candidates = set(
                self.rng.sample(list(range(self.mapping.get_n_procs())), random_nodes)
            )

            req_responses = 0
            for rc in random_candidates:
                logging.debug("Current random discovery: {}".format(rc))
                if rc not in self.othersInfo and rc != self.uid:
                    logging.debug("Sending discovery request to {}".format(rc))
                    self.communication.send(
                        rc,
                        self.create_message_to_send(boolean_flags=["RANDOM_DISCOVERY"]),
                    )
                    req_responses += 1
            self.mutex.release()

            while req_responses > 0:
                logging.debug(
                    "Waiting for {} random discovery responses.".format(req_responses)
                )
                req_responses -= 1
                random_discovery_response = self.responseQueue.get(block=True)
                logging.debug(
                    "Received discovery response from {}".format(
                        random_discovery_response[0]
                    )
                )
                self.mutex.acquire()
                self.othersInfo.update(random_discovery_response[1])
                self.mutex.release()

            self.mutex.acquire()
            self.process_candidates_without_lock(
                random_candidates.union(self.out_edges), response
            )
            self.mutex.release()

            logging.debug("Completed KNN Round {}".format(round))

            logging.debug("OutNodes: {}".format(self.out_edges))

        # Send out_edges and BYE to all

        to_send = self.create_message_to_send(boolean_flags=["KNNBYE"])
        logging.info("Sending KNNByes")
        self.mutex.acquire()
        self.exit_receiver = True
        to_send["CLOSE"] = list(self.out_edges)  # Optimize to only send Yes/No
        for receiver in range(self.mapping.get_n_procs()):
            if receiver != self.uid:
                self.communication.send(receiver, to_send)
        self.mutex.release()
        logging.info("KNNByes Sent")
        t.join()
        logging.info("KNN Receiver Thread Returned")

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
        initial_neighbors=4,
        K=4,
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

        self.rng = Random()
        self.rng.seed(self.uid + 100)

        self.K = K

        self.initial_neighbors = initial_neighbors
        self.in_edges = set()
        self.out_edges = set(
            self.rng.sample(
                list(self.graph.neighbors(self.uid)), self.initial_neighbors
            )
        )
        self.responseQueue = queue.Queue()
        self.mutex = Lock()
        self.othersInfo = {self.uid: list(self.dataset.get_label_distribution())}
        # ld = self.dataset.get_label_distribution()
        # ld_keys = sorted(list(ld.keys()))
        # self.othersInfo = {self.uid: []}
        # for key in range(max(ld_keys) + 1):
        #     if key in ld:
        #         self.othersInfo[self.uid].append(ld[key])
        #     else:
        #         self.othersInfo[self.uid].append(0)
        logging.info("Label Distributions: {}".format(self.othersInfo))

        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
