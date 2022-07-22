import importlib
import json
import logging
import math
import os
from collections import deque

import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.communication.TCP import TCP
from decentralizepy.graphs.Graph import Graph
from decentralizepy.graphs.Star import Star
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node
from decentralizepy.train_test_evaluation import TrainTestHelper


class DPSGDNode(Node):
    """
    This class defines the node for DPSGD

    """

    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        if self.uid == 0:
            dataset = self.dataset
            if self.centralized_train_eval:
                dataset_params_copy = self.dataset_params.copy()
                if "sizes" in dataset_params_copy:
                    del dataset_params_copy["sizes"]
                self.whole_dataset = self.dataset_class(
                    self.rank,
                    self.machine_id,
                    self.mapping,
                    sizes=[1.0],
                    **dataset_params_copy
                )
                dataset = self.whole_dataset
            if self.centralized_test_eval:
                tthelper = TrainTestHelper(
                    dataset,  # self.whole_dataset,
                    # self.model_test, # todo: this only works if eval_train is set to false
                    self.model,
                    self.loss,
                    self.weights_store_dir,
                    self.mapping.get_n_procs(),
                    self.trainer,
                    self.testing_comm,
                    self.star,
                    self.threads_per_proc,
                    eval_train=self.centralized_train_eval,
                )

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            self.trainer.train(self.dataset)
            to_send = self.sharing.get_data_to_send()

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                sender, data = self.receive()

                if "HELLO" in data:
                    logging.critical(
                        "Received unexpected {} from {}".format("HELLO", sender)
                    )
                    raise RuntimeError("A neighbour wants to connect during training!")
                elif "BYE" in data:
                    logging.debug("Received {} from {}".format("BYE", sender))
                    self.barrier.remove(sender)
                else:
                    logging.debug("Received message from {}".format(sender))
                    self.peer_deques[sender].append(data)

            self.sharing._averaging(self.peer_deques)

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "grad_mean": {},
                    "grad_std": {},
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
            if hasattr(self.sharing, "mean"):
                results_dict["grad_mean"][iteration + 1] = self.sharing.mean
            if hasattr(self.sharing, "std"):
                results_dict["grad_std"][iteration + 1] = self.sharing.std

            rounds_to_train_evaluate -= 1

            if rounds_to_train_evaluate == 0 and not self.centralized_train_eval:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

            rounds_to_test -= 1

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                if self.centralized_test_eval:
                    if self.uid == 0:
                        ta, tl, trl = tthelper.train_test_evaluation(iteration)
                        results_dict["test_acc"][iteration + 1] = ta
                        results_dict["test_loss"][iteration + 1] = tl
                        if trl is not None:
                            results_dict["train_loss"][iteration + 1] = trl
                    else:
                        self.testing_comm.send(0, self.model.get_weights())
                        sender, data = self.testing_comm.receive()
                        assert sender == 0 and data == "finished"
                else:
                    logging.info("Evaluating on test set.")
                    ta, tl = self.dataset.test(self.model, self.loss)
                    results_dict["test_acc"][iteration + 1] = ta
                    results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

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
        centralized_train_eval,
        centralized_test_eval,
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
        centralized_train_eval : bool
            If set the train set evaluation happens at the node with uid 0
        centralized_test_eval : bool
            If set the train set evaluation happens at the node with uid 0
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
        self.centralized_train_eval = centralized_train_eval
        self.centralized_test_eval = centralized_test_eval
        self.sent_disconnections = False

        logging.info("Rank: %d", self.rank)
        logging.info("type(graph): %s", str(type(self.rank)))
        logging.info("type(mapping): %s", str(type(self.mapping)))

        if centralized_test_eval or centralized_train_eval:
            self.star = Star(self.mapping.get_n_procs())

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
        if self.centralized_test_eval:
            self.testing_comm = TCP(
                self.rank,
                self.machine_id,
                self.mapping,
                self.star.n_procs,
                self.addresses_filepath,
                offset=self.star.n_procs,
            )
            self.testing_comm.connect_neighbors(self.star.neighbors(self.uid))

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
        centralized_train_eval=False,
        centralized_test_eval=True,
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
        centralized_train_eval : bool
            If set the train set evaluation happens at the node with uid 0
        centralized_test_eval : bool
            If set the train set evaluation happens at the node with uid 0
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
            centralized_train_eval,
            centralized_test_eval,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = deque()
        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        for n in self.my_neighbors:
            self.peer_deques[n] = deque()

        self.connect_neighbors()

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if len(self.peer_deques[k]) == 0:
                return False
        return True

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
        centralized_train_eval=0,
        centralized_test_eval=1,
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
        centralized_train_eval : int
            If set then the train set evaluation happens at the node with uid 0.
            Note: If it is True then centralized_test_eval needs to be true as well!
        centralized_test_eval : int
            If set then the trainset evaluation happens at the node with uid 0
        args : optional
            Other arguments

        """
        centralized_train_eval = centralized_train_eval == 1
        centralized_test_eval = centralized_test_eval == 1
        # If centralized_train_eval is True then centralized_test_eval needs to be true as well!
        assert not centralized_train_eval or centralized_test_eval

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
            centralized_train_eval == 1,
            centralized_test_eval == 1,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
