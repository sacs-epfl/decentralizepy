import importlib
import json
import logging
import math
import os

import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping


class Node:
    """
    This class defines the node (entity that performs learning, sharing and communication).

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
            force=True,
        )

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        test_after,
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
        log_dir : str
            Logging directory
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0

        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.iterations = iterations
        self.test_after = test_after
        self.reset_optimizer = reset_optimizer

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

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
        torch.manual_seed(dataset_configs["random_seed"])
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
        test_after=5,
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
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            test_after,
            reset_optimizer,
        )
        self.init_log(log_dir, rank, log_level)
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])
        self.init_sharing(config["SHARING"])

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        self.communication.connect_neighbors(self.graph.neighbors(self.uid))
        rounds_to_test = self.test_after

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            self.trainer.train(self.dataset)

            self.sharing.step()

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            loss_after_sharing = self.trainer.eval_loss(self.dataset)

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

            results_dict["train_loss"][iteration + 1] = loss_after_sharing
            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.sharing, "total_meta"):
                results_dict["total_meta"][iteration + 1] = self.sharing.total_meta
            if hasattr(self.sharing, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.sharing.total_data
            if hasattr(self.sharing, "mean"):
                results_dict["grad_mean"][iteration + 1] = self.sharing.mean
            if hasattr(self.sharing, "std"):
                results_dict["grad_std"][iteration + 1] = self.sharing.std

            self.save_plot(
                results_dict["train_loss"],
                "train_loss",
                "Training Loss",
                "Communication Rounds",
                os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
            )

            rounds_to_test -= 1

            if self.dataset.__testing__ and rounds_to_test == 0:
                logging.info("Evaluating on test set.")
                rounds_to_test = self.test_after
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                self.save_plot(
                    results_dict["test_loss"],
                    "test_loss",
                    "Testing Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_test_loss.png".format(self.rank)),
                )
                self.save_plot(
                    results_dict["test_acc"],
                    "test_acc",
                    "Testing Accuracy",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_test_acc.png".format(self.rank)),
                )

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

        self.communication.disconnect_neighbors()
        logging.info("All neighbors disconnected. Process complete!")

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
        reset_optimizer=1,
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
        log_dir : str
            Logging directory
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        total_threads = os.cpu_count()
        threads_per_proc = max(math.floor(total_threads / mapping.procs_per_machine), 1)
        torch.set_num_threads(threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            log_level,
            test_after,
            reset_optimizer,
            *args
        )
        logging.info(
            "Each proc uses %d threads out of %d.", threads_per_proc, total_threads
        )

        self.run()
