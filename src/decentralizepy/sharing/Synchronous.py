import logging
from collections import deque

import torch


class Synchronous:
    """
    Synchronous training

    """

    def __init__(
        self, rank, machine_id, communication, mapping, graph, model, dataset, log_dir
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)

        """
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.graph = graph
        self.model = model
        self.dataset = dataset
        self.communication_round = 0
        self.log_dir = log_dir
        self.total_data = 0

        self.peer_deques = dict()
        self.my_neighbors = self.graph.neighbors(self.uid)
        for n in self.my_neighbors:
            self.peer_deques[n] = deque()

        with torch.no_grad():
            self.init_model = {}
            for k, v in self.model.state_dict().items():
                self.init_model[k] = v.clone().detach()

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for _, i in self.peer_deques.items():
            if len(i) == 0:
                return False
        return True

    def get_neighbors(self, neighbors):
        """
        Choose which neighbors to share with

        Parameters
        ----------
        neighbors : list(int)
            List of all neighbors

        Returns
        -------
        list(int)
            Neighbors to share with

        """
        # modify neighbors here
        return neighbors

    def serialized_gradient(self):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        dict
            Model converted to dict

        """
        m = dict()
        for key, val in self.model.state_dict().items():
            m[key] = val - self.init_model[key]  # this is -lr*gradient
        self.total_data += len(self.communication.encrypt(m))
        return m

    def serialized_model(self):
        """
        Convert model to a dictionary. Here we can choose how much to share

        Returns
        -------
        dict
            Model converted to dict

        """
        m = dict()
        for key, val in self.model.state_dict().items():
            m[key] = val.clone().detach()
        self.total_data += len(self.communication.encrypt(m))
        return m

    def deserialized_model(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        return m

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        pass

    def _post_step(self):
        """
        Called at the end of step.

        """
        with torch.no_grad():
            self.init_model = {}
            for k, v in self.model.state_dict().items():
                self.init_model[k] = v.clone().detach()

    def _apply_gradients(self):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            total = dict()
            for i, n in enumerate(self.peer_deques):
                gradient = self.peer_deques[n].popleft()
                logging.debug(
                    "Applying gradient from neighbor {}".format(
                        n,
                    )
                )
                grad = self.deserialized_model(gradient)

                for key, value in grad.items():
                    if key in total:
                        total[key] += value
                    else:
                        total[key] = value

            my_grad = self.serialized_gradient()
            for key, value in my_grad.items():
                if key in total:
                    total[key] += value
                else:
                    total[key] = value
        new_model = {}
        for key, value in self.init_model.items():
            new_model[key] = value + total[key] * (1 / (len(self.my_neighbors) + 1))

        self.model.load_state_dict(new_model)

    def step(self):
        """
        Perform a sharing step. Implements D-PSGD.

        """
        self._pre_step()
        logging.info("--- COMMUNICATION ROUND {} ---".format(self.communication_round))
        if self.uid != 0:
            gradient = self.serialized_gradient()
            # Should be only one neighbour

            self.communication.send(0, gradient)

            logging.info("Waiting for messages from central node")
            sender, data = self.communication.receive()
            logging.debug("Received model from {}".format(sender))
            logging.info(
                "Deserialized received model from {} of iteration {}".format(
                    sender, self.communication_round
                )
            )
            self.model.load_state_dict(data)
        else:
            logging.info("Waiting for messages from leaf nodes")
            while not self.received_from_all():
                sender, data = self.communication.receive()
                logging.debug("Received gradient from {}".format(sender))
                self.peer_deques[sender].append(data)
                logging.info(
                    "Deserialized gradient model from {} of iteration {}".format(
                        sender, self.communication_round
                    )
                )
            self._apply_gradients()

            data = self.serialized_model()

            all_neighbors = self.graph.neighbors(self.uid)
            iter_neighbors = self.get_neighbors(all_neighbors)
            for neighbor in iter_neighbors:
                self.communication.send(neighbor, data)

        self.communication_round += 1
        self._post_step()
