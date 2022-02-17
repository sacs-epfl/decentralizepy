import json
import logging
import math
import random

import torch

from decentralizepy.sharing.Sharing import Sharing


class RoundRobinPartial(Sharing):
    """
    This class implements the Round robin partial model sharing.

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        alpha=1.0,
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
            Dataset for sharing data. Not implemented yet.
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)
        alpha : float
            Percentage of model to share

        """
        super().__init__(
            rank, machine_id, communication, mapping, graph, model, dataset, log_dir
        )
        self.alpha = alpha
        random.seed(self.mapping.get_uid(rank, machine_id))
        n_params = self.model.count_params()
        self.block_size = math.ceil(self.alpha * n_params)
        self.num_blocks = n_params // self.block_size
        self.current_block = random.randint(0, self.num_blocks - 1)

    def serialized_model(self):
        """
        Convert model to json dict. self.alpha specifies the fraction of model to send.

        Returns
        -------
        dict
            Model converted to json dict

        """

        with torch.no_grad():
            logging.info("Extracting params to send")

            tensors_to_cat = [v.data.flatten() for v in self.model.parameters()]
            T = torch.cat(tensors_to_cat, dim=0)
            block_start = self.current_block * self.block_size
            block_end = min(T.shape[0], (self.current_block + 1) * self.block_size)
            self.current_block = (self.current_block + 1) % self.num_blocks
            T_send = T[block_start:block_end]

            logging.info("Generating dictionary to send")

            m = dict()

            m["block_start"] = block_start
            m["block_end"] = block_end

            m["params"] = T_send.numpy().tolist()

            logging.info("Elements sending: {}".format(len(m["params"])))

            logging.info("Generated dictionary to send")

            for key in m:
                m[key] = json.dumps(m[key])

            logging.info("Converted dictionary to json")
            self.total_data += len(self.communication.encrypt(m["params"]))
            return m

    def deserialized_model(self, m):
        """
        Convert received json dict to state_dict.

        Parameters
        ----------
        m : dict
            json dict received

        Returns
        -------
        state_dict
            state_dict of received

        """
        with torch.no_grad():
            state_dict = self.model.state_dict()

            shapes = []
            lens = []
            tensors_to_cat = []
            for _, v in state_dict.items():
                shapes.append(v.shape)
                t = v.flatten()
                lens.append(t.shape[0])
                tensors_to_cat.append(t)

            T = torch.cat(tensors_to_cat, dim=0)
            block_start = json.loads(m["block_start"])
            block_end = json.loads(m["block_end"])
            T[block_start:block_end] = torch.tensor(json.loads(m["params"]))
            start_index = 0
            for i, key in enumerate(state_dict):
                end_index = start_index + lens[i]
                state_dict[key] = T[start_index:end_index].reshape(shapes[i])
                start_index = end_index

            return state_dict
