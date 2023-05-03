import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from decentralizepy.sharing.Sharing import Sharing
from decentralizepy.utils import conditional_value, identity


class STC(Sharing):
    """
    This class implements STC from https://ieeexplore.ieee.org/document/8889996

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
        dict_ordered=True,
        change_transformer=identity,
        compress=True,
        compression_package="decentralizepy.compression.EliasFpzipLossy",
        compression_class="EliasFpzipLossy",
        float_precision=8,
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
        alpha : float
            Percentage of model to share
        dict_ordered : bool
            Specifies if the python dict maintains the order of insertion
        change_transformer : function
            Function to transform the change in the model
        compress : bool
            Whether to compress the data
        compression_package : str
            Package to use for compression
        compression_class : str
            Class to use for compression

        """
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            compress,
            compression_package,
            compression_class,
            float_precision
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered
        self.change_transformer = change_transformer

        # getting the initial model
        self.shapes = []
        self.lens = []
        with torch.no_grad():
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                self.shapes.append(v.shape)
                t = v.flatten()
                self.lens.append(t.shape[0])
                tensors_to_cat.append(t)
            self.prev_model = torch.cat(tensors_to_cat, dim=0)
            self.residuals = torch.zeros_like(self.prev_model)
            self.model.model_change = torch.zeros_like(self.prev_model)
        self.number_of_params = self.prev_model.shape[0]

    def compress_data(self, data):
        result = dict(data)
        if self.compress:
            if "indices" in result:
                result["indices"] = self.compressor.compress(result["indices"])
            if "params" in result:
                result["params"] = self.compressor.compress_float(result["params"])
        return result

    def decompress_data(self, data):
        data = dict(data)
        if self.compress:
            if "indices" in data:
                data["indices"] = self.compressor.decompress(data["indices"])
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        return data
    
    def flatten(self, m):
        """
        Method to flatten a torch model

        Parameters
        ----------
        m : state_dict
            Model to flatten

        Returns
        -------
        torch.tensor
            Flattened model

        """
            
        with torch.no_grad():
            tensors_to_cat = []
            for _, v in m.items():
                t = v.flatten()
                tensors_to_cat.append(t)
            return torch.cat(tensors_to_cat, dim=0)
    
    def unflatten(self, m):
        """
        Method to unflatten a torch model

        Parameters
        ----------
        m : torch.tensor
            Flattened model
        
        Returns
        -------
        state_dict
            Unflattened model

        """
        with torch.no_grad():
            result = dict()
            start = 0
            for i, key in enumerate(self.model.state_dict()):
                end = start + self.lens[i]
                result[key] = m[start:end].view(self.shapes[i])
                start = end
        return result

    def extract_top_gradients(self):
        """
        Extract the indices and values of the topK gradients.
        The gradients must have been accumulated.

        Returns
        -------
        tuple
            (a,b). a: The magnitudes of the topK gradients, b: Their indices.

        """

        logging.debug("Returning topk gradients")
        G_topk = torch.abs(self.model.model_change)
        _, index = torch.topk(
            G_topk, round(self.alpha * G_topk.shape[0]), dim=0, sorted=True
        )

        index, _ = torch.sort(index)
        return self.model.model_change[index], index

    def serialized_model(self):
        """
        Convert model to a dict. self.alpha specifies the fraction of model to send.

        Returns
        -------
        dict
            Model converted to a dict

        """

        with torch.no_grad():
            G_topk_values, G_topk_indices = self.extract_top_gradients()

            logging.debug("Extracting topk params")

            T_topk = G_topk_values

            logging.debug("Generating dictionary to send")

            m = dict()

            if not self.dict_ordered:
                raise NotImplementedError

            m["alpha"] = self.alpha

            m["indices"] = G_topk_indices.numpy().astype(np.int32)

            m["params"] = T_topk.numpy()

            assert len(m["indices"]) == len(m["params"])
            logging.debug("Elements sending: {}".format(len(m["indices"])))

            return self.compress_data(m)

    def deserialized_model(self, m, return_flat_tensor=False):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            dict received
        return_flat_tensor : bool
            Whether to return a flat tensor or a state_dict        

        Returns
        -------
        torch.tensor or state_dict
            Model converted to a torch.tensor or state_dict

        """

        with torch.no_grad():
            m = self.decompress_data(m)

            if not self.dict_ordered:
                raise NotImplementedError

            T = torch.zeros(self.number_of_params)
            index_tensor = torch.tensor(m["indices"], dtype=torch.long)
            logging.debug("Original tensor: {}".format(T[index_tensor]))
            T[index_tensor] = torch.tensor(m["params"])
            logging.debug("Final tensor: {}".format(T[index_tensor]))

            return T if return_flat_tensor else self.unflatten(T)

    def _pre_step(self):
        """
        Called at the beginning of step.

        Algorithm 2 of the paper: Line 10.

        """
        logging.debug("PartialModel _pre_step")
        with torch.no_grad():
            flattened_model = self.flatten(self.model.state_dict())
            self.model.model_change = flattened_model - self.prev_model + self.residuals
            self.prev_model = flattened_model
            

    def _post_step(self):
        """
        Called at the end of step.

        """
        logging.debug("PartialModel _post_step")
        with torch.no_grad():
            return

    def process_received(self, m = None):
        """
        Process the received dict.
        Algorithm 2 of the paper: Lines 7-9.

        Parameters
        ----------
        m : dict
            dict received

        """
        logging.debug("PartialModel process_received")
        with torch.no_grad():
            if m is None:
                deserialized = 0
            else:
                if "iteration" in m:
                    iteration = m["iteration"]
                    del m["iteration"]
                if "degree" in m:
                    del m["degree"]
                if "CHANNEL" in m:
                    del m["CHANNEL"]
                deserialized = self.deserialized_model(m, return_flat_tensor=True)
            cur_model = self.flatten(self.model.state_dict()) + deserialized
            self.model.load_state_dict(self.unflatten(cur_model))

    def get_data_to_send(self, *args, **kwargs):
        with torch.no_grad():
            self._pre_step()
            data = self.serialized_model() # Algorithm 2 of the paper: Line 11
            self.residuals = self.model.model_change - self.deserialized_model(data, return_flat_tensor=True) # Algorithm 2 of the paper: Line 12
            data["iteration"] = self.communication_round
            return data
    
    def server_broadcast(self, *args, **kwargs):
        """
        Broadcast the model to all working nodes
        Algorithm 2 of the paper: Lines 19-22.

        """
        with torch.no_grad():
            data = self.serialized_model()
            self.residuals = self.model.model_change - self.deserialized_model(data, return_flat_tensor=True)
            self.process_received(data) # A trick to reuse the code :)
            data["iteration"] = self.communication_round
            return data
    
    def _averaging_server(self, peer_deques):
        """
        Averages the received models of all working nodes
        Algorithm 2 of the paper: Lines 17-18.

        """
        with torch.no_grad():
            total = torch.zeros_like(self.model.model_change)
            weight = 1.0 / len(peer_deques)
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                if "iteration" in data:
                    iteration = data["iteration"]
                    del data["iteration"]
                if "degree" in data:
                    del data["degree"]
                if "CHANNEL" in data:
                    del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data, return_flat_tensor=True)
                total += weight * data

            self.model.model_change = self.residuals + total

            self.communication_round += 1
        return total