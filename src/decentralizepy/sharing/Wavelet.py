import json
import logging
import os
from pathlib import Path
from time import time

import numpy as np
import pywt
import torch

from decentralizepy.sharing.Sharing import Sharing


class Wavelet(Sharing):
    """
    This class implements the wavelet version of model sharing
    It is based on PartialModel.py

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
        save_shared=False,
        metadata_cap=1.0,
        pickle=True,
        wavelet="haar",
        level=4,
        change_based_selection=True,
        accumulation=False,
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
        save_shared : bool
            Specifies if the indices of shared parameters should be logged
        metadata_cap : float
            Share full model when self.alpha > metadata_cap
        pickle : bool
            use pickle to serialize the model parameters
        wavelet: str
            name of the wavelet to be used in gradient compression
        level: int
            name of the wavelet to be used in gradient compression
        change_based_selection : bool
            use frequency change to select topk frequencies
        accumulation : bool
            True if the the indices to share should be selected based on accumulated frequency change
        """
        super().__init__(
            rank, machine_id, communication, mapping, graph, model, dataset, log_dir
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered
        self.save_shared = save_shared
        self.metadata_cap = metadata_cap
        self.total_meta = 0

        self.pickle = pickle
        self.wavelet = wavelet
        self.level = level
        self.accumulation = accumulation

        logging.info("subsampling pickling=" + str(pickle))

        if self.save_shared:
            # Only save for 2 procs: Save space
            if rank != 0 or rank != 1:
                self.save_shared = False

        if self.save_shared:
            self.folder_path = os.path.join(
                self.log_dir, "shared_params/{}".format(self.rank)
            )
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        self.change_based_selection = change_based_selection

    def apply_wavelet(self):
        """
        Does wavelet transformation of the model parameters and selects topK (alpha) of them in the frequency domain
        based on the undergone change during the current training step

        Returns
        -------
        tuple
            (a,b). a: selected wavelet coefficients, b: Their indices.

        """

        logging.info("Returning dwt compressed model weights")
        tensors_to_cat = [v.data.flatten() for _, v in self.model.state_dict().items()]
        concated = torch.cat(tensors_to_cat, dim=0)

        coeff = pywt.wavedec(concated.numpy(), self.wavelet, level=self.level)
        data, coeff_slices = pywt.coeffs_to_array(
            coeff
        )  # coeff_slices will be reproduced on the receiver
        data = data.ravel()

        if self.change_based_selection:
            assert len(self.model.accumulated_gradients) == 1
            diff = self.model.accumulated_gradients[0]
            _, index = torch.topk(
                diff.abs(),
                round(self.alpha * len(data)),
                dim=0,
                sorted=False,
            )
        else:
            _, index = torch.topk(
                torch.from_numpy(data).abs(),
                round(self.alpha * len(data)),
                dim=0,
                sorted=False,
            )

        return torch.from_numpy(data[index]), index

    def serialized_model(self):
        """
        Convert model to json dict. self.alpha specifies the fraction of model to send.

        Returns
        -------
        dict
            Model converted to json dict

        """
        if self.alpha > self.metadata_cap:  # Share fully
            return super().serialized_model()

        with torch.no_grad():
            topk, indices = self.apply_wavelet()

            self.model.rewind_accumulation(indices)

            if self.save_shared:
                shared_params = dict()
                shared_params["order"] = list(self.model.state_dict().keys())
                shapes = dict()
                for k, v in self.model.state_dict().items():
                    shapes[k] = list(v.shape)
                shared_params["shapes"] = shapes

                shared_params[self.communication_round] = indices.tolist()  # is slow

                shared_params["alpha"] = self.alpha

                with open(
                    os.path.join(
                        self.folder_path,
                        "{}_shared_params.json".format(self.communication_round + 1),
                    ),
                    "w",
                ) as of:
                    json.dump(shared_params, of)

            m = dict()

            if not self.dict_ordered:
                raise NotImplementedError

            m["alpha"] = self.alpha

            m["params"] = topk.numpy()

            m["indices"] = indices.numpy().astype(np.int32)

            self.total_data += len(self.communication.encrypt(m["params"]))
            self.total_meta += len(self.communication.encrypt(m["indices"])) + len(
                self.communication.encrypt(m["alpha"])
            )

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
        if self.alpha > self.metadata_cap:  # Share fully
            return super().deserialized_model(m)

        with torch.no_grad():
            state_dict = self.model.state_dict()

            if not self.dict_ordered:
                raise NotImplementedError

            shapes = []
            lens = []
            tensors_to_cat = []
            for _, v in state_dict.items():
                shapes.append(v.shape)
                t = v.flatten()
                lens.append(t.shape[0])
                tensors_to_cat.append(t)

            T = torch.cat(tensors_to_cat, dim=0)

            indices = m["indices"]
            alpha = m["alpha"]
            params = m["params"]

            params_tensor = torch.tensor(params)
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            ret = dict()
            ret["indices"] = indices_tensor
            ret["params"] = params_tensor
            return ret

    def step(self):
        """
        Perform a sharing step. Implements D-PSGD.

        """
        t_start = time()
        data = self.serialized_model()
        t_post_serialize = time()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        all_neighbors = self.graph.neighbors(my_uid)
        iter_neighbors = self.get_neighbors(all_neighbors)
        data["degree"] = len(all_neighbors)
        data["iteration"] = self.communication_round
        for neighbor in iter_neighbors:
            self.communication.send(neighbor, data)
        t_post_send = time()
        logging.info("Waiting for messages from neighbors")
        while not self.received_from_all():
            sender, data = self.communication.receive()
            logging.debug("Received model from {}".format(sender))
            degree = data["degree"]
            iteration = data["iteration"]
            del data["degree"]
            del data["iteration"]
            self.peer_deques[sender].append((degree, iteration, data))
            logging.info(
                "Deserialized received model from {} of iteration {}".format(
                    sender, iteration
                )
            )
        t_post_recv = time()

        logging.info("Starting model averaging after receiving from all neighbors")
        total = None
        weight_total = 0

        # FFT of this model
        shapes = []
        lens = []
        tensors_to_cat = []
        # TODO: should we detach
        for _, v in self.model.state_dict().items():
            shapes.append(v.shape)
            t = v.flatten()
            lens.append(t.shape[0])
            tensors_to_cat.append(t)
        concated = torch.cat(tensors_to_cat, dim=0)
        coeff = pywt.wavedec(concated.numpy(), self.wavelet, level=self.level)
        wt_params, coeff_slices = pywt.coeffs_to_array(
            coeff
        )  # coeff_slices will be reproduced on the receiver
        shape = wt_params.shape
        wt_params = wt_params.ravel()

        for i, n in enumerate(self.peer_deques):
            degree, iteration, data = self.peer_deques[n].popleft()
            logging.debug(
                "Averaging model from neighbor {} of iteration {}".format(n, iteration)
            )
            data = self.deserialized_model(data)
            params = data["params"]
            indices = data["indices"]
            # use local data to complement
            topkwf = wt_params.copy()  # .clone().detach()
            topkwf[indices] = params
            topkwf = torch.from_numpy(topkwf.reshape(shape))

            weight = 1 / (max(len(self.peer_deques), degree) + 1)  # Metro-Hastings
            weight_total += weight
            if total is None:
                total = weight * topkwf
            else:
                total += weight * topkwf

        # Metro-Hastings
        total += (1 - weight_total) * wt_params

        avg_wf_params = pywt.array_to_coeffs(
            total, coeff_slices, output_format="wavedec"
        )
        reverse_total = torch.from_numpy(
            pywt.waverec(avg_wf_params, wavelet=self.wavelet)
        )

        start_index = 0
        std_dict = {}
        for i, key in enumerate(self.model.state_dict()):
            end_index = start_index + lens[i]
            std_dict[key] = reverse_total[start_index:end_index].reshape(shapes[i])
            start_index = end_index

        self.model.load_state_dict(std_dict)

        logging.info("Model averaging complete")

        self.communication_round += 1

        t_end = time()

        logging.info(
            "Sharing::step | Serialize: %f; Send: %f; Recv: %f; Averaging: %f; Total: %f",
            t_post_serialize - t_start,
            t_post_send - t_post_serialize,
            t_post_recv - t_post_send,
            t_end - t_post_recv,
            t_end - t_start,
        )
