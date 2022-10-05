import json
import logging
import os

import numpy as np
import torch
import torch.fft as fft

from decentralizepy.sharing.PartialModel import PartialModel


def change_transformer_fft(x):
    """
    Transforms the model changes into frequency domain

    Parameters
    ----------
    x : torch.Tensor
        Model change in the space domain

    Returns
    -------
    x : torch.Tensor
        Representation of the change int the frequency domain
    """
    return fft.rfft(x)


class FFT(PartialModel):
    """
    This class implements the fft version of model sharing
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
        change_based_selection=True,
        save_accumulated="",
        accumulation=True,
        accumulate_averaging_changes=False,
        compress=False,
        compression_package=None,
        compression_class=None,
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
        change_based_selection : bool
            use frequency change to select topk frequencies
        save_accumulated : bool
            True if accumulated weight change in the frequency domain should be written to file. In case of accumulation
            the accumulated change is stored.
        accumulation : bool
            True if the the indices to share should be selected based on accumulated frequency change
        accumulate_averaging_changes: bool
            True if the accumulation should account the model change due to averaging

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
            alpha,
            dict_ordered,
            save_shared,
            metadata_cap,
            accumulation,
            save_accumulated,
            change_transformer_fft,
            accumulate_averaging_changes,
            compress,
            compression_package,
            compression_class,
        )
        self.change_based_selection = change_based_selection

    def apply_fft(self):
        """
        Does fft transformation of the model parameters and selects topK (alpha) of them in the frequency domain
        based on the undergone change during the current training step

        Returns
        -------
        tuple
            (a,b). a: selected fft frequencies (complex numbers), b: Their indices.

        """

        logging.info("Returning fft compressed model weights")
        with torch.no_grad():

            flat_fft = self.pre_share_model_transformed
            if self.change_based_selection:
                diff = self.model.model_change
                _, index = torch.topk(
                    diff.abs(), round(self.alpha * len(diff)), dim=0, sorted=False
                )
            else:
                _, index = torch.topk(
                    flat_fft.abs(),
                    round(self.alpha * len(flat_fft)),
                    dim=0,
                    sorted=False,
                )
        index, _ = torch.sort(index)
        return flat_fft[index], index

    def serialized_model(self):
        """
        Convert model to json dict. self.alpha specifies the fraction of model to send.

        Returns
        -------
        dict
            Model converted to json dict

        """
        m = dict()
        if self.alpha >= self.metadata_cap:  # Share fully
            data = self.pre_share_model_transformed
            m["params"] = data.numpy()
            if self.model.accumulated_changes is not None:
                self.model.accumulated_changes = torch.zeros_like(
                    self.model.accumulated_changes
                )
            return self.compress_data(m)

        with torch.no_grad():
            topk, indices = self.apply_fft()
            self.model.shared_parameters_counter[indices] += 1
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

            if not self.dict_ordered:
                raise NotImplementedError

            m["alpha"] = self.alpha
            m["params"] = topk.numpy()
            m["indices"] = indices.numpy().astype(np.int32)
            m["send_partial"] = True

        return self.compress_data(m)

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
        m = self.decompress_data(m)

        ret = dict()
        if "send_partial" not in m:
            params = m["params"]
            params_tensor = torch.tensor(params)
            ret["params"] = params_tensor

        with torch.no_grad():
            if not self.dict_ordered:
                raise NotImplementedError

            indices = m["indices"]
            alpha = m["alpha"]
            params = m["params"]

            params_tensor = torch.tensor(params)
            indices_tensor = torch.tensor(indices, dtype=torch.long)
            ret["indices"] = indices_tensor
            ret["params"] = params_tensor
            ret["send_partial"] = True
        return ret

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            total = None
            weight_total = 0
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            pre_share_model = torch.cat(tensors_to_cat, dim=0)
            flat_fft = self.change_transformer(pre_share_model)

            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data)
                params = data["params"]
                if "indices" in data:
                    indices = data["indices"]
                    # use local data to complement
                    topkf = flat_fft.clone().detach()
                    topkf[indices] = params
                else:
                    topkf = params

                weight = 1 / (max(len(peer_deques), degree) + 1)  # Metro-Hastings
                weight_total += weight
                if total is None:
                    total = weight * topkf
                else:
                    total += weight * topkf

            # Metro-Hastings
            total += (1 - weight_total) * flat_fft
            reverse_total = fft.irfft(total)

            start_index = 0
            std_dict = {}
            for i, key in enumerate(self.model.state_dict()):
                end_index = start_index + self.lens[i]
                std_dict[key] = reverse_total[start_index:end_index].reshape(
                    self.shapes[i]
                )
                start_index = end_index

        self.model.load_state_dict(std_dict)
        self._post_step()
        self.communication_round += 1
