import json
import logging
import os

import numpy as np
import pywt
import torch

from decentralizepy.sharing.PartialModel import PartialModel


def change_transformer_wavelet(x, wavelet, level):
    """
    Transforms the model changes into wavelet frequency domain

    Parameters
    ----------
    x : torch.Tensor
        Model change in the space domain
    wavelet : str
        name of the wavelet to be used in gradient compression
    level: int
        name of the wavelet to be used in gradient compression

    Returns
    -------
    x : torch.Tensor
        Representation of the change int the wavelet domain
    """
    coeff = pywt.wavedec(x, wavelet, level=level)
    data, coeff_slices = pywt.coeffs_to_array(coeff)
    return torch.from_numpy(data.ravel())


class Wavelet(PartialModel):
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
        wavelet="haar",
        level=4,
        change_based_selection=True,
        save_accumulated="",
        accumulation=False,
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
        wavelet: str
            name of the wavelet to be used in gradient compression
        level: int
            name of the wavelet to be used in gradient compression
        change_based_selection : bool
            use frequency change to select topk frequencies
        save_accumulated : bool
            True if accumulated weight change in the wavelet domain should be written to file. In case of accumulation
            the accumulated change is stored.
        accumulation : bool
            True if the the indices to share should be selected based on accumulated frequency change
        accumulate_averaging_changes: bool
            True if the accumulation should account the model change due to averaging
        """
        self.wavelet = wavelet
        self.level = level

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
            lambda x: change_transformer_wavelet(x, wavelet, level),
            accumulate_averaging_changes,
            compress,
            compression_package,
            compression_class,
        )

        self.change_based_selection = change_based_selection

        # Do a dummy transform to get the shape and coefficents slices
        coeff = pywt.wavedec(self.init_model.numpy(), self.wavelet, level=self.level)
        data, coeff_slices = pywt.coeffs_to_array(coeff)
        self.wt_shape = data.shape
        self.coeff_slices = coeff_slices

    def apply_wavelet(self):
        """
        Does wavelet transformation of the model parameters and selects topK (alpha) of them in the frequency domain
        based on the undergone change during the current training step

        Returns
        -------
        tuple
            (a,b). a: selected wavelet coefficients, b: Their indices.

        """

        logging.debug("Returning wavelet compressed model weights")
        data = self.pre_share_model_transformed
        if self.change_based_selection:
            diff = self.model.model_change
            _, index = torch.topk(
                diff.abs(),
                round(self.alpha * len(diff)),
                dim=0,
                sorted=False,
            )
        else:
            _, index = torch.topk(
                data.abs(),
                round(self.alpha * len(data)),
                dim=0,
                sorted=False,
            )
        index, _ = torch.sort(index)
        return data[index], index

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
            topk, indices = self.apply_wavelet()
            self.model.shared_parameters_counter[indices] += 1
            self.model.rewind_accumulation(indices)
            if self.save_shared:
                shared_params = dict()
                shared_params["order"] = list(self.model.state_dict().keys())
                shapes = dict()
                for k, v in self.model.state_dict().items():
                    shapes[k] = list(v.shape)
                shared_params["shapes"] = shapes

                # is slow
                shared_params[self.communication_round] = indices.tolist()

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
        m = self.decompress_data(m)
        ret = dict()
        if "send_partial" not in m:
            params = m["params"]
            params_tensor = torch.tensor(params)
            ret["params"] = params_tensor
            return ret

        with torch.no_grad():
            if not self.dict_ordered:
                raise NotImplementedError
            alpha = m["alpha"]

            params_tensor = torch.tensor(m["params"])
            indices_tensor = torch.tensor(m["indices"], dtype=torch.long)
            ret = dict()
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
            wt_params = self.pre_share_model_transformed
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
                    topkwf = wt_params.clone().detach()
                    topkwf[indices] = params
                    topkwf = topkwf.reshape(self.wt_shape)
                else:
                    topkwf = params.reshape(self.wt_shape)

                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                if total is None:
                    total = weight * topkwf
                else:
                    total += weight * topkwf

            # Metro-Hastings
            total += (1 - weight_total) * wt_params

            avg_wf_params = pywt.array_to_coeffs(
                total.numpy(), self.coeff_slices, output_format="wavedec"
            )
            reverse_total = torch.from_numpy(
                pywt.waverec(avg_wf_params, wavelet=self.wavelet)
            )

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

    def _averaging_server(self, peer_deques):
        """
        Averages the received models of all working nodes

        """
        with torch.no_grad():
            total = None
            wt_params = self.pre_share_model_transformed
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
                    topkwf = wt_params.clone().detach()
                    topkwf[indices] = params
                    topkwf = topkwf.reshape(self.wt_shape)
                else:
                    topkwf = params.reshape(self.wt_shape)

                weight = 1 / len(peer_deques)
                if total is None:
                    total = weight * topkwf
                else:
                    total += weight * topkwf

            avg_wf_params = pywt.array_to_coeffs(
                total.numpy(), self.coeff_slices, output_format="wavedec"
            )
            reverse_total = torch.from_numpy(
                pywt.waverec(avg_wf_params, wavelet=self.wavelet)
            )

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
