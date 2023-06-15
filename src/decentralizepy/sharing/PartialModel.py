import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from decentralizepy.sharing.Sharing import Sharing
from decentralizepy.utils import conditional_value, identity


class PartialModel(Sharing):
    """
    This class implements the vanilla version of partial model sharing.

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
        accumulation=False,
        save_accumulated="",
        change_transformer=identity,
        accumulate_averaging_changes=False,
        compress=False,
        compression_package=None,
        compression_class=None,
        float_precision=None,
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
        accumulation : bool
            True if the the indices to share should be selected based on accumulated frequency change
        save_accumulated : bool
            True if accumulated weight change should be written to file. In case of accumulation the accumulated change
            is stored. If a change_transformer is used then the transformed change is stored.
        change_transformer : (x: Tensor) -> Tensor
            A function that transforms the model change into other domains. Default: identity function
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
            compress,
            compression_package,
            compression_class,
            float_precision,
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered
        self.save_shared = save_shared
        self.metadata_cap = metadata_cap
        self.accumulation = accumulation
        self.save_accumulated = conditional_value(save_accumulated, "", False)
        self.change_transformer = change_transformer
        self.accumulate_averaging_changes = accumulate_averaging_changes

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
            self.init_model = torch.cat(tensors_to_cat, dim=0)
            if self.accumulation:
                self.model.accumulated_changes = torch.zeros_like(
                    self.change_transformer(self.init_model)
                )
                self.prev = self.init_model
        self.number_of_params = self.init_model.shape[0]
        if self.save_accumulated:
            self.model_change_path = os.path.join(
                self.log_dir, "model_change/{}".format(self.rank)
            )
            Path(self.model_change_path).mkdir(parents=True, exist_ok=True)

            self.model_val_path = os.path.join(
                self.log_dir, "model_val/{}".format(self.rank)
            )
            Path(self.model_val_path).mkdir(parents=True, exist_ok=True)

        # Only save for 2 procs: Save space
        if self.save_shared and not (rank == 0 or rank == 1):
            self.save_shared = False

        if self.save_shared:
            self.folder_path = os.path.join(
                self.log_dir, "shared_params/{}".format(self.rank)
            )
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        self.model.shared_parameters_counter = torch.zeros(
            self.change_transformer(self.init_model).shape[0], dtype=torch.int32
        )

    def compress_data(self, data):
        result = dict(data)
        if self.compress:
            if "indices" in result:
                result["indices"] = self.compressor.compress(result["indices"])
            if "params" in result:
                result["params"] = self.compressor.compress_float(result["params"])
        return result

    def decompress_data(self, data):
        if self.compress:
            if "indices" in data:
                data["indices"] = self.compressor.decompress(data["indices"])
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        return data

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
        std, mean = torch.std_mean(G_topk, unbiased=False)
        self.std = std.item()
        self.mean = mean.item()
        _, index = torch.topk(
            G_topk, round(self.alpha * G_topk.shape[0]), dim=0, sorted=True
        )

        index, _ = torch.sort(index)
        return _, index

    def serialized_model(self):
        """
        Convert model to a dict. self.alpha specifies the fraction of model to send.

        Returns
        -------
        dict
            Model converted to a dict

        """
        if self.alpha >= self.metadata_cap:  # Share fully
            if self.model.accumulated_changes is not None:
                self.model.accumulated_changes = torch.zeros_like(
                    self.model.accumulated_changes
                )
            return super().serialized_model()

        with torch.no_grad():
            _, G_topk = self.extract_top_gradients()
            self.model.shared_parameters_counter[G_topk] += 1
            if self.accumulation:
                self.model.rewind_accumulation(G_topk)
            if self.save_shared:
                shared_params = dict()
                shared_params["order"] = list(self.model.state_dict().keys())
                shapes = dict()
                for k, v in self.model.state_dict().items():
                    shapes[k] = list(v.shape)
                shared_params["shapes"] = shapes

                shared_params[self.communication_round] = G_topk.tolist()

                with open(
                    os.path.join(
                        self.folder_path,
                        "{}_shared_params.json".format(self.communication_round + 1),
                    ),
                    "w",
                ) as of:
                    json.dump(shared_params, of)

            logging.debug("Extracting topk params")

            T_topk = self.pre_share_model[G_topk]

            logging.debug("Generating dictionary to send")

            m = dict()

            if not self.dict_ordered:
                raise NotImplementedError

            m["alpha"] = self.alpha

            m["indices"] = G_topk.numpy().astype(np.int32)

            m["params"] = T_topk.numpy()

            m["send_partial"] = True

            assert len(m["indices"]) == len(m["params"])
            logging.debug("Elements sending: {}".format(len(m["indices"])))

            logging.debug("Generated dictionary to send")

            logging.debug("Converted dictionary to pickle")

            return self.compress_data(m)

    def deserialized_model(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            dict received

        Returns
        -------
        state_dict
            state_dict of received

        """
        if "send_partial" not in m:
            return super().deserialized_model(m)

        with torch.no_grad():
            m = self.decompress_data(m)

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
            index_tensor = torch.tensor(m["indices"], dtype=torch.long)
            logging.debug("Original tensor: {}".format(T[index_tensor]))
            T[index_tensor] = torch.tensor(m["params"])
            logging.debug("Final tensor: {}".format(T[index_tensor]))
            start_index = 0
            for i, key in enumerate(state_dict):
                end_index = start_index + lens[i]
                state_dict[key] = T[start_index:end_index].reshape(shapes[i])
                start_index = end_index

            return state_dict

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        logging.debug("PartialModel _pre_step")
        with torch.no_grad():
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            self.pre_share_model = torch.cat(tensors_to_cat, dim=0)
            # Would only need one of the transforms
            self.pre_share_model_transformed = self.change_transformer(
                self.pre_share_model
            )
            change = self.change_transformer(self.pre_share_model - self.init_model)
            if self.accumulation:
                if not self.accumulate_averaging_changes:
                    # Need to accumulate in _pre_step as the accumulation gets rewind during the step
                    self.model.accumulated_changes += change
                    change = self.model.accumulated_changes.clone().detach()
                else:
                    # For the legacy implementation, we will only rewind currently accumulated values
                    # and add the model change due to averaging in the end
                    change += self.model.accumulated_changes
            # stores change of the model due to training, change due to averaging is not accounted
            self.model.model_change = change

    def _post_step(self):
        """
        Called at the end of step.

        """
        logging.debug("PartialModel _post_step")
        with torch.no_grad():
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            post_share_model = torch.cat(tensors_to_cat, dim=0)
            self.init_model = post_share_model
            if self.accumulation:
                if self.accumulate_averaging_changes:
                    self.model.accumulated_changes += self.change_transformer(
                        self.init_model - self.prev
                    )
                self.prev = self.init_model
            self.model.model_change = None
        if self.save_accumulated:
            self.save_change()

    def save_vector(self, v, s):
        """
        Saves the given vector to the file.

        Parameters
        ----------
        v : torch.tensor
            The torch tensor to write to file
        s : str
            Path to folder to write to

        """
        output_dict = dict()
        output_dict["order"] = list(self.model.state_dict().keys())
        shapes = dict()
        for k, v1 in self.model.state_dict().items():
            shapes[k] = list(v1.shape)
        output_dict["shapes"] = shapes

        output_dict["tensor"] = v.tolist()

        with open(
            os.path.join(s, "{}.json".format(self.communication_round + 1),), "w",
        ) as of:
            json.dump(output_dict, of)

    def save_change(self):
        """
        Saves the change and the gradient values for every iteration

        """
        self.save_vector(self.model.model_change, self.model_change_path)
