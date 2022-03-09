import json
import logging
import os
from pathlib import Path

import torch

from decentralizepy.sharing.Sharing import Sharing


class TopKParams(Sharing):
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

        """
        super().__init__(
            rank, machine_id, communication, mapping, graph, model, dataset, log_dir
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered
        self.save_shared = save_shared
        self.metadata_cap = metadata_cap
        self.total_meta = 0

        if self.save_shared:
            # Only save for 2 procs: Save space
            if rank != 0 or rank != 1:
                self.save_shared = False

        if self.save_shared:
            self.folder_path = os.path.join(
                self.log_dir, "shared_params/{}".format(self.rank)
            )
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)

    def extract_top_params(self):
        """
        Extract the indices and values of the topK params layerwise.
        The gradients must have been accumulated.

        Returns
        -------
        tuple
            (a,b,c). a: The topK params, b: Their indices, c: The offsets

        """

        logging.info("Returning TopKParams gradients")
        values_list = []
        index_list = []
        offsets = [0]
        off = 0
        for _, v in self.model.state_dict().items():
            flat = v.flatten()
            values, index = torch.topk(
                flat.abs(), round(self.alpha * flat.size(dim=0)), dim=0, sorted=False
            )
            values_list.append(flat[index])
            index_list.append(index)
            off += values.size(dim=0)
            offsets.append(off)
        cat_values = torch.cat(values_list, dim=0)
        cat_index = torch.cat(index_list, dim=0)

        # logging.debug("Subsampling vector is of size: " + str(subsample.size(dim = 0)))
        return (cat_values, cat_index, offsets)

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
            values, index, offsets = self.extract_top_params()

            if self.save_shared:
                shared_params = dict()
                shared_params["order"] = list(self.model.state_dict().keys())
                shapes = dict()
                for k, v in self.model.state_dict().items():
                    shapes[k] = list(v.shape)
                shared_params["shapes"] = shapes

                shared_params[self.communication_round] = index.tolist()
                # TODO: store offsets

                with open(
                    os.path.join(
                        self.folder_path,
                        "{}_shared_params.json".format(self.communication_round + 1),
                    ),
                    "w",
                ) as of:
                    json.dump(shared_params, of)

            logging.info("Extracting topk params")

            logging.info("Generating dictionary to send")

            m = dict()

            if not self.dict_ordered:
                raise NotImplementedError

            m["indices"] = index.numpy()
            m["params"] = values.numpy()
            m["offsets"] = offsets

            assert len(m["indices"]) == len(m["params"])
            logging.info("Elements sending: {}".format(len(m["indices"])))

            logging.info("Generated dictionary to send")

            # for key in m:
            #    m[key] = json.dumps(m[key])

            logging.info("Converted dictionary to json")
            self.total_data += len(self.communication.encrypt(m["params"]))
            self.total_meta += len(self.communication.encrypt(m["indices"])) + len(
                self.communication.encrypt(m["offsets"])
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
            offsets = m["offsets"]
            params = torch.tensor(m["params"])
            indices = torch.tensor(m["indices"])

            for i, (_, v) in enumerate(state_dict.items()):
                shapes.append(v.shape)
                t = v.flatten().clone().detach()  # it is not always copied
                lens.append(t.shape[0])
                index = indices[offsets[i] : offsets[i + 1]]
                t[index] = params[offsets[i] : offsets[i + 1]]
                tensors_to_cat.append(t)

            start_index = 0
            for i, key in enumerate(state_dict):
                end_index = start_index + lens[i]
                state_dict[key] = tensors_to_cat[i].reshape(shapes[i])
                start_index = end_index

            return state_dict
