import json
import logging
import os
from pathlib import Path

import torch

from decentralizepy.sharing.Sharing import Sharing


class SubSampling(Sharing):
    """
    This class implements the subsampling version of model sharing
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
        layerwise=False,
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
        pickle : bool
            use pickle to serialize the model parameters

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
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered
        self.save_shared = save_shared
        self.metadata_cap = metadata_cap

        # self.random_seed_generator = torch.Generator()
        # # Will use the random device if supported by CPU, else uses the system time
        # # In the latter case we could get duplicate seeds on some of the machines
        # self.random_seed_generator.seed()

        self.random_generator = torch.Generator()
        # Will use the random device if supported by CPU, else uses the system time
        # In the latter case we could get duplicate seeds on some of the machines
        self.random_generator.seed()
        self.seed = self.random_generator.initial_seed()

        self.pickle = pickle
        self.layerwise = layerwise

        logging.debug("subsampling pickling=" + str(pickle))

        if self.save_shared:
            # Only save for 2 procs: Save space
            if rank != 0 or rank != 1:
                self.save_shared = False

        if self.save_shared:
            self.folder_path = os.path.join(
                self.log_dir, "shared_params/{}".format(self.rank)
            )
            Path(self.folder_path).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            tensors_to_cat = []
            for _, v in self.model.state_dict().items():
                t = v.flatten()
                tensors_to_cat.append(t)
            self.init_model = torch.cat(tensors_to_cat, dim=0)

        self.model.shared_parameters_counter = torch.zeros(
            self.init_model.shape[0], dtype=torch.int32
        )

    def apply_subsampling(self):
        """
        Creates a random binary mask that is used to subsample the parameters that will be shared

        Returns
        -------
        tuple
            (a,b,c). a: the selected parameters as flat vector, b: the random seed used to crate the binary mask
                     c: the alpha

        """

        logging.debug("Returning subsampling gradients")
        if not self.layerwise:
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            concated = torch.cat(tensors_to_cat, dim=0)

            curr_seed = self.seed + self.communication_round  # is increased in step
            self.random_generator.manual_seed(curr_seed)
            # logging.debug("Subsampling seed for uid = " + str(self.uid) + " is: " + str(curr_seed))
            # Or we could use torch.bernoulli
            binary_mask = (
                torch.rand(
                    size=(concated.size(dim=0),), generator=self.random_generator
                )
                <= self.alpha
            )
            subsample = concated[binary_mask]
            self.model.shared_parameters_counter[binary_mask] += 1
            # logging.debug("Subsampling vector is of size: " + str(subsample.size(dim = 0)))
            return (subsample, curr_seed, self.alpha)
        else:
            values_list = []
            offsets = [0]
            off = 0
            curr_seed = self.seed + self.communication_round  # is increased in step
            self.random_generator.manual_seed(curr_seed)
            for _, v in self.model.state_dict().items():
                flat = v.flatten()
                binary_mask = (
                    torch.rand(
                        size=(flat.size(dim=0),), generator=self.random_generator
                    )
                    <= self.alpha
                )
                # TODO: support shared_parameters_counter
                selected = flat[binary_mask]
                values_list.append(selected)
                off += selected.size(dim=0)
                offsets.append(off)
            subsample = torch.cat(values_list, dim=0)
            return (subsample, curr_seed, self.alpha)

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
            subsample, seed, alpha = self.apply_subsampling()

            if self.save_shared:
                shared_params = dict()
                shared_params["order"] = list(self.model.state_dict().keys())
                shapes = dict()
                for k, v in self.model.state_dict().items():
                    shapes[k] = list(v.shape)
                shared_params["shapes"] = shapes

                # TODO: should store the shared indices and not the value
                # shared_params[self.communication_round] = subsample.tolist() # is slow

                shared_params["seed"] = seed

                shared_params["alpha"] = alpha

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

            m["seed"] = seed
            m["alpha"] = alpha
            m["params"] = subsample.numpy()

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
        if self.alpha > self.metadata_cap:  # Share fully
            return super().deserialized_model(m)

        m = self.decompress_data(m)

        with torch.no_grad():
            state_dict = self.model.state_dict()

            if not self.dict_ordered:
                raise NotImplementedError

            seed = m["seed"]
            alpha = m["alpha"]
            params = m["params"]

            random_generator = (
                torch.Generator()
            )  # new generator, such that we do not overwrite the other one
            random_generator.manual_seed(seed)

            shapes = []
            lens = []
            tensors_to_cat = []
            binary_submasks = []
            for _, v in state_dict.items():
                shapes.append(v.shape)
                t = v.flatten()
                lens.append(t.shape[0])
                tensors_to_cat.append(t)
                if self.layerwise:
                    binary_mask = (
                        torch.rand(size=(t.size(dim=0),), generator=random_generator)
                        <= alpha
                    )
                    binary_submasks.append(binary_mask)

            T = torch.cat(tensors_to_cat, dim=0)

            params_tensor = torch.from_numpy(params)

            if not self.layerwise:
                binary_mask = (
                    torch.rand(size=(T.size(dim=0),), generator=random_generator)
                    <= alpha
                )
            else:
                binary_mask = torch.cat(binary_submasks, dim=0)

            logging.debug("Original tensor: {}".format(T[binary_mask]))
            T[binary_mask] = params_tensor
            logging.debug("Final tensor: {}".format(T[binary_mask]))

            start_index = 0
            for i, key in enumerate(state_dict):
                end_index = start_index + lens[i]
                state_dict[key] = T[start_index:end_index].reshape(shapes[i])
                start_index = end_index

            return state_dict
