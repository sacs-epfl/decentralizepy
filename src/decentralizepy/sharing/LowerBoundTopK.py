import logging

import numpy as np
import torch

from decentralizepy.sharing.PartialModel import PartialModel


class LowerBoundTopK(PartialModel):
    """
    This class implements a bounded version of topK.

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
        lower_bound=0.1,
        metro_hastings=True,
        compress=False,
        compression_package=None,
        compression_class=None,
        **kwargs,
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
        lower_bound : float
            Increases the communication budget per communication round by lower_bound, i.e. the new effective
            alpha will be alpha + alpha*lower_bound. The extra budget is used to randomly selected parameters that
            were shared in less than alpha*lower_bound*100 percentage of the rounds.
        metro_hastings: bool
            If True then metro hastings averaging is used otherwise it does per parameter averaging.

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
            compression_class**kwargs,
        )
        self.lower_bound = lower_bound
        self.metro_hastings = metro_hastings
        if self.lower_bound > 0:
            self.start_lower_bounding_at = 1 / self.lower_bound

    def extract_top_gradients(self):
        """
        Extract the indices and values of the topK gradients.
        The gradients must have been accumulated.

        Returns
        -------
        tuple
            (a,b). a: The magnitudes of the topK gradients, b: Their indices.

        """
        if self.lower_bound == 0.0:
            return super().extract_top_gradients()

        logging.info("Returning topk gradients bounded")
        G_topk = torch.abs(self.model.model_change)
        std, mean = torch.std_mean(G_topk, unbiased=False)
        self.std = std.item()
        self.mean = mean.item()

        val, ind = torch.topk(
            G_topk, round(self.alpha * G_topk.shape[0]), dim=0, sorted=False
        )
        if self.communication_round > self.start_lower_bounding_at:
            # because the superclass increases it where it is inconvenient for this subclass
            currently_shared = self.model.shared_parameters_counter.clone().detach()
            currently_shared[ind] += 1
            ind_small = (
                currently_shared < self.communication_round * self.lower_bound
            ).nonzero(as_tuple=True)[0]
            ind_small_unique = np.setdiff1d(
                ind_small.numpy(), ind.numpy(), assume_unique=True
            )
            take_max = round(self.lower_bound * self.alpha * G_topk.shape[0])
            logging.info(
                "lower: %i %i %i", len(ind_small), len(ind_small_unique), take_max
            )
            if take_max > ind_small_unique.shape[0]:
                take_max = ind_small_unique.shape[0]
            to_take = torch.rand(ind_small_unique.shape[0])
            _, ind_of_to_take = torch.topk(to_take, take_max, dim=0, sorted=False)
            ind_bound = torch.from_numpy(ind_small_unique)[ind_of_to_take]
            logging.info("lower bounding: %i %i", len(ind), len(ind_bound))
            # val = torch.concat(val, G_topk[ind_bound]) # not really needed, as thes are abs values and not further used
            ind = torch.cat([ind, ind_bound])

        return val, ind

    def deserialized_model_avg(self, m):
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

        m = self.decompress_data(m)

        with torch.no_grad():
            state_dict = self.model.state_dict()

            if not self.dict_ordered:
                raise NotImplementedError

            # could be made more efficent
            T = torch.zeros_like(self.init_model)
            index_tensor = torch.tensor(m["indices"], dtype=torch.long)
            logging.debug("Original tensor: {}".format(T[index_tensor]))
            T[index_tensor] = torch.tensor(m["params"])
            logging.debug("Final tensor: {}".format(T[index_tensor]))

            return T, index_tensor

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        if self.metro_hastings:
            super()._averaging()
        else:
            with torch.no_grad():

                tensors_to_cat = []
                for _, v in self.model.state_dict().items():
                    t = v.flatten()
                    tensors_to_cat.append(t)
                T = torch.cat(tensors_to_cat, dim=0)
                weight_total = 0
                weight_vector = torch.ones_like(self.init_model)
                datas = []
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
                    data, ind = self.deserialized_model_avg(data)
                    weight_vector[ind] += 1
                    # weight = 1 / (max(len(self.peer_deques), degree) + 1)  # Metro-Hastings
                    # weight_total += weight
                    datas.append(data)

                weight_vector = 1.0 / weight_vector
                # speed up by exploiting sparsity
                T = T * weight_vector
                for d in datas:
                    T += d * weight_vector

                start_index = 0
                total = dict()
                for i, key in enumerate(self.model.state_dict()):
                    end_index = start_index + self.lens[i]
                    total[key] = T[start_index:end_index].reshape(self.shapes[i])
                    start_index = end_index

            logging.info("new averaging")
            self.model.load_state_dict(total)
            self._post_step()
            self.communication_round += 1
