import logging

import torch

from decentralizepy.sharing.PartialModel import PartialModel
from decentralizepy.utils import identity


class TopKNormalized(PartialModel):
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
        epsilon=0.01,
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
        epsilon : float
            numerical stability parameter used during normalization

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
            change_transformer,
            accumulate_averaging_changes,
        )
        self.epsilon = epsilon

    def extract_top_gradients(self):
        """
        Extract the indices and values of the topK gradients.
        The gradients must have been accumulated.

        Returns
        -------
        tuple
            (a,b). a: The magnitudes of the topK gradients, b: Their indices.

        """

        logging.info("Returning topk gradients")
        G_topk = torch.abs(self.model.model_change)
        G_topk_normalized = G_topk / (torch.abs(self.pre_share_model) + self.epsilon)
        std, mean = torch.std_mean(G_topk, unbiased=False)
        self.std = std.item()
        self.mean = mean.item()
        return torch.topk(
            G_topk_normalized, round(self.alpha * G_topk.shape[0]), dim=0, sorted=False
        )
