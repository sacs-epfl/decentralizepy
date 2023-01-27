import logging

import numpy as np
import torch

from decentralizepy.sharing.PartialModel import PartialModel


class TopKPlusRandom(PartialModel):
    """
    This class implements partial model sharing with some random additions.

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
            compress,
            compression_package,
            compression_class,
        )

    def extract_top_gradients(self):
        """
        Extract the indices and values of the topK gradients and put some extra random.
        The gradients must have been accumulated.

        Returns
        -------
        tuple
            (a,b). a: The magnitudes of the topK gradients, b: Their indices.

        """
        logging.debug("Returning topk gradients")
        G = torch.abs(self.model.model_change)
        std, mean = torch.std_mean(G, unbiased=False)
        self.std = std.item()
        self.mean = mean.item()
        elements_to_pick = round(self.alpha / 2.0 * G.shape[0])
        G_topK = torch.topk(G, min(G.shape[0], elements_to_pick), dim=0, sorted=False)
        more_indices = np.arange(G.shape[0], dtype=int)
        np.delete(more_indices, G_topK[1].numpy())
        more_indices = np.random.choice(
            more_indices, min(more_indices.shape[0], elements_to_pick)
        )
        G_topK0 = torch.cat([G_topK[0], G[more_indices]], dim=0)
        G_topK1 = torch.cat([G_topK[1], torch.tensor(more_indices)], dim=0)
        return G_topK0, G_topK1
