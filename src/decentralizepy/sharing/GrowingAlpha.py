import logging

from decentralizepy.sharing.PartialModel import PartialModel


class GrowingAlpha(PartialModel):
    """
    This class implements the basic growing partial model sharing using a linear function.

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
        init_alpha=0.0,
        max_alpha=1.0,
        k=10,
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
        init_alpha : float
            Percentage of model to share initially
        max_alpha : float
            Maximum alpha to reach in k steps
        k : int
            Steps to reach maximum alpha. Also steps after which alpha increases.
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
            init_alpha,
            dict_ordered,
            save_shared,
            metadata_cap,
        )
        self.init_alpha = init_alpha
        self.max_alpha = max_alpha
        self.k = k

    def step(self):
        """
        Perform a sharing step. Implements D-PSGD with alpha increasing as a linear function.

        """
        if (self.communication_round + 1) % self.k == 0:
            self.alpha += (self.max_alpha - self.init_alpha) / self.k
            self.alpha = min(self.alpha, 1.00)

        if self.alpha == 0.0:
            logging.info("Not sending/receiving data (alpha=0.0)")
            self.communication_round += 1
            return

        super().step()
