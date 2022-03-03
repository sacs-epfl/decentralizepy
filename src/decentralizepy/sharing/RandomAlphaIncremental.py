import random

from decentralizepy.sharing.PartialModel import PartialModel


class RandomAlphaIncremental(PartialModel):
    """
    This class implements the partial model sharing with a random alpha from an increasing range each iteration.

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
        dict_ordered=True,
        save_shared=False,
        metadata_cap=1.0,
        range_start=0.1,
        range_end=0.2,
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
            1.0,
            dict_ordered,
            save_shared,
            metadata_cap,
        )
        random.seed(self.mapping.get_uid(self.rank, self.machine_id))
        self.range_start = range_start
        self.range_end = range_end

    def step(self):
        """
        Perform a sharing step. Implements D-PSGD with alpha randomly chosen from an increasing range.

        """
        self.alpha = round(random.random(self.range_start, self.range_end), 2)
        self.range_end = min(1.0, self.range_end + round(random.random(0.0, 0.1), 2))
        super().step()
