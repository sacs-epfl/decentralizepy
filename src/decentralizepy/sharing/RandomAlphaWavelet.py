import random

from decentralizepy.sharing.Wavelet import Wavelet


class RandomAlpha(Wavelet):
    """
    This class implements the partial model sharing with a random alpha each iteration.

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
        alpha_list="[0.1, 0.2, 0.3, 0.4, 1.0]",
        dict_ordered=True,
        save_shared=False,
        metadata_cap=1.0,
        wavelet="haar",
        level=4,
        change_based_selection=True,
        save_accumulated="",
        accumulation=False,
        accumulate_averaging_changes=False,
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
            wavelet,
            level,
            change_based_selection,
            save_accumulated,
            accumulation,
            accumulate_averaging_changes,
        )
        self.alpha_list = eval(alpha_list)
        random.seed(self.mapping.get_uid(self.rank, self.machine_id))

    def step(self):
        """
        Perform a sharing step. Implements D-PSGD with alpha randomly chosen.

        """
        self.alpha = random.choice(self.alpha_list)
        super().step()
