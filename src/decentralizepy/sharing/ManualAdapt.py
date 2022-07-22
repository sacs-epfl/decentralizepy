# Deprecated
import logging

from decentralizepy.sharing.PartialModel import PartialModel


class ManualAdapt(PartialModel):
    """
    This class implements the basic growing partial model sharing provided when and what alpha to set.

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
        change_alpha,
        change_rounds,
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
        change_alpha : list
            List of alphas to set. change_alpha[0] must be initial alpha.
        change_rounds : list
            List of iterations to change alpha. len(change_alpha) = len(change_rounds) + 1.
        dict_ordered : bool
            Specifies if the python dict maintains the order of insertion
        save_shared : bool
            Specifies if the indices of shared parameters should be logged
        metadata_cap : float
            Share full model when self.alpha > metadata_cap

        """
        assert change_alpha != ""
        assert change_alpha != None
        assert change_rounds != ""
        assert change_rounds != None

        if type(change_alpha) == str:
            change_alpha = eval(change_alpha)
        if type(change_rounds) == str:
            change_rounds = eval(change_rounds)

        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            change_alpha[0],
            dict_ordered,
            save_shared,
            metadata_cap,
            compress,
            compression_package,
            compression_class,
        )
        self.change_alpha = change_alpha[1:]
        self.change_rounds = change_rounds

    def get_data_to_send(self):
        """
        Perform a sharing step. Implements D-PSGD with alpha manually given.

        """
        if (
            len(self.change_rounds)
            and (self.communication_round + 1) == self.change_rounds[0]
        ):
            self.alpha = min(self.change_alpha[0], 1.00)
            self.change_alpha = self.change_alpha[1:]
            self.change_rounds = self.change_rounds[1:]

        if self.alpha == 0.0:
            logging.info("Not sending/receiving data (alpha=0.0)")
            self.communication_round += 1
            return dict()

        return super().get_data_to_send()
