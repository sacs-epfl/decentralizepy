import logging

import torch
from torch import fft

from decentralizepy.training.Training import Training


class ModelChangeAccumulator(Training):
    """
    This class implements the training module which also accumulates the model change at the beginning of a communication round.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        accumulation=True,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs: bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.
        accumulation : bool
            True if the model change should be accumulated across communication steps

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            model,
            optimizer,
            loss,
            log_dir,
            rounds,
            full_epochs,
            batch_size,
            shuffle,
        )
        self.accumulation = accumulation

    def train(self, dataset):
        """
        Does one training iteration.
        If self.accumulation is True then it accumulates model parameter changes in model.prev_model_params.
        Otherwise it stores the current model parameters in model.prev_model_params.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """

        tensors_to_cat = [v.data.flatten() for _, v in self.model.state_dict().items()]
        concated = torch.cat(tensors_to_cat, dim=0)
        if self.accumulation:
            if self.model.prev_model_params is None:
                logging.info("Initialize model parameter accumulation.")
                self.model.prev_model_params = torch.zeros_like(concated)
                self.model.prev = concated
            else:
                logging.info("model parameter accumulation step")
                self.model.prev_model_params += concated - self.model.prev
                self.model.prev = concated
        else:
            logging.info("model parameter reset")
            self.model.prev_model_params = concated
        super().train(dataset)
