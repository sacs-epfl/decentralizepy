import logging

import torch
from torch import fft

from decentralizepy.training.Training import Training


class FrequencyAccumulator(Training):
    """
    This class implements the training module which also accumulates the fft frequency at the beginning of steps a communication round.

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
        self.init_model = None
        self.prev = None

    def train(self, dataset):
        """
        Does one training iteration.
        If self.accumulation is True then it accumulates model fft frequency changes in model.accumulated_frequency.
        Otherwise it stores the current fft frequency representation of the model in model.accumulated_frequency.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        with torch.no_grad():
            self.model.accumulated_gradients = []
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            concated = torch.cat(tensors_to_cat, dim=0)
            self.init_model = fft.rfft(concated)
            if self.accumulation:
                if self.model.accumulated_changes is None:
                    self.model.accumulated_changes = torch.zeros_like(self.init_model)
                    self.prev = self.init_model
                else:
                    self.model.accumulated_changes += self.init_model - self.prev
                    self.prev = self.init_model

        super().train(dataset)

        with torch.no_grad():
            tensors_to_cat = [
                v.data.flatten() for _, v in self.model.state_dict().items()
            ]
            concated = torch.cat(tensors_to_cat, dim=0)
            end_model = fft.rfft(concated)
            change = end_model - self.init_model
            if self.accumulation:
                change += self.model.accumulated_changes

            self.model.accumulated_gradients.append(change)
