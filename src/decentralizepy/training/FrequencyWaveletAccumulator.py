import logging

import numpy as np
import pywt
import torch

from decentralizepy.training.Training import Training


class FrequencyWaveletAccumulator(Training):
    """
    This class implements the training module which also accumulates the wavelet frequency at the beginning of steps a communication round.

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
        wavelet="haar",
        level=4,
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
        self.wavelet = wavelet
        self.level = level
        self.accumulation = accumulation

    def train(self, dataset):
        """
        Does one training iteration.
        If self.accumulation is True then it accumulates model wavelet frequency changes in model.accumulated_frequency.
        Otherwise it stores the current wavelet frequency representation of the model in model.accumulated_frequency.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """

        # this looks at the change from the last round averaging of the frequencies
        tensors_to_cat = [v.data.flatten() for _, v in self.model.state_dict().items()]
        concated = torch.cat(tensors_to_cat, dim=0)
        coeff = pywt.wavedec(concated.numpy(), self.wavelet, level=self.level)
        data, coeff_slices = pywt.coeffs_to_array(coeff)
        data = data.ravel()
        if self.accumulation:
            if self.model.accumulated_frequency is None:
                logging.info("Initialize wavelet frequency accumulation")
                self.model.accumulated_frequency = np.zeros_like(
                    data
                )  # torch.zeros_like(data)
                self.model.prev = data
            else:
                logging.info("wavelet frequency accumulation step")
                self.model.accumulated_frequency += data - self.model.prev
                self.model.prev = data
        else:
            logging.info("wavelet frequency accumulation reset")
            self.model.accumulated_frequency = data
        super().train(dataset)
