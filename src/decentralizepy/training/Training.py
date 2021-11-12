import torch
from decentralizepy import utils
import logging
class Training:
    """
    This class implements the training module for a single node.
    """
    def __init__(self, model, optimizer, loss, epochs_per_round = "", batch_size = "", shuffle = ""):
        """
        Constructor
        Parameters
        ----------
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        epochs_per_round : int, optional
            Number of epochs per training call
        batch_size : int, optional
            Number of items to learn over, in one batch
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs_per_round = utils.conditional_value(epochs_per_round, "", 1)
        self.batch_size = utils.conditional_value(batch_size, "", 1)
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def train(self, dataset):
        """
        One training iteration
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        for epoch in range(self.epochs_per_round):
            epoch_loss = 0.0
            for data, target in trainset:
                self.model.zero_grad()
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                loss_val.backward()
                self.optimizer.step()
            logging.info("Epoch_loss: %d", epoch_loss)
