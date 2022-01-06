import logging

import torch

from decentralizepy import utils


class Training:
    """
    This class implements the training module for a single node.
    """

    def __init__(
        self, model, optimizer, loss, epochs_per_round="", batch_size="", shuffle=""
    ):
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
        shuffle : bool
            True if the dataset should be shuffled before training. Not implemented yet! TODO
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs_per_round = utils.conditional_value(epochs_per_round, "", int(1))
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def reset_optimizer(self, optimizer):
        """
        Replace the current optimizer with a new one
        Parameters
        ----------
        optimizer : torch.optim
            A new optimizer
        """
        self.optimizer = optimizer

    def eval_loss(self, dataset):
        """
        Evaluate the loss on the training set
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        epoch_loss = 0.0
        count = 0
        with torch.no_grad():
            for data, target in trainset:
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def train(self, dataset):
        """
        One training iteration, goes through the entire dataset
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)

        for epoch in range(self.epochs_per_round):
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                self.model.zero_grad()
                output = self.model(data)
                loss_val = self.loss(output, target)
                epoch_loss += loss_val.item()
                loss_val.backward()
                self.optimizer.step()
                count += 1
            logging.info("Epoch: {} loss: {}".format(epoch, epoch_loss / count))
