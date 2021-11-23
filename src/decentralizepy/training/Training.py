import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

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
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.epochs_per_round = utils.conditional_value(epochs_per_round, "", int(1))
        self.batch_size = utils.conditional_value(batch_size, "", int(1))
        self.shuffle = utils.conditional_value(shuffle, "", False)

    def imshow(self, img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def reset_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, dataset):
        """
        One training iteration
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)

        # dataiter = iter(trainset)
        # images, labels = dataiter.next()
        # self.imshow(torchvision.utils.make_grid(images[:16]))
        # plt.savefig(' '.join('%5s' % j for j in labels) + ".png")
        # print(' '.join('%5s' % j for j in labels[:16]))

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
