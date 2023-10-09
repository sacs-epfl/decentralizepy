import logging

import torch

from decentralizepy import utils
from decentralizepy.training.Training import Training


class LLMTraining(Training):
    """
    This class implements the training module for a single node.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss=None,
        log_dir=".",
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
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
        full_epochs : bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.

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
            for batch in trainset:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs[0]
                epoch_loss += loss.item()
                count += 1
        loss = epoch_loss / count
        logging.info("Loss after iteration: {}".format(loss))
        return loss

    def trainstep(self, batch):
        """
        One training step on a minibatch.

        Parameters
        ----------
        batch : any
            Data item

        Returns
        -------
        int
            Loss Value for the step

        """
        self.optimizer.zero_grad()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_full(self, dataset):
        """
        One training iteration, goes through the entire dataset

        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.

        """
        trainset = dataset.get_trainset(self.batch_size, self.shuffle)
        for epoch in range(self.rounds):
            epoch_loss = 0.0
            count = 0
            for batch in trainset:
                logging.debug(
                    "Starting minibatch {} with num_samples: {}".format(
                        count, len(batch["input_ids"])
                    )
                )
                epoch_loss += self.trainstep(batch)
                count += 1
            logging.debug("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train(self, dataset):
        """
        One training iteration

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.train()

        if self.full_epochs:
            self.train_full(dataset)
        else:
            iter_loss = 0.0
            count = 0
            trainset = dataset.get_trainset(self.batch_size, self.shuffle)
            while count < self.rounds:
                for data in trainset:
                    iter_loss += self.trainstep(data)
                    count += 1
                    logging.debug("Round: {} loss: {}".format(count, iter_loss / count))
                    if count >= self.rounds:
                        break
