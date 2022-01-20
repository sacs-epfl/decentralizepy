import logging

from decentralizepy.training.Training import Training


class GradientAccumulator(Training):
    def __init__(
        self,
        model,
        optimizer,
        loss,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
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
            True if the dataset should be shuffled before training.
        """
        super().__init__(
            model, optimizer, loss, rounds, full_epochs, batch_size, shuffle
        )

    def trainstep(self, data, target):
        """
        One training step on a minibatch.
        Parameters
        ----------
        data : any
            Data item
        target : any
            Label
        Returns
        -------
        int
            Loss Value for the step
        """
        self.model.zero_grad()
        output = self.model(data)
        loss_val = self.loss(output, target)
        loss_val.backward()
        logging.debug("Accumulating Gradients")
        self.model.accumulated_gradients.append(
            {
                k: v.grad.clone().detach()
                for k, v in zip(self.model.state_dict(), self.model.parameters())
            }
        )
        self.optimizer.step()
        return loss_val.item()

    def train_full(self, trainset):
        """
        One training iteration, goes through the entire dataset
        Parameters
        ----------
        trainset : torch.utils.data.Dataloader
            The training dataset.
        """
        for epoch in range(self.rounds):
            epoch_loss = 0.0
            count = 0
            for data, target in trainset:
                epoch_loss += self.trainstep(data, target)
                count += 1
            logging.info("Epoch: {} loss: {}".format(epoch, epoch_loss / count))

    def train(self, dataset):
        """
        One training iteration with accumulation of gradients in model.accumulated_gradients.
        Goes through the entire dataset.
        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)
        """
        self.model.accumulated_gradients = []
        super().train(dataset)
