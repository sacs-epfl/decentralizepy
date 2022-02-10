import logging

from decentralizepy.training.Training import Training


class GradientAccumulator(Training):
    """
    This class implements the training module which also accumulates gradients of steps in a list.

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
