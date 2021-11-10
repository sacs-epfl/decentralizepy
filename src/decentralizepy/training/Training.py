from torch.optim import SGD

from decentralizepy import utils
class Training:
    """
    This class implements the training module for a single node.
    """
    def __init__(self, model, optimizer, epochs_per_round = "", batch_size = ""):
        """
        Constructor
        Parameters
        ----------
        epochs_per_round : int
            Number of epochs per training call
        batch_size : int
            Number of items to learn over, in one batch
        """
        self.epochs_per_round = utils.conditional_value(epochs_per_round, "", 1)
        self.batch_size = utils.conditional_value(batch_size, "", 1)

    def train(self, trainset):
        """
        One training iteration
        Parameters
        ----------
        trainset : decentralizepy.datasets.Data
            The training dataset. Should implement __getitem__(i)
        """
        raise NotImplementedError
