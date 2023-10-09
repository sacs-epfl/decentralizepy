import torch

from decentralizepy.datasets.Data import Data


class LLMData(Data):
    """
    This class defines the API for Data.

    """

    def __init__(self, x, y):
        """
        Constructor

        Parameters
        ----------
        x : numpy array
            A numpy array of data samples
        y : numpy array
            A numpy array of outputs corresponding to the sample

        """
        self.x = x
        self.y = y

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns
        -------
        int
            Number of samples

        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Function to get the item with index i.

        Parameters
        ----------
        idx : int
            Index

        Returns
        -------
        dict
            A dict of the ith data sample, its attention_mask and label

        """
        item = {key: torch.tensor(val[idx]) for key, val in self.x.items()}
        item["labels"] = torch.tensor(self.y[idx])
        return item
