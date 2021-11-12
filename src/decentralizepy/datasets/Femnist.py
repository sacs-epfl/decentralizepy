import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner

NUM_CLASSES = 62
IMAGE_SIZE = (28, 28)
FLAT_SIZE = 28 * 28
PIXEL_RANGE = 256.0


class Femnist(Dataset):
    """
    Class for the FEMNIST dataset
    """

    def __read_dir__(self, data_dir):
        """
        Function to read all the FEMNIST data files in the directory
        Parameters
        ----------
        data_dir : str
            Path to the folder containing the data files
        Returns
        -------
        3-tuple
            A tuple containing list of clients, number of samples per client,
            and the data items per client
        """
        clients = []
        num_samples = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, "r") as inf:
                client_data = json.load(inf)
            clients.extend(client_data["users"])
            num_samples.extend(client_data["num_samples"])
            data.update(client_data["user_data"])

        return clients, num_samples, data

    def __init__(self, rank, n_procs="", train_dir="", test_dir="", sizes=""):
        """
        Constructor which reads the data files, instantiates and partitions the dataset
        Parameters
        ----------
        rank : int, optional
            Rank of the current process (to get the partition). Default value is assigned 0
        n_procs : int, optional
            The number of processes among which to divide the data. Default value is assigned 1
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to n_procs and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        """
        super().__init__(rank, n_procs, train_dir, test_dir, sizes)

        if self.__training__:
            logging.info("Loading training set.")
            clients, num_samples, train_data = self.__read_dir__(self.train_dir)
            c_len = len(clients)

            if self.sizes == None:  # Equal distribution of data among processes
                e = c_len // self.n_procs
                frac = e / c_len
                self.sizes = [frac] * self.n_procs
                self.sizes[-1] += 1.0 - frac * self.n_procs
                logging.debug("Size fractions: {}".format(sizes))

            my_clients = DataPartitioner(clients, self.sizes).use(self.rank)
            my_train_data = {"x": [], "y": []}
            self.clients = []
            self.num_samples = []
            logging.debug("Clients Length: %d", c_len)
            logging.debug("My_clients_len: %d", my_clients.__len__())
            for i in range(my_clients.__len__()):
                cur_client = my_clients.__getitem__(i)
                self.clients.append(cur_client)
                my_train_data["x"].extend(train_data[cur_client]["x"])
                my_train_data["y"].extend(train_data[cur_client]["y"])
                self.num_samples.append(len(train_data[cur_client]["y"]))
            self.train_x = (
                np.array(my_train_data["x"], dtype=np.dtype("float32"))
                .reshape(-1, 28, 28, 1)
                .transpose(0, 3, 1, 2)
            )
            self.train_y = np.array(
                my_train_data["y"], dtype=np.dtype("int64")
            ).reshape(-1)
            logging.debug("train_x.shape: %s", str(self.train_x.shape))
            logging.debug("train_y.shape: %s", str(self.train_y.shape))

        if self.__testing__:
            logging.info("Loading training set.")
            _, _, test_data = self.__read_dir__(self.test_dir)
            test_data = test_data.values()
            self.test_x = (
                np.array(test_data["x"], dtype=np.dtype("float32"))
                .reshape(-1, 28, 28, 1)
                .transpose(0, 3, 1, 2)
            )
            self.test_y = np.array(test_data["y"], dtype=np.dtype("int64")).reshape(
                -1
            )
            logging.debug("test_x.shape: %s", str(self.test_x.shape))
            logging.debug("test_y.shape: %s", str(self.test_y.shape))

        # TODO: Add Validation

    def get_client_ids(self):
        """
        Function to retrieve all the clients of the current process
        Returns
        -------
        list(str)
            A list of strings of the client ids.
        """
        return self.clients

    def get_client_id(self, i):
        """
        Function to get the client id of the ith sample
        Parameters
        ----------
        i : int
            Index of the sample
        Returns
        -------
        str
            Client ID
        Raises
        ------
        IndexError
            If the sample index is out of bounds
        """
        lb = 0
        for j in range(len(self.clients)):
            if i < lb + self.num_samples[j]:
                return self.clients[j]

        raise IndexError("i is out of bounds!")

    def get_trainset(self, batch_size, shuffle = False):
        """
        Function to get the training set
        Parameters
        ----------
        batch_size : int
            Batch size for learning
        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)
        Raises
        ------
        RuntimeError
            If the training set was not initialized
        """
        if self.__training__:
            return DataLoader(Data(self.train_x, self.train_y), batch_size = batch_size, shuffle = shuffle)
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set
        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)
        Raises
        ------
        RuntimeError
            If the test set was not initialized
        """
        if self.__testing__:
            return Data(self.test_x, self.test_y)
        raise RuntimeError("Test set not initialized!")


class LogisticRegression(nn.Module):
    """
    Class for a Logistic Regression Neural Network for FEMNIST
    """

    def __init__(self):
        """
        Constructor. Instantiates the Logistic Regression Model
            with 28*28 Input and 62 output classes
        """
        super().__init__()
        self.fc1 = nn.Linear(FLAT_SIZE, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model
        Parameters
        ----------
        x : torch.tensor
            The input torch tensor
        Returns
        -------
        torch.tensor
            The output torch tensor
        """
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
