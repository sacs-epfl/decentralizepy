import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import decentralizepy.utils as utils
from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

IMAGE_DIM = 84
CHANNELS = 3
NUM_CLASSES = 2


class Celeba(Dataset):
    """
    Class for the Celeba dataset

    """

    def __read_file__(self, file_path):
        """
        Read data from the given json file

        Parameters
        ----------
        file_path : str
            The file path

        Returns
        -------
        tuple
            (users, num_samples, data)

        """
        with open(file_path, "r") as inf:
            client_data = json.load(inf)
        return (
            client_data["users"],
            client_data["num_samples"],
            client_data["user_data"],
        )

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
            u, n, d = self.__read_file__(file_path)
            clients.extend(u)
            num_samples.extend(n)
            data.update(d)
        return clients, num_samples, data

    def file_per_user(self, dir, write_dir):
        """
        Function to read all the FEMNIST data files and write one file per user

        Parameters
        ----------
        dir : str
            Path to the folder containing the data files
        write_dir : str
            Path to the folder to write the files

        """
        clients, num_samples, train_data = self.__read_dir__(dir)
        for index, client in enumerate(clients):
            my_data = dict()
            my_data["users"] = [client]
            my_data["num_samples"] = num_samples[index]
            my_samples = {"x": train_data[client]["x"], "y": train_data[client]["y"]}
            my_data["user_data"] = {client: my_samples}
            with open(os.path.join(write_dir, client + ".json"), "w") as of:
                json.dump(my_data, of)
                print("Created File: ", client + ".json")

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        files = os.listdir(self.train_dir)
        files = [f for f in files if f.endswith(".json")]
        files.sort()
        c_len = len(files)

        # clients, num_samples, train_data = self.__read_dir__(self.train_dir)

        if self.sizes == None:  # Equal distribution of data among processes
            e = c_len // self.n_procs
            frac = e / c_len
            self.sizes = [frac] * self.n_procs
            self.sizes[-1] += 1.0 - frac * self.n_procs
            logging.debug("Size fractions: {}".format(self.sizes))

        self.uid = self.mapping.get_uid(self.rank, self.machine_id)
        my_clients = DataPartitioner(files, self.sizes).use(self.uid)
        my_train_data = {"x": [], "y": []}
        self.clients = []
        self.num_samples = []
        logging.debug("Clients Length: %d", c_len)
        logging.debug("My_clients_len: %d", my_clients.__len__())
        for i in range(my_clients.__len__()):
            cur_file = my_clients.__getitem__(i)

            clients, _, train_data = self.__read_file__(
                os.path.join(self.train_dir, cur_file)
            )
            for cur_client in clients:
                logging.debug("Got data of client: {}".format(cur_client))
                self.clients.append(cur_client)
                my_train_data["x"].extend(self.process_x(train_data[cur_client]["x"]))
                my_train_data["y"].extend(train_data[cur_client]["y"])
                self.num_samples.append(len(train_data[cur_client]["y"]))

        logging.debug(
            "Initial shape of x: {}".format(
                np.array(my_train_data["x"], dtype=np.dtype("float32")).shape
            )
        )
        self.train_x = (
            np.array(my_train_data["x"], dtype=np.dtype("float32"))
            .reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
            .transpose(0, 3, 1, 2)  # Channel first: torch
        )
        self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        logging.info("train_x.shape: %s", str(self.train_x.shape))
        logging.info("train_y.shape: %s", str(self.train_y.shape))
        assert self.train_x.shape[0] == self.train_y.shape[0]
        assert self.train_x.shape[0] > 0

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            test_x.extend(self.process_x(test_data["x"]))
            test_y.extend(test_data["y"])
        self.test_x = (
            np.array(test_x, dtype=np.dtype("float32"))
            .reshape(-1, IMAGE_DIM, IMAGE_DIM, CHANNELS)
            .transpose(0, 3, 1, 2)
        )
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        logging.info("test_x.shape: %s", str(self.test_x.shape))
        logging.info("test_y.shape: %s", str(self.test_y.shape))
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        train_dir="",
        test_dir="",
        images_dir="",
        sizes="",
        test_batch_size=128,
    ):
        """
        Constructor which reads the data files, instantiates and partitions the dataset

        Parameters
        ----------
        rank : int
            Rank of the current process (to get the partition).
        machine_id : int
            Machine ID
        mapping : decentralizepy.mappings.Mapping
            Mapping to convert rank, machine_id -> uid for data partitioning
            It also provides the total number of global processes
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str. optional
            Path to the testing data files Required to instantiate the testing set
        sizes : list(int), optional
            A list of fractions specifying how much data to alot each process. Sum of fractions should be 1.0
            By default, each process gets an equal amount.
        test_batch_size : int, optional
            Batch size during testing. Default value is 64

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )
        self.IMAGES_DIR = utils.conditional_value(images_dir, "", None)
        assert self.IMAGES_DIR != None

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

        # TODO: Add Validation

    def process_x(self, raw_x_batch):
        """
        Preprocesses the whole batch of images

        Returns
        -------
        np.array
            The images as a numpy array

        """
        x_batch = [self._load_image(i) for i in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def _load_image(self, img_name):
        """
        Open and load image.

        Returns
        -------
        np.array
            The image as a numpy array

        """
        img = Image.open(os.path.join(self.IMAGES_DIR, img_name[:-4] + ".png"))
        img = img.resize((IMAGE_DIM, IMAGE_DIM)).convert("RGB")
        return np.array(img)

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

    def get_trainset(self, batch_size=1, shuffle=False):
        """
        Function to get the training set

        Parameters
        ----------
        batch_size : int, optional
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
            return DataLoader(
                Data(self.train_x, self.train_y), batch_size=batch_size, shuffle=shuffle
            )
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
            return DataLoader(
                Data(self.test_x, self.test_y), batch_size=self.test_batch_size
            )
        raise RuntimeError("Test set not initialized!")

    def test(self, model, loss):
        """
        Function to evaluate model on the test dataset.

        Parameters
        ----------
        model : decentralizepy.models.Model
            Model to evaluate
        loss : torch.nn.loss
            Loss function to evaluate

        Returns
        -------
        tuple
            (accuracy, loss_value)

        """
        testloader = self.get_testset()

        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(NUM_CLASSES)]
        total_pred = [0 for _ in range(NUM_CLASSES)]

        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for elems, labels in testloader:
                outputs = model(elems)
                loss_val += loss(outputs, labels).item()
                count += 1
                _, predictions = torch.max(outputs, 1)
                for label, prediction in zip(labels, predictions):
                    logging.debug("{} predicted as {}".format(label, prediction))
                    if label == prediction:
                        correct_pred[label] += 1
                        total_correct += 1
                    total_pred[label] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")

        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = loss_val / count
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


class CNN(Model):
    """
    Class for a CNN Model for Celeba

    """

    def __init__(self):
        """
        Constructor. Instantiates the CNN Model
            with 84*84*3 Input and 2 output classes

        """
        super().__init__()
        # 2.8k parameters
        self.conv1 = nn.Conv2d(CHANNELS, 32, 3, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv3 = nn.Conv2d(32, 32, 3, padding="same")
        self.conv4 = nn.Conv2d(32, 32, 3, padding="same")
        self.fc1 = nn.Linear(5 * 5 * 32, NUM_CLASSES)

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
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
