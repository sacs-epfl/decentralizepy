import collections
import json
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.Partitioner import DataPartitioner
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

VOCAB_LEN = 9999  # 10000 was used as it needed to be +1 due to using mask_zero in the tf embedding
SEQ_LEN = 10
EMBEDDING_DIM = 200


class Reddit(Dataset):
    """
    Class for the Reddit dataset
    --  Based on https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        and Femnist.py
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
        Function to read all the Reddit data files in the directory

        Parameters
        ----------
        data_dir : str
            Path to the folder containing the data files

        Returns
        -------
        3-tuple
            A tuple containing list of users, number of samples per client,
            and the data items per client

        """
        users = []
        num_samples = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith(".json")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            u, n, d = self.__read_file__(file_path)
            users.extend(u)
            num_samples.extend(n)
            data.update(d)
        return users, num_samples, data

    def file_per_user(self, dir, write_dir):
        """
        Function to read all the Reddit data files and write one file per user

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
            e = c_len // self.num_partitions
            frac = e / c_len
            self.sizes = [frac] * self.num_partitions
            self.sizes[-1] += 1.0 - frac * self.num_partitions
            logging.debug("Size fractions: {}".format(self.sizes))

        my_clients = DataPartitioner(files, self.sizes).use(self.dataset_id)
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
                self.clients.append(cur_client)
                processed_x, processed_y = self.prepare_data(train_data[cur_client])
                # processed_x is an list of fixed size word id arrays that represent a phrase
                # processed_y is a list of word ids that each represent the next word of a phrase
                my_train_data["x"].extend(processed_x)
                my_train_data["y"].extend(processed_y)
                self.num_samples.append(len(processed_y))
        # turns the list of lists into a single list
        self.train_y = np.array(my_train_data["y"], dtype=np.dtype("int64")).reshape(-1)
        self.train_x = np.array(
            my_train_data["x"], dtype=np.dtype("int64")
        )  # .reshape(-1)
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
            processed_x, processed_y = self.prepare_data(test_data)
            # processed_x is an list of fixed size word id arrays that represent a phrase
            # processed_y is a list of word ids that each represent the next word of a phrase
            test_x.extend(processed_x)
            test_y.extend(processed_y)
        self.test_y = np.array(test_y, dtype=np.dtype("int64")).reshape(-1)
        self.test_x = np.array(test_x, dtype=np.dtype("int64"))
        logging.info("test_x.shape: %s", str(self.test_x.shape))
        logging.info("test_y.shape: %s", str(self.test_y.shape))
        assert self.test_x.shape[0] == self.test_y.shape[0]
        assert self.test_x.shape[0] > 0

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1024,
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
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
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
            only_local,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )
        if self.train_dir and Path(self.train_dir).exists():
            vocab_path = os.path.join(self.train_dir, "../../vocab/reddit_vocab.pck")
            (
                self.vocab,
                self.vocab_size,
                self.unk_symbol,
                self.pad_symbol,
            ) = self._load_vocab(vocab_path)
            logging.info("The reddit vocab has %i tokens.", len(self.vocab))
        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

        # TODO: Add Validation

    def _load_vocab(self, VOCABULARY_PATH):
        """
        loads the training vocabulary
        copied from https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        Parameters
        ----------
        VOCABULARY_PATH : str
            Path to the pickled training vocabulary
        Returns
        -------
            Tuple
                vocabulary, size, unk symbol, pad symbol
        """
        vocab_file = pickle.load(open(VOCABULARY_PATH, "rb"))
        vocab = collections.defaultdict(lambda: vocab_file["unk_symbol"])
        vocab.update(vocab_file["vocab"])

        return (
            vocab,
            vocab_file["size"],
            vocab_file["unk_symbol"],
            vocab_file["pad_symbol"],
        )

    def prepare_data(self, data):
        """
        copied from https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        Parameters
        ----------
        data

        Returns
        -------

        """
        data_x = data["x"]
        data_y = data["y"]

        # flatten lists
        def flatten_lists(data_x_by_comment, data_y_by_comment):
            data_x_by_seq, data_y_by_seq = [], []
            for c, l in zip(data_x_by_comment, data_y_by_comment):
                data_x_by_seq.extend(c)
                data_y_by_seq.extend(l["target_tokens"])

            return data_x_by_seq, data_y_by_seq

        data_x, data_y = flatten_lists(data_x, data_y)

        data_x_processed = self.process_x(data_x)
        data_y_processed = self.process_y(data_y)

        filtered_x, filtered_y = [], []
        for i in range(len(data_x_processed)):
            if np.sum(data_y_processed[i]) != 0:
                filtered_x.append(data_x_processed[i])
                filtered_y.append(data_y_processed[i])

        return (filtered_x, filtered_y)

    def _tokens_to_ids(self, raw_batch):
        """
        Turns an list of list of tokens that are of the same size (with padding <PAD>) if needed
        into a list of list of word ids

        [['<BOS>', 'do', 'you', 'have', 'proof', 'of', 'purchase', 'for', 'clay', 'play'], [ ...], ...]
        turns into:
        [[   5   45   13   24 1153   11 1378   17 6817  165], ...]

        copied from https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        Parameters
        ----------
        raw_batch : list
            list of fixed size token lists

        Returns
        -------
            2D array with the rows representing fixed size token_ids pharases
        """

        def tokens_to_word_ids(tokens, word2id):
            return [word2id[word] for word in tokens]

        to_ret = [tokens_to_word_ids(seq, self.vocab) for seq in raw_batch]
        return np.array(to_ret)

    def process_x(self, raw_x_batch):
        """
        copied from https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        Parameters
        ----------
        raw_x_batch

        Returns
        -------

        """
        tokens = self._tokens_to_ids([s for s in raw_x_batch])
        return tokens

    def process_y(self, raw_y_batch):
        """
        copied from https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        Parameters
        ----------
        raw_y_batch

        Returns
        -------

        """
        tokens = self._tokens_to_ids([s for s in raw_y_batch])

        def getNextWord(token_ids):
            n = len(token_ids)
            for i in range(n):
                # gets the word at the end of the phrase that should be predicted
                # that is the last token that is not a pad.
                if token_ids[n - i - 1] != self.pad_symbol:
                    return token_ids[n - i - 1]
            return self.pad_symbol

        return [getNextWord(t) for t in tokens]

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

        correct_pred = [0 for _ in range(VOCAB_LEN)]
        total_pred = [0 for _ in range(VOCAB_LEN)]

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


class RNN(Model):
    """
    Class for a RNN Model for Reddit

    """

    def __init__(self):
        """
        Constructor. Instantiates the RNN Model to predict the next word of a sequence of word.
        Based on the TensorFlow model found here: https://gitlab.epfl.ch/sacs/efficient-federated-learning/-/blob/master/grad_guessing/data_utils.py
        """
        super().__init__()

        # input_length does not exist
        self.embedding = nn.Embedding(VOCAB_LEN, EMBEDDING_DIM, padding_idx=0)
        self.rnn_cells = nn.LSTM(EMBEDDING_DIM, 256, batch_first=True, num_layers=2)
        # activation function is added in the forward pass
        # Note: the tensorflow implementation did not use any activation function in this step?
        # should I use one.
        self.l1 = nn.Linear(256, 128)
        # the tf model used sofmax activation here
        self.l2 = nn.Linear(128, VOCAB_LEN)

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
        x = self.embedding(x)
        x = self.rnn_cells(x)
        last_layer_output = x[1][0][1, ...]
        x = F.relu(self.l1(last_layer_output))
        x = self.l2(x)
        # softmax is applied by the CrossEntropyLoss used during training
        return x
