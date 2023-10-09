import logging
import os
from collections import defaultdict
from random import Random
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.datasets.text.LLMData import LLMData
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.models.Model import Model

NUM_CLASSES = 5


class AmazonReview(Dataset):
    """
    Class for the Amazon review dataset
    --  Based on https://github.com/SymbioticLab/FedScale/blob/master/fedscale/dataloaders/amazon.py
    """

    def __read_file__(self, file_path):
        """
        Read data from the given csv file

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
            client_data = pd.read_csv(inf)
            client_data.drop(columns=["label_id"])
        client_id = file_path.split("/")[-1].split(".")[0]
        return (
            [client_id],  # client id
            [len(client_data)],  # number of samples
            {client_id: client_data},
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
        files = [f for f in files if f.endswith(".csv")]
        for f in files:
            file_path = os.path.join(data_dir, f)
            u, n, d = self.__read_file__(file_path)
            users.extend(u)
            num_samples.extend(n)
            data.update(d)
        return users, num_samples, data

    def load_trainset(self):
        """
        Loads the training set. Partitions it if needed.

        """
        logging.info("Loading training set.")
        files = os.listdir(self.train_dir)
        files = [f for f in files if f.endswith(".csv")]
        files.sort()
        c_len = len(files)

        rng = Random()
        rng.seed(self.random_seed)
        rng.shuffle(files)

        my_clients = [files[self.dataset_id]]
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
                my_train_data["x"].extend(train_data[cur_client]["data_path"].to_list())
                my_train_data["y"].extend(
                    train_data[cur_client]["label_name"].to_list()
                )
                self.num_samples.append(len(train_data[cur_client]))
        # turns the list of lists into a single list
        self.train_y = torch.nn.functional.one_hot(
            torch.tensor(my_train_data["y"]).to(torch.int64) - 1,
            num_classes=NUM_CLASSES,
        ).to(torch.float32)
        self.train_x = self.tokenizer(
            my_train_data["x"], return_tensors="pt", truncation=True, padding=True
        )
        assert self.train_x["input_ids"].shape[0] == self.train_y.shape[0]
        assert self.train_y.shape[0] > 0

    def load_testset(self):
        """
        Loads the testing set.

        """
        logging.info("Loading testing set.")
        _, _, d = self.__read_dir__(self.test_dir)
        test_x = []
        test_y = []
        for test_data in d.values():
            test_x.extend(test_data["data_path"].to_list())
            test_y.extend(test_data["label_name"].to_list())
        self.test_y = torch.nn.functional.one_hot(
            torch.tensor(test_y).to(torch.int64) - 1, num_classes=NUM_CLASSES
        ).to(torch.float32)
        self.test_x = self.tokenizer(
            test_x, return_tensors="pt", truncation=True, padding=True
        )
        assert self.test_x["input_ids"].shape[0] == self.test_y.shape[0]
        assert self.test_y.shape[0] > 0

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        random_seed: int = 1234,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=128,
        tokenizer="BERT",
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
        random_seed : int, optional
            Random seed for the dataset. Default value is 1234
        only_local : bool, optional
            True if the dataset needs to be partioned only among local procs, False otherwise
        train_dir : str, optional
            Path to the training data files. Required to instantiate the training set
            The training set is partitioned according to the number of global processes and sizes
        test_dir : str, optional
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
            random_seed,
            only_local,
            train_dir,
            test_dir,
            sizes,
            test_batch_size,
        )
        if tokenizer == "MobileBERT":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/mobilebert-uncased", model_max_length=512
            )
        elif tokenizer == "BERT":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", model_max_length=512
            )
        else:
            raise ValueError("Tokenizer not supported")

        if self.__training__:
            self.load_trainset()

        if self.__testing__:
            self.load_testset()

    def get_trainset(self, batch_size=1, shuffle=True):
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
                LLMData(self.train_x, self.train_y),
                batch_size=batch_size,
                shuffle=shuffle,
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        """
        Function to get the test set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.text.LLMData)

        Raises
        ------
        RuntimeError
            If the test set was not initialized

        """
        if self.__testing__:
            return DataLoader(
                LLMData(self.test_x, self.test_y),
                batch_size=self.test_batch_size,
                shuffle=True,
            )
        raise RuntimeError("Test set not initialized!")

    def test(self, model, one_batch=True):
        model.eval()

        correct_pred = torch.tensor([0 for _ in range(NUM_CLASSES)]).to(torch.int64)
        total_pred = torch.tensor([0 for _ in range(NUM_CLASSES)]).to(torch.int64)

        total_correct = 0
        total_predicted = 0

        testloader = self.get_testset()

        with torch.no_grad():
            loss_val = 0.0
            count = 0
            for batch in testloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss_val += outputs.loss.item()
                logits = outputs.logits
                count += 1
                _, predictions = torch.max(logits, 1)
                _, non_hot_label = torch.max(labels, 1)

                correct_mask = non_hot_label == predictions
                correct_pred += torch.bincount(
                    non_hot_label[correct_mask], minlength=non_hot_label.max() + 1
                )
                total_pred += torch.bincount(
                    non_hot_label, minlength=non_hot_label.max() + 1
                )

                total_correct += correct_mask.sum().item()
                total_predicted += len(non_hot_label)

                break

                # for label, prediction in zip(non_hot_label, predictions):
                #     # print("{} predicted as {}".format(label, prediction))
                #     if label == prediction:
                #         correct_pred[label] += 1
                #         total_correct += 1
                #     total_pred[label] += 1
                #     total_predicted += 1

            for key, value in enumerate(correct_pred):
                if total_pred[key] != 0:
                    accuracy = 100 * float(value) / total_pred[key]
                else:
                    accuracy = 100.0
                logging.debug(
                    "Accuracy for class {} is: {:.1f} %".format(key, accuracy)
                )

            accuracy = 100 * float(total_correct) / total_predicted
            loss_val = loss_val / count
            logging.info("Overall test accuracy is: {:.1f} %".format(accuracy))
            return accuracy, loss_val


class MobileBERT(AutoModelForSequenceClassification):
    """
    Class for a LLM Model for Amazon Reviews

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor. Instantiates the LLM.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "google/mobilebert-uncased",
            num_labels=NUM_CLASSES,
            problem_type="multi_label_classification",
        )

    def __getattr__(self, __name: str):
        return self.model.__getattribute__(__name)

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)


class BERT(AutoModelForSequenceClassification):
    """
    Class for a LLM Model for Amazon Reviews

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor. Instantiates the LLM.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=NUM_CLASSES,
            problem_type="multi_label_classification",
        )

    def __getattr__(self, __name: str):
        return self.model.__getattribute__(__name)

    def __call__(self, *args, **kwargs):
        return self.model.__call__(*args, **kwargs)
