import logging
import math
import os
import zipfile

import pandas as pd
import requests
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from decentralizepy.datasets.Data import Data
from decentralizepy.datasets.Dataset import Dataset
from decentralizepy.mappings import Mapping
from decentralizepy.models.Model import Model


class MovieLens(Dataset):
    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size=1,
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
        self.n_users, self.n_items, df_train, df_test = self._load_data()
        self.train_data, self.test_data = self._split_data(
            df_train, df_test, self.num_partitions
        )

        # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        # [0,   1,   2,   3,   4,   5,   6,   7,   8,   9  ]
        self.NUM_CLASSES = 10
        self.RATING_DICT = {
            0.5: 0,
            1.0: 1,
            1.5: 2,
            2.0: 3,
            2.5: 4,
            3.0: 5,
            3.5: 6,
            4.0: 7,
            4.5: 8,
            5.0: 9,
        }

    def _load_data(self):
        f_ratings = os.path.join(self.train_dir, "ml-latest-small", "ratings.csv")
        names = ["user_id", "item_id", "rating", "timestamp"]
        df_ratings = pd.read_csv(f_ratings, sep=",", names=names, skiprows=1).drop(
            columns=["timestamp"]
        )
        # map item_id properly
        items_count = df_ratings["item_id"].nunique()
        items_ids = sorted(list(df_ratings["item_id"].unique()))
        assert items_count == len(items_ids)
        for i in range(0, items_count):
            df_ratings.loc[df_ratings["item_id"] == items_ids[i], "item_id"] = i + 1

        # split train, test - 70% : 30%
        grouped_users = df_ratings.groupby(["user_id"])
        users_count = len(grouped_users)
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        for i in range(0, users_count):
            df_user = df_ratings[df_ratings["user_id"] == i + 1]
            df_user_train = df_user.sample(frac=0.7)
            df_user_test = pd.concat([df_user, df_user_train]).drop_duplicates(
                keep=False
            )
            assert len(df_user_train) + len(df_user_test) == len(df_user)

            df_train = pd.concat([df_train, df_user_train])
            df_test = pd.concat([df_test, df_user_test])

        # 610, 9724
        return users_count, items_count, df_train, df_test

    def _split_data(self, train_data, test_data, world_size):
        # SPLITTING BY USERS: group by users and split the data accordingly
        mod = self.n_users % world_size
        users_count = self.n_users // world_size
        if self.dataset_id < mod:
            users_count += 1
            offset = users_count * self.dataset_id
        else:
            offset = users_count * self.dataset_id + mod

        my_train_data = pd.DataFrame()
        my_test_data = pd.DataFrame()
        for i in range(offset, offset + users_count):
            my_train_data = pd.concat(
                [my_train_data, train_data[train_data["user_id"] == i + 1]]
            )
            my_test_data = pd.concat(
                [my_test_data, test_data[test_data["user_id"] == i + 1]]
            )

        logging.info("Data split for test and train.")
        return my_train_data, my_test_data

    def get_trainset(self, batch_size=1, shuffle=False):
        if self.__training__:
            train_x = self.train_data[["user_id", "item_id"]].to_numpy()
            train_y = self.train_data.rating.values.astype("float32")
            return DataLoader(
                Data(train_x, train_y), batch_size=batch_size, shuffle=shuffle
            )
        raise RuntimeError("Training set not initialized!")

    def get_testset(self):
        if self.__testing__:
            test_x = self.test_data[["user_id", "item_id"]].to_numpy()
            test_y = self.test_data.rating.values
            return DataLoader(Data(test_x, test_y), batch_size=self.test_batch_size)
        raise RuntimeError("Test set not initialized!")

    def test(self, model, loss):
        test_set = self.get_testset()
        logging.debug("Test Loader instantiated.")

        correct_pred = [0 for _ in range(self.NUM_CLASSES)]
        total_pred = [0 for _ in range(self.NUM_CLASSES)]
        total_correct = 0
        total_predicted = 0

        with torch.no_grad():
            loss_val = 0.0
            loss_predicted = 0.0
            count = 0

            for test_x, test_y in test_set:
                output = model(test_x)
                loss_val += loss(output, test_y).item()
                count += 1

                # threshold values to range [0.5, 5.0]
                o1 = (output > 5.0).nonzero(as_tuple=False)
                output[o1] = 5.0
                o1 = (output < 0.5).nonzero(as_tuple=False)
                output[o1] = 0.5
                # round a number to the closest half integer
                output = torch.round(output * 2) / 2
                loss_predicted += metrics.mean_absolute_error(output, test_y)

                for rating, prediction in zip(test_y.tolist(), output):
                    # print(rating, prediction)
                    logging.debug("{} predicted as {}".format(rating, prediction))
                    if rating == prediction:
                        correct_pred[self.RATING_DICT[rating]] += 1
                        total_correct += 1
                    total_pred[self.RATING_DICT[rating]] += 1
                    total_predicted += 1

        logging.debug("Predicted on the test set")
        for key, value in enumerate(correct_pred):
            if total_pred[key] != 0:
                accuracy = 100 * float(value) / total_pred[key]
            else:
                accuracy = 100.0
            logging.debug("Accuracy for class {} is: {:.1f} %".format(key, accuracy))

        accuracy = 100 * float(total_correct) / total_predicted
        loss_val = math.sqrt(loss_val / count)
        loss_predicted = loss_predicted / count
        logging.info(
            "MSE loss: {:.8f} | Rounded MAE loss: {:.8f}".format(
                loss_val, loss_predicted
            )
        )
        logging.info("Overall accuracy is: {:.1f} %".format(accuracy))
        return accuracy, loss_val


# todo: this class should be in 'models' package; add support for reading it from there and move it
class MatrixFactorization(Model):
    """
    Class for a Matrix Factorization model for MovieLens.
    """

    def __init__(self, n_users=610, n_items=9724, n_factors=20):
        """
        Instantiates the Matrix Factorization model with user and item embeddings.

        Parameters
        ----------
        n_users
            The number of unique users.
        n_items
            The number of unique items.
        n_factors
            The number of columns in embeddings matrix.
        """
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_factors.weight.data.uniform_(-0.05, 0.05)
        self.item_factors.weight.data.uniform_(-0.05, 0.05)

    def forward(self, data):
        """
        Forward pass of the model, it does matrix multiplication and returns predictions for given users and items.
        """
        users = torch.LongTensor(data[:, 0]) - 1
        items = torch.LongTensor(data[:, 1]) - 1
        u, it = self.user_factors(users), self.item_factors(items)
        x = (u * it).sum(dim=1, keepdim=True)
        return x.squeeze(1)


def download_movie_lens(dest_path):
    """
    Downloads the movielens latest small dataset.
    This data set consists of:
        * 100836 ratings from 610 users on 9742 movies.
        * Each user has rated at least 20 movies.

    https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
    """
    url = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    req = requests.get(url, stream=True)

    print("Downloading MovieLens Latest Small data...")
    with open(os.path.join(dest_path, "ml-latest-small.zip"), "wb") as fd:
        for chunk in req.iter_content(chunk_size=None):
            fd.write(chunk)
    with zipfile.ZipFile(os.path.join(dest_path, "ml-latest-small.zip"), "r") as z:
        z.extractall(dest_path)
    print("Downloaded MovieLens Latest Small dataset at", dest_path)


if __name__ == "__main__":
    path = "/mnt/nfs/shared/leaf/data/movielens"
    zip_file = os.path.join(path, "ml-latest-small.zip")
    if not os.path.isfile(zip_file):
        download_movie_lens(path)
