from random import Random

import numpy as np

""" Adapted from https://pytorch.org/tutorials/intermediate/dist_tuto.html """


class Partition(object):
    """
    Class for holding the data partition

    """

    def __init__(self, data, index):
        """
        Constructor. Caches the data and the indices

        Parameters
        ----------
        data : indexable
        index : list
            A list of indices

        """
        self.data = data
        self.index = index

    def __len__(self):
        """
        Function to retrieve the length

        Returns
        -------
        int
            Number of items in the data

        """
        return len(self.index)

    def __getitem__(self, index):
        """
        Retrieves the item in data with the given index

        Parameters
        ----------
        index : int

        Returns
        -------
        Data
            The data sample with the given `index` in the dataset

        """
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0], seed=1234):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, rank):
        """
        Get the partition for the process with the given `rank`

        Parameters
        ----------
        rank : int
            Rank of the current process

        Returns
        -------
        Partition
            The dataset partition of the current process

        """
        return Partition(self.data, self.partitions[rank])


class SimpleDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0], seed=1234):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]


class KShardDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset

    """

    def __init__(self, data, sizes=[1.0], shards=1, seed=1234):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            A list of fractions for each process
        shards : int
            Number of shards to allot to process
        seed : int, optional
            Seed for generating a random subset

        """
        self.data = data
        self.partitions = []
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng = Random()
        rng.seed(seed)

        for frac in sizes:
            self.partitions.append([])
            for _ in range(shards):
                start = rng.randint(0, len(indexes) - 1)
                part_len = int(frac * data_len) // shards
                if start + part_len > len(indexes):
                    self.partitions[-1].extend(indexes[start:])
                    self.partitions[-1].extend(
                        indexes[: (start + part_len - len(indexes))]
                    )
                    indexes = indexes[(start + part_len - len(indexes)) : start]
                else:
                    self.partitions[-1].extend(indexes[start : start + part_len])
                    index_start = indexes[:start]
                    index_start.extend(indexes[start + part_len :])
                    indexes = index_start


class DirichletDataPartitioner(DataPartitioner):
    """
    Class to partition the dataset using Dirichlet Function
    Modified from https://gitlab.epfl.ch/sacs/collaborative-inference/-/blob/main/src/decentralizepy/datasets/Partitioner.py#L207

    """

    def __init__(self, data, sizes=[1.0], seed=1234, alpha=0.1, num_classes=10):
        """
        Constructor. Partitions the data according the parameters

        Parameters
        ----------
        data : indexable
            An indexable list of data items
        sizes : list(float)
            Not used for partitioning, but kept for compatibility
        shards : int
            Number of shards to allot to process
        seed : int, optional
            Seed for generating a random subset
        alpha : float
            Degree of heterogeneity. Lower is more heterogeneous.

        """

        self.data = data
        self.seed = seed
        self.num_classes = num_classes
        self.alpha = alpha
        self.partitions, self.ratio = self.__getDirichletData__(
            np.array(data.targets), len(sizes), seed, self.alpha, num_classes
        )

    def __getDirichletData__(self, labelList, n_nets, seed, alpha, K):
        """
        Function to partition the data using Dirichlet Function

        Parameters
        ----------
        labelList : np.ndarray
            Array of labels
        n_nets : int
            Number of clients
        seed : int
            Seed for generating a random subset
        alpha : float
            Degree of heterogeneity. Lower is more heterogeneous.
        K : int
            Number of classes
        
        """

        min_size = 0
        N = len(labelList)
        rng = np.random.default_rng(seed)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                rng.shuffle(idx_k)
                proportions = rng.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            rng.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        counts = local_sizes  # return counts insteads of ratios

        return idx_batch, counts
