from decentralizepy import utils
from decentralizepy.mappings.Mapping import Mapping


class Dataset:
    """
    This class defines the Dataset API.
    All datasets must follow this API.

    """

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        only_local=False,
        train_dir="",
        test_dir="",
        sizes="",
        test_batch_size="",
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
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.only_local = only_local
        self.dataset_id = self.rank if self.only_local else self.uid
        self.num_partitions = (
            self.mapping.get_local_procs_count()
            if self.only_local
            else self.mapping.get_n_procs()
        )
        self.train_dir = utils.conditional_value(train_dir, "", None)
        self.test_dir = utils.conditional_value(test_dir, "", None)
        self.sizes = utils.conditional_value(sizes, "", None)
        self.test_batch_size = utils.conditional_value(test_batch_size, "", 64)
        self.num_classes = None
        if self.sizes:
            if type(self.sizes) == str:
                self.sizes = eval(self.sizes)

        if train_dir:
            self.__training__ = True
        else:
            self.__training__ = False

        if test_dir:
            self.__testing__ = True
        else:
            self.__testing__ = False

        self.label_distribution = None

    def get_label_distribution(self):
        # Only supported for classification
        if self.label_distribution == None:
            self.label_distribution = [0 for _ in range(self.num_classes)]
            tr_set = self.get_trainset()
            for _, ys in tr_set:
                for y in ys:
                    y_val = y.item()
                    self.label_distribution[y_val] += 1

        return self.label_distribution

    def get_trainset(self):
        """
        Function to get the training set

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        raise NotImplementedError

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
        raise NotImplementedError
