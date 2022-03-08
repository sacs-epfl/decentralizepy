class Mapping:
    """
    This class defines the bidirectional mapping between:
        1. The unique identifier
        2. machine_id and rank

    """

    def __init__(self, n_procs):
        """
        Constructor

        Parameters
        ----------
        n_procs : int
            Total number of processes

        """
        self.n_procs = n_procs

    def get_n_procs(self):
        """
        Gives the global sum of all processes that are spawned on the machines

        Returns
        -------
        int
            the number of global processes
        """

        return self.n_procs

    def get_uid(self, rank: int, machine_id: int):
        """
        Gives the global unique identifier of the node

        Parameters
        ----------
        rank : int
            Node's rank on its machine
        machine_id : int
            node's machine in the cluster

        Returns
        -------
        int
            the unique identifier

        """

        raise NotImplementedError

    def get_machine_and_rank(self, uid: int):
        """
        Gives the rank and machine_id of the node

        Parameters
        ----------
        uid : int
            globally unique identifier of the node

        Returns
        -------
        2-tuple
            a tuple of rank and machine_id

        """

        raise NotImplementedError

    def get_local_procs_count(self):
        """
        Gives number of processes that run on the node

        Returns
        -------
        int
            the number of local processes

        """

        raise NotImplementedError
