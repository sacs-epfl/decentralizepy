from decentralizepy.mappings.Mapping import Mapping


class Linear(Mapping):
    """
    This class defines the mapping:
        uid = machine_id * procs_per_machine + rank

    """

    def __init__(self, n_machines, procs_per_machine):
        """
        Constructor

        Parameters
        ----------
        n_machines : int
            Number of machines involved in learning
        procs_per_machine : int
            Number of processes spawned per machine

        """
        super().__init__(n_machines * procs_per_machine)
        self.n_machines = n_machines
        self.procs_per_machine = procs_per_machine

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
        return machine_id * self.procs_per_machine + rank

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
        return (uid % self.procs_per_machine), (uid // self.procs_per_machine)

    def get_local_procs_count(self):
        """
        Gives number of processes that run on the node

        Returns
        -------
        int
            the number of local processes

        """

        return self.procs_per_machine
