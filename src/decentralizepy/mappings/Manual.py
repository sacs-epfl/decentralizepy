from decentralizepy.mappings.Mapping import Mapping


class Manual(Mapping):
    """
    This class defines the manual mapping

    """

    def __init__(
        self, n_machines, procs_per_machine, global_service_machine=0, current_machine=0
    ):
        """
        Constructor

        Parameters
        ----------
        n_machines : int
            Number of machines involved in learning
        procs_per_machine : list(int)
            A list of number of processes spawned per machine
        global_service_machine: int, optional
            Machine ID on which the server/services are hosted
        current_machine: int, optional
            Machine ID of local machine

        """

        self.n_procs = 0
        for i in procs_per_machine:
            self.n_procs += i
        super().__init__(self.n_procs)
        self.n_machines = n_machines
        self.procs_per_machine = procs_per_machine
        self.global_service_machine = global_service_machine
        self.current_machine = current_machine

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
        if rank < 0:
            return rank
        cur_uid = 0
        for i in range(machine_id):
            cur_uid += self.procs_per_machine[i]
        return cur_uid + rank

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
        if uid < 0:
            return uid, self.global_service_machine

        machine, rank = 0, 0
        for procs in self.procs_per_machine:
            if uid < procs:
                rank = uid
                break
            else:
                machine += 1
                uid -= procs
        return rank, machine

    def get_local_procs_count(self):
        """
        Gives number of processes that run on the node

        Returns
        -------
        int
            the number of local processes

        """

        return self.procs_per_machine[self.current_machine]
