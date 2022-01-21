class Communication:
    """
    Communcation API

    """

    def __init__(self, rank, machine_id, mapping, total_procs):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank of the process
        machine_id : int
            Machine id of the process
        mapping : decentralizepy.mappings.Mapping
            uid, rank, machine_id invertible mapping
        total_procs : int
            Total number of processes

        """
        self.total_procs = total_procs
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = mapping.get_uid(rank, machine_id)
        self.total_bytes = 0

    def encrypt(self, data):
        """
        Encode/Encrypt data.

        Parameters
        ----------
        data : dict
            Data dict to send

        Returns
        -------
        byte
            Encoded data

        """
        raise NotImplementedError

    def decrypt(self, sender, data):
        """
        Decodes received data.

        Parameters
        ----------
        sender : byte
            sender of the data
        data : byte
            Data received

        Returns
        -------
        tuple
            (sender: int, data: dict)

        """
        raise NotImplementedError

    def connect_neighbors(self, neighbors):
        """
        Connects all neighbors.

        Parameters
        ----------
        neighbors : list(int)
            List of neighbors

        """
        raise NotImplementedError

    def receive(self):
        """
        Returns ONE message received.

        Returns
        ----------
        dict
            Received and decrypted data

        """
        raise NotImplementedError

    def send(self, uid, data):
        """
        Send a message to a process.

        Parameters
        ----------
        uid : int
            Neighbor's unique ID
        data : dict
            Message as a Python dictionary

        """
        raise NotImplementedError

    def disconnect_neighbors(self):
        """
        Disconnects all neighbors.

        """
        raise NotImplementedError
