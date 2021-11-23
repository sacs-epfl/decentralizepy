class Communication:
    """
    Communcation API
    """

    def __init__(self, rank, machine_id, mapping, total_procs):
        self.total_procs = total_procs
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = mapping.get_uid(rank, machine_id)

    def encrypt(self, data):
        raise NotImplementedError

    def decrypt(self, sender, data):
        raise NotImplementedError

    def connect_neighbors(self, neighbors):
        raise NotImplementedError

    def receive(self):
        raise NotImplementedError

    def send(self, uid, data):
        raise NotImplementedError

    def disconnect_neighbors(self):
        raise NotImplementedError
