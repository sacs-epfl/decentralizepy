import json
import logging
import pickle
from collections import deque

import zmq

from decentralizepy.communication.Communication import Communication

HELLO = b"HELLO"
BYE = b"BYE"


class TCP(Communication):
    """
    TCP Communication API

    """

    def addr(self, rank, machine_id):
        """
        Returns TCP address of the process.

        Parameters
        ----------
        rank : int
            Local rank of the process
        machine_id : int
            Machine id of the process

        Returns
        -------
        str
            Full address of the process using TCP

        """
        machine_addr = self.ip_addrs[str(machine_id)]
        port = (2 * rank + 1) + self.offset
        assert port > 0
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        total_procs,
        addresses_filepath,
        offset=9000,
    ):
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
        addresses_filepath : str
            JSON file with machine_id -> ip mapping
        compression_package : str
            Import path of a module that implements the compression.Compression.Compression class
        compression_class : str
            Name of the compression class inside the compression package

        """
        super().__init__(rank, machine_id, mapping, total_procs)

        with open(addresses_filepath) as addrs:
            self.ip_addrs = json.load(addrs)

        self.total_procs = total_procs
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.offset = offset
        self.uid = mapping.get_uid(rank, machine_id)
        self.identity = str(self.uid).encode()
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.IDENTITY, self.identity)
        self.router.bind(self.addr(rank, machine_id))

        self.total_data = 0
        self.total_meta = 0

        self.peer_deque = deque()
        self.peer_sockets = dict()

    def __del__(self):
        """
        Destroys zmq context

        """
        self.context.destroy(linger=0)

    def encrypt(self, data):
        """
        Encode data as python pickle.

        Parameters
        ----------
        data : dict
            Data dict to send

        Returns
        -------
        byte
            Encoded data

        """
        data_len = 0
        if "params" in data:
            data_len = len(pickle.dumps(data["params"]))
        output = pickle.dumps(data)
        self.total_meta += len(output) - data_len
        self.total_data += data_len
        return output

    def decrypt(self, sender, data):
        """
        Decode received pickle data.

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
        sender = int(sender.decode())
        data = pickle.loads(data)
        return sender, data

    def init_connection(self, neighbor):
        """
        Initiates a socket to a given node.

        Parameters
        ----------
        neighbor : int
            neighbor to connect to

        """
        logging.debug("Connecting to my neighbour: {}".format(neighbor))
        id = str(neighbor).encode()
        req = self.context.socket(zmq.DEALER)
        req.setsockopt(zmq.IDENTITY, self.identity)
        req.connect(self.addr(*self.mapping.get_machine_and_rank(neighbor)))
        self.peer_sockets[id] = req

    def receive(self):
        """
        Returns ONE message received.

        Returns
        ----------
        dict
            Received and decrypted data

        Raises
        ------
        RuntimeError
            If received HELLO

        """

        sender, recv = self.router.recv_multipart()
        return self.decrypt(sender, recv)

    def send(self, uid, data, encrypt=True):
        """
        Send a message to a process.

        Parameters
        ----------
        uid : int
            Neighbor's unique ID
        data : dict
            Message as a Python dictionary

        """
        if encrypt:
            to_send = self.encrypt(data)
        else:
            to_send = data
        data_size = len(to_send)
        self.total_bytes += data_size
        id = str(uid).encode()
        self.peer_sockets[id].send(to_send)
        logging.debug("{} sent the message to {}.".format(self.uid, uid))
        logging.info("Sent message size: {}".format(data_size))
