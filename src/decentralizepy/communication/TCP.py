import importlib
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
        port = rank + self.offset
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        total_procs,
        addresses_filepath,
        compress=False,
        offset=20000,
        compression_package=None,
        compression_class=None,
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
        self.offset = 20000 + offset
        self.uid = mapping.get_uid(rank, machine_id)
        self.identity = str(self.uid).encode()
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.IDENTITY, self.identity)
        self.router.bind(self.addr(rank, machine_id))
        self.sent_disconnections = False
        self.compress = compress

        if compression_package and compression_class:
            compressor_module = importlib.import_module(compression_package)
            compressor_class = getattr(compressor_module, compression_class)
            self.compressor = compressor_class()
            logging.info(f"Using the {compressor_class} to compress the data")
        else:
            assert not self.compress

        self.total_data = 0
        self.total_meta = 0

        self.peer_deque = deque()
        self.peer_sockets = dict()
        self.barrier = set()

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
        if self.compress:
            if "indices" in data:
                data["indices"] = self.compressor.compress(data["indices"])
                meta_len = len(
                    pickle.dumps(data["indices"])
                )  # ONLY necessary for the statistics
            if "params" in data:
                data["params"] = self.compressor.compress_float(data["params"])
            output = pickle.dumps(data)
            # the compressed meta data gets only a few bytes smaller after pickling
            self.total_meta += meta_len
            self.total_data += len(output) - meta_len
        else:
            output = pickle.dumps(data)
            # centralized testing uses its own instance
            if type(data) == dict:
                if "indices" in data:
                    meta_len = len(pickle.dumps(data["indices"]))
                else:
                    meta_len = 0
                self.total_meta += meta_len
                self.total_data += len(output) - meta_len
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
        if self.compress:
            data = pickle.loads(data)
            if "indices" in data:
                data["indices"] = self.compressor.decompress(data["indices"])
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        else:
            data = pickle.loads(data)
        return sender, data

    def connect_neighbors(self, neighbors):
        """
        Connects all neighbors. Sends HELLO. Waits for HELLO.
        Caches any data received while waiting for HELLOs.

        Parameters
        ----------
        neighbors : list(int)
            List of neighbors

        Raises
        ------
        RuntimeError
            If received BYE while waiting for HELLO

        """
        logging.info("Sending connection request to neighbors")
        for uid in neighbors:
            logging.debug("Connecting to my neighbour: {}".format(uid))
            id = str(uid).encode()
            req = self.context.socket(zmq.DEALER)
            req.setsockopt(zmq.IDENTITY, self.identity)
            req.connect(self.addr(*self.mapping.get_machine_and_rank(uid)))
            self.peer_sockets[id] = req
            req.send(HELLO)

        num_neighbors = len(neighbors)
        while len(self.barrier) < num_neighbors:
            sender, recv = self.router.recv_multipart()

            if recv == HELLO:
                logging.debug("Received {} from {}".format(HELLO, sender))
                self.barrier.add(sender)
            elif recv == BYE:
                logging.debug("Received {} from {}".format(BYE, sender))
                raise RuntimeError(
                    "A neighbour wants to disconnect before training started!"
                )
            else:
                logging.debug(
                    "Received message from {} @ connect_neighbors".format(sender)
                )

                self.peer_deque.append(self.decrypt(sender, recv))

        logging.info("Connected to all neighbors")
        self.initialized = True

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
        assert self.initialized == True
        if len(self.peer_deque) != 0:
            resp = self.peer_deque.popleft()
            return resp

        sender, recv = self.router.recv_multipart()

        if recv == HELLO:
            logging.debug("Received {} from {}".format(HELLO, sender))
            raise RuntimeError(
                "A neighbour wants to connect when everyone is connected!"
            )
        elif recv == BYE:
            logging.debug("Received {} from {}".format(BYE, sender))
            self.barrier.remove(sender)
            return self.receive()
        else:
            logging.debug("Received message from {}".format(sender))
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
        assert self.initialized == True
        if encrypt:
            to_send = self.encrypt(data)
        else:
            to_send = data
        data_size = len(to_send)
        self.total_bytes += data_size
        id = str(uid).encode()
        self.peer_sockets[id].send(to_send)
        logging.debug("{} sent the message to {}.".format(self.uid, uid))
        logging.info("Sent this round: {}".format(data_size))

    def disconnect_neighbors(self):
        """
        Disconnects all neighbors.

        """
        assert self.initialized == True
        if not self.sent_disconnections:
            logging.info("Disconnecting neighbors")
            for sock in self.peer_sockets.values():
                sock.send(BYE)
            self.sent_disconnections = True
            while len(self.barrier):
                sender, recv = self.router.recv_multipart()
                if recv == BYE:
                    logging.debug("Received {} from {}".format(BYE, sender))
                    self.barrier.remove(sender)
                else:
                    logging.critical(
                        "Received unexpected {} from {}".format(recv, sender)
                    )
                    raise RuntimeError(
                        "Received a message when expecting BYE from {}".format(sender)
                    )
