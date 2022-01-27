import json
import logging
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
        port = rank + 20000
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(self, rank, machine_id, mapping, total_procs, addresses_filepath):
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

        """
        super().__init__(rank, machine_id, mapping, total_procs)

        with open(addresses_filepath) as addrs:
            self.ip_addrs = json.load(addrs)

        self.total_procs = total_procs
        self.rank = rank
        self.machine_id = machine_id
        self.mapping = mapping
        self.uid = mapping.get_uid(rank, machine_id)
        self.identity = str(self.uid).encode()
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.IDENTITY, self.identity)
        self.router.bind(self.addr(rank, machine_id))
        self.sent_disconnections = False

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
        Encode data using utf8.

        Parameters
        ----------
        data : dict
            Data dict to send

        Returns
        -------
        byte
            Encoded data

        """
        return json.dumps(data).encode("utf8")

    def decrypt(self, sender, data):
        """
        Decode received data from utf8.

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
        data = json.loads(data.decode("utf8"))
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
        assert self.initialized == True
        to_send = self.encrypt(data)
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
