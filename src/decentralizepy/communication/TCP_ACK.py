import json
import logging
import pickle
import queue
from collections import deque
from threading import Event, Lock, Thread
from time import sleep

import zmq

import socket

from decentralizepy.communication.Communication import Communication

HELLO = b"HELLO"
BYE = b"BYE"
RESEND_TIMEOUT = 0.5  # s
RECV_TIMEOUT = 50  # ms


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
        machine_addr = socket.gethostbyname(self.ip_addrs[str(machine_id)])
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
        recv_timeout=RECV_TIMEOUT,
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
        self.recv_timeout = recv_timeout
        self.uid = mapping.get_uid(rank, machine_id)
        self.identity = str(self.uid).encode()
        self.context = zmq.Context()
        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.IDENTITY, self.identity)
        self.router.setsockopt(zmq.RCVTIMEO, self.recv_timeout)
        self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.router.bind(self.addr(rank, machine_id))

        self.total_data = 0
        self.total_meta = 0

        self.peer_deque = deque()
        self.peer_sockets = dict()

        self.receiverQueue = queue.Queue()
        self.mutex = Lock()
        self.senderThread = Thread(target=self.keep_sending, daemon=True)
        self.receiverThread = Thread(target=self.keep_receiving, daemon=True)
        self.senderQueue = dict()
        self.terminateEvent = Event()

        self.sent_IDs = dict()
        self.received_IDs = dict()

        self.senderThread.start()
        self.receiverThread.start()

    def keep_sending(self):
        while True:
            if self.terminateEvent.is_set():
                return

            self.mutex.acquire()
            for key in self.senderQueue:
                for _, m in self.senderQueue[key]:
                    self.push_message(key, m)
            self.mutex.release()
            sleep(RESEND_TIMEOUT)

    def keep_receiving(self):
        while True:
            if self.terminateEvent.is_set():
                return

            try:
                sender, recv = self.router.recv_multipart()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.EAGAIN:
                    continue
                else:
                    raise

            s, id, isAck, r = self.decrypt(sender, recv)
            if isAck:
                logging.debug("Received acknowledgement from {} id {}".format(s, id))
                self.mutex.acquire()
                self.senderQueue[s][:] = [
                    tup for tup in self.senderQueue[s] if id != tup[0]
                ]
                # for i, v in enumerate(self.senderQueue[s]):
                #     if id == v[0]:
                #         del self.senderQueue[s][i]
                #         break
                self.mutex.release()
            else:
                logging.debug("Sending acknowledgement to {} id {}".format(s, id))
                self.mutex.acquire()
                self.push_message(s, self.encrypt({}, id, isAck=True))
                self.mutex.release()
                if s not in self.received_IDs:
                    self.received_IDs[s] = set()
                if id not in self.received_IDs[s]:
                    logging.debug("Received {} from {}".format(id, s))
                    self.received_IDs[s].add(id)
                    self.receiverQueue.put((s, r))
                else:
                    logging.debug("Duplicate {} from {}".format(id, s))

    def __del__(self):
        """
        Destroys zmq context

        """
        self.context.destroy(linger=0)

    def encrypt(self, data, id=0, isAck=False):
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
        output = pickle.dumps({"id": id, "isAck": isAck, "data": data})
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
            (sender: int, id: int, isAck: bool, data: dict)

        """
        sender = int(sender.decode())
        data = pickle.loads(data)
        id, isAck, data = data["id"], data["isAck"], data["data"]
        return sender, id, isAck, data

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

    def destroy_connection(self, neighbor, linger=None):
        id = str(neighbor).encode()
        if self.already_connected(neighbor):
            self.peer_sockets[id].close(linger=linger)
            del self.peer_sockets[id]

    def already_connected(self, neighbor):
        id = str(neighbor).encode()
        return id in self.peer_sockets

    def receive(self, block=True):
        """
        Returns ONE message received.

        Returns
        -------
        dict
            Received and decrypted data

        Raises
        ------
        RuntimeError
            If received HELLO

        """
        while True:
            try:
                return self.receiverQueue.get(block=True, timeout=self.recv_timeout)
            except queue.Empty as _:
                if not block:
                    return None
                else:
                    continue

    def push_message(self, uid, data):
        """
        Send a message to a process.

        Parameters
        ----------
        uid : int
            Neighbor's unique ID
        data : dict
            Message as a Python dictionary

        """
        id = str(uid).encode()
        if id in self.peer_sockets:
            self.peer_sockets[id].send(data)
        else:
            logging.debug("Not sending message to {}: Initialize Connection".format(id))

    def get_message_id(self, uid):
        if uid not in self.sent_IDs:
            self.sent_IDs[uid] = 0
        m_id = self.sent_IDs[uid]
        self.sent_IDs[uid] += 1
        return m_id

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
        assert encrypt
        message_id = self.get_message_id(uid)
        to_send = self.encrypt(data, message_id)

        data_size = len(to_send)
        self.total_bytes += data_size

        self.mutex.acquire()

        if uid not in self.senderQueue:
            self.senderQueue[uid] = []
        self.senderQueue[uid].append((message_id, to_send))

        self.push_message(uid, to_send)
        self.mutex.release()

        logging.debug("{} sent the message to {}.".format(self.uid, uid))
        logging.debug("Sent message size: {}".format(data_size))

    # def terminate(self):
    #     """
    #     Safely terminate the communication sockets.

    #     """
    #     self.terminateEvent.set()
    #     self.senderThread.join()
    #     self.receiverThread.join()
