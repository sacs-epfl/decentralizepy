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
        machine_addr = self.ip_addrs[str(machine_id)]
        port = rank + 20000
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(self, rank, machine_id, mapping, total_procs, addresses_filepath):
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
        self.context.destroy(linger=0)

    def encrypt(self, data):
        return json.dumps(data).encode("utf8")

    def decrypt(self, sender, data):
        sender = int(sender.decode())
        data = json.loads(data.decode("utf8"))
        return sender, data

    def connect_neighbors(self, neighbors):
        for uid in neighbors:
            logging.info("Connecting to my neighbour: {}".format(uid))
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
                logging.info("Recieved {} from {}".format(HELLO, sender))
                self.barrier.add(sender)
            elif recv == BYE:
                logging.info("Recieved {} from {}".format(BYE, sender))
                raise RuntimeError(
                    "A neighbour wants to disconnect before training started!"
                )
            else:
                logging.debug(
                    "Recieved message from {} @ connect_neighbors".format(sender)
                )

                self.peer_deque.append(self.decrypt(sender, recv))

    def receive(self):
        if len(self.peer_deque) != 0:
            resp = self.peer_deque[0]
            self.peer_deque.popleft()
            return resp

        sender, recv = self.router.recv_multipart()

        if recv == HELLO:
            logging.info("Recieved {} from {}".format(HELLO, sender))
            raise RuntimeError(
                "A neighbour wants to connect when everyone is connected!"
            )
        elif recv == BYE:
            logging.info("Recieved {} from {}".format(BYE, sender))
            self.barrier.remove(sender)
            return self.receive()
        else:
            logging.debug("Recieved message from {}".format(sender))
            return self.decrypt(sender, recv)

    def send(self, uid, data):
        to_send = self.encrypt(data)
        id = str(uid).encode()
        self.peer_sockets[id].send(to_send)
        logging.info("{} sent the message to {}.".format(self.uid, uid))

    def disconnect_neighbors(self):
        if not self.sent_disconnections:
            for sock in self.peer_sockets.values():
                sock.send(BYE)
            self.sent_disconnections = True
            while len(self.barrier):
                sender, recv = self.router.recv_multipart()
                if recv == BYE:
                    logging.info("Recieved {} from {}".format(BYE, sender))
                    self.barrier.remove(sender)
                else:
                    logging.critical(
                        "Recieved unexpected {} from {}".format(recv, sender)
                    )
                    raise RuntimeError(
                        "Received a message when expecting BYE from {}".format(sender)
                    )
