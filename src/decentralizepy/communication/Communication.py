import json
import logging
from collections import deque

import zmq

HELLO = b"HELLO"
BYE = b"BYE"


class Communication:
    """
    Communcation API
    """

    def addr(self, rank, machine_id):
        machine_addr = self.ip_addrs[str(machine_id)]
        port = rank + 20000
        return "tcp://{}:{}".format(machine_addr, port)

    def __init__(self, rank, machine_id, total_procs, addresses_filepath, mapping):
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

    def encrypt(self, data):
        return json.dumps(data).encode("utf8")

    def decrypt(self, sender, data):
        sender = int(sender.decode())
        data = json.loads(data.decode("utf8"))
        return sender, data

    def connect_neighbours(self, neighbours):
        for uid in neighbours:
            id = str(uid).encode()
            req = self.context.socket(zmq.DEALER)
            req.setsockopt(zmq.IDENTITY, self.identity)
            req.connect(self.addr(*self.mapping.get_machine_and_rank(uid)))
            self.peer_sockets[id] = req
            req.send(HELLO)

        num_neighbours = len(neighbours)
        while len(self.barrier) < num_neighbours:
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
                logging.info(
                    "Recieved message from {} @ connect_neighbours".format(sender)
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
            if not self.sent_disconnections:
                for sock in self.peer_sockets.values():
                    sock.send(BYE)
                self.sent_disconnections = True
        else:
            logging.info("Recieved message from {}".format(sender))
            return self.decrypt(sender, recv)

    def send(self, uid, data):
        to_send = self.encrypt(data)
        id = str(uid).encode()
        self.peer_sockets[id].send(to_send)
        print("Message sent")
