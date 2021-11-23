import json
import logging
from collections import deque

import numpy
import torch


class Sharing:
    """
    API defining who to share with and what, and what to do on receiving
    """

    def __init__(self, rank, machine_id, communication, mapping, graph, model, dataset):
        self.rank = rank
        self.machine_id = machine_id
        self.uid = mapping.get_uid(rank, machine_id)
        self.communication = communication
        self.mapping = mapping
        self.graph = graph
        self.model = model
        self.dataset = dataset

        self.peer_deques = dict()
        my_neighbors = self.graph.neighbors(self.uid)
        for n in my_neighbors:
            self.peer_deques[n] = deque()

    def received_from_all(self):
        for _, i in self.peer_deques.items():
            if len(i) == 0:
                return False
        return True

    def get_neighbors(self, neighbors):
        # modify neighbors here
        return neighbors

    def serialized_model(self):
        m = dict()
        for key, val in self.model.state_dict().items():
            m[key] = json.dumps(val.numpy().tolist())
        return m

    def deserialized_model(self, m):
        state_dict = dict()
        for key, value in m.items():
            state_dict[key] = torch.from_numpy(numpy.array(json.loads(value)))
        return state_dict

    def step(self):
        data = self.serialized_model()
        my_uid = self.mapping.get_uid(self.rank, self.machine_id)
        all_neighbors = self.graph.neighbors(my_uid)
        iter_neighbors = self.get_neighbors(all_neighbors)
        data["degree"] = len(all_neighbors)
        for neighbor in iter_neighbors:
            self.communication.send(neighbor, data)

        while not self.received_from_all():
            sender, data = self.communication.receive()
            logging.info("Received model from {}".format(sender))
            degree = data["degree"]
            del data["degree"]
            self.peer_deques[sender].append((degree, self.deserialized_model(data)))

        logging.info("Starting model averaging after receiving from all neighbors")
        total = dict()
        weight_total = 0
        for i, n in enumerate(self.peer_deques):
            logging.debug("Averaging model from neighbor {}".format(i))
            degree, data = self.peer_deques[n].popleft()
            weight = 1 / (max(len(self.peer_deques), degree) + 1)  # Metro-Hastings
            weight_total += weight
            for key, value in data.items():
                if key in total:
                    total[key] += value * weight
                else:
                    total[key] = value * weight

        for key, value in self.model.state_dict().items():
            total[key] += (1 - weight_total) * value  # Metro-Hastings

        self.model.load_state_dict(total)

        logging.info("Model averaging complete")
