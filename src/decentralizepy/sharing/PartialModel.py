import json
import logging

import numpy
import torch

from decentralizepy.sharing.Sharing import Sharing


class PartialModel(Sharing):
    def __init__(
        self, rank, machine_id, communication, mapping, graph, model, dataset, alpha=1.0
    ):
        super().__init__(
            rank, machine_id, communication, mapping, graph, model, dataset
        )
        self.alpha = alpha

    def extract_sorted_gradients(self):
        logging.info("Summing up gradients")
        assert len(self.model.accumulated_gradients) > 0
        gradient_sum = self.model.accumulated_gradients[0]
        for i in range(1, len(self.model.accumulated_gradients)):
            for key in self.model.accumulated_gradients[i]:
                gradient_sum[key] += self.model.accumulated_gradients[i][key]
        gradient_sequence = []

        logging.info("Turning gradients into tuples")

        for key, gradient in gradient_sum.items():
            for index, val in enumerate(torch.flatten(gradient)):
                gradient_sequence.append((val, key, index))

        logging.info("Sorting gradient tuples")

        gradient_sequence.sort()  # bottleneck
        return gradient_sequence

    def serialized_model(self):
        gradient_sequence = self.extract_sorted_gradients()
        logging.info("Extracted sorted gradients")
        gradient_sequence = gradient_sequence[
            : round(len(gradient_sequence) * self.alpha)
        ]

        m = dict()
        for _, key, index in gradient_sequence:
            if key not in m:
                m[key] = []
            m[key].append(
                (
                    index,
                    torch.flatten(self.model.state_dict()[key])[index].numpy().tolist(),
                )
            )

        logging.info("Generated dictionary to send")

        for key in m:
            m[key] = json.dumps(m[key])

        logging.info("Converted dictionary to json")

        return m

    def deserialized_model(self, m):
        state_dict = self.model.state_dict()

        for key, value in m.items():
            for index, param_val in json.loads(value):
                torch.flatten(state_dict[key])[index] = param_val
        return state_dict
