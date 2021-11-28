import json
import math

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
        assert len(self.model.accumulated_gradients) > 0
        gradient_sum = self.model.accumulated_gradients[0]
        for i in range(1, len(self.model.accumulated_gradients)):
            for key in self.model.accumulated_gradients[i]:
                gradient_sum[key] += self.model.accumulated_gradients[i][key]
        gradient_sequence = []

        for key, gradient in gradient_sum.items():
            for index, val in enumerate(torch.flatten(gradient)):
                gradient_sequence.append((val, key, index))

        gradient_sequence.sort()
        return gradient_sequence

    def serialized_model(self):
        gradient_sequence = self.extract_sorted_gradients()
        gradient_sequence = gradient_sequence[
            : math.round(len(gradient_sequence) * self.alpha)
        ]

        m = dict()
        for _, key, index in gradient_sequence:
            if key not in m:
                m[key] = []
            m[key].append(index, torch.flatten(self.model.state_dict()[key])[index])

        for key in m:
            m[key] = json.dumps(m[key])

        return m

    def deserialized_model(self, m):
        state_dict = self.model.state_dict()

        for key, value in m.items():
            for index, param_val in json.loads(value):
                torch.flatten(state_dict[key])[index] = param_val
            state_dict[key] = torch.from_numpy(numpy.array(json.loads(value)))
        return state_dict
