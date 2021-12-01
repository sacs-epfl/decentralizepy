import json
import logging

import numpy
import torch

from decentralizepy.sharing.Sharing import Sharing


class PartialModel(Sharing):
    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        alpha=1.0,
        dict_ordered=True,
    ):
        super().__init__(
            rank, machine_id, communication, mapping, graph, model, dataset
        )
        self.alpha = alpha
        self.dict_ordered = dict_ordered

    def extract_top_gradients(self):
        logging.info("Summing up gradients")
        assert len(self.model.accumulated_gradients) > 0
        gradient_sum = self.model.accumulated_gradients[0]
        for i in range(1, len(self.model.accumulated_gradients)):
            for key in self.model.accumulated_gradients[i]:
                gradient_sum[key] += self.model.accumulated_gradients[i][key]

        logging.info("Returning topk gradients")
        tensors_to_cat = [v.data.flatten() for _, v in gradient_sum.items()]
        G_topk = torch.abs(torch.cat(tensors_to_cat, dim=0))
        return torch.topk(
            G_topk, round(self.alpha * G_topk.shape[0]), dim=0, sorted=False
        )

    def serialized_model(self):
        with torch.no_grad():
            _, G_topk = self.extract_top_gradients()
            logging.info("Extracting topk params")

            tensors_to_cat = [v.data.flatten() for v in self.model.parameters()]
            T = torch.cat(tensors_to_cat, dim=0)
            T_topk = T[G_topk]

            logging.info("Generating dictionary to send")

            m = dict()

            if not self.dict_ordered:
                raise NotImplementedError

            m["indices"] = G_topk.numpy().tolist()
            m["params"] = T_topk.numpy().tolist()

            assert len(m["indices"]) == len(m["params"])
            logging.info("Elements sending: {}".format(len(m["indices"])))

            logging.info("Generated dictionary to send")

            for key in m:
                m[key] = json.dumps(m[key])

            logging.info("Converted dictionary to json")

            return m

    def deserialized_model(self, m):
        with torch.no_grad():
            state_dict = self.model.state_dict()

            if not self.dict_ordered:
                raise NotImplementedError

            shapes = []
            lens = []
            tensors_to_cat = []
            for _, v in state_dict.items():
                shapes.append(v.shape)
                t = v.flatten()
                lens.append(t.shape[0])
                tensors_to_cat.append(t)

            T = torch.cat(tensors_to_cat, dim=0)
            index_tensor = torch.tensor(json.loads(m["indices"]))
            logging.debug("Original tensor: {}".format(T[index_tensor]))
            T[index_tensor] = torch.tensor(json.loads(m["params"]))
            logging.debug("Final tensor: {}".format(T[index_tensor]))
            start_index = 0
            for i, key in enumerate(state_dict):
                end_index = start_index + lens[i]
                state_dict[key] = T[start_index:end_index].reshape(shapes[i])
                start_index = end_index

            return state_dict
