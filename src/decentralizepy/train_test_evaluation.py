import logging
import os
from pathlib import Path

import numpy as np
import torch

from decentralizepy.graphs import Graph


class TrainTestHelper:
    def __init__(
        self,
        dataset,
        model,
        loss,
        dir,
        n_procs,
        trainer,
        comm,
        graph: Graph,
        threads_per_proc,
        eval_train=False,
    ):
        self.dataset = dataset
        self.model = model
        self.loss = loss
        self.dir = Path(dir)
        self.n_procs = n_procs
        self.trainer = trainer
        self.comm = comm
        self.star = graph
        self.threads_per_proc = threads_per_proc
        self.eval_train = eval_train

    def train_test_evaluation(self, iteration):
        with torch.no_grad():
            self.model.eval()
            total_threads = os.cpu_count()
            torch.set_num_threads(total_threads)

            neighbors = self.star.neighbors(0)
            state_dict_copy = {}
            shapes = []
            lens = []
            to_cat = []
            for key, val in self.model.state_dict().items():
                shapes.append(val.shape)
                clone_val = val.clone().detach()
                state_dict_copy[key] = clone_val
                flat = clone_val.flatten()
                to_cat.append(flat)
                lens.append(flat.shape[0])

            my_weight = torch.cat(to_cat)
            weights = [my_weight]
            # TODO: add weight of node 0
            for i in neighbors:
                sender, data = self.comm.receive()
                logging.info(f"Received weight from {sender}")
                weights.append(data)

            # averaging
            average_weight = np.mean([w.numpy() for w in weights], axis=0)

            start_index = 0
            average_weight_dict = {}
            for i, key in enumerate(state_dict_copy):
                end_index = start_index + lens[i]
                average_weight_dict[key] = torch.from_numpy(
                    average_weight[start_index:end_index].reshape(shapes[i])
                )
                start_index = end_index
            self.model.load_state_dict(average_weight_dict)
            if self.eval_train:
                logging.info("Evaluating on train set.")
                trl = self.trainer.eval_loss(self.dataset)
            else:
                trl = None
            logging.info("Evaluating on test set.")
            ta, tl = self.dataset.test(self.model, self.loss)
            # reload old weight
            self.model.load_state_dict(state_dict_copy)

            if trl is not None:
                print(iteration, ta, tl, trl)
            else:
                print(iteration, ta, tl)

            torch.set_num_threads(self.threads_per_proc)
            for neighbor in neighbors:
                self.comm.send(neighbor, "finished")
            self.model.train()
        return ta, tl, trl
