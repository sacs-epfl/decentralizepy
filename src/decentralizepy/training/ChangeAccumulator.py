import json
import os
from pathlib import Path

import torch

from decentralizepy.training.Training import Training
from decentralizepy.utils import conditional_value


class ChangeAccumulator(Training):
    """
    This class implements the training module which also accumulates model change in a list.

    """

    def __init__(
        self,
        rank,
        machine_id,
        mapping,
        model,
        optimizer,
        loss,
        log_dir,
        rounds="",
        full_epochs="",
        batch_size="",
        shuffle="",
        save_accumulated="",
        accumulation=True,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        model : torch.nn.Module
            Neural Network for training
        optimizer : torch.optim
            Optimizer to learn parameters
        loss : function
            Loss function
        log_dir : str
            Directory to log the model change.
        rounds : int, optional
            Number of steps/epochs per training call
        full_epochs: bool, optional
            True if 1 round = 1 epoch. False if 1 round = 1 minibatch
        batch_size : int, optional
            Number of items to learn over, in one batch
        shuffle : bool
            True if the dataset should be shuffled before training.
        save_accumulated : bool
            True if accumulated weight change should be written to file
        accumulation : bool
            True if the model change should be accumulated across communication steps

        """
        super().__init__(
            rank,
            machine_id,
            mapping,
            model,
            optimizer,
            loss,
            log_dir,
            rounds,
            full_epochs,
            batch_size,
            shuffle,
        )
        self.save_accumulated = conditional_value(save_accumulated, "", False)
        self.communication_round = 0
        if self.save_accumulated:
            self.model_change_path = os.path.join(
                self.log_dir, "model_change/{}".format(self.rank)
            )
            Path(self.model_change_path).mkdir(parents=True, exist_ok=True)

            self.model_val_path = os.path.join(
                self.log_dir, "model_val/{}".format(self.rank)
            )
            Path(self.model_val_path).mkdir(parents=True, exist_ok=True)
        self.accumulation = accumulation
        self.init_model = None
        self.prev = None

    def save_vector(self, v, s):
        """
        Saves the given vector to the file.

        Parameters
        ----------
        v : torch.tensor
            The torch tensor to write to file
        s : str
            Path to folder to write to

        """
        output_dict = dict()
        output_dict["order"] = list(self.model.state_dict().keys())
        shapes = dict()
        for k, v1 in self.model.state_dict().items():
            shapes[k] = list(v1.shape)
        output_dict["shapes"] = shapes

        output_dict["tensor"] = v.tolist()

        with open(
            os.path.join(
                s,
                "{}.json".format(self.communication_round + 1),
            ),
            "w",
        ) as of:
            json.dump(output_dict, of)

    def save_change(self):
        """
        Saves the change and the gradient values for every iteration

        """
        tensors_to_cat = [
            v.data.flatten() for _, v in self.model.accumulated_gradients[0].items()
        ]
        change = torch.abs(torch.cat(tensors_to_cat, dim=0))
        self.save_vector(change, self.model_change_path)

    def save_model_params(self):
        """
        Saves the change and the gradient values for every iteration

        """
        tensors_to_cat = [v.data.flatten() for _, v in self.model.state_dict().items()]
        params = torch.abs(torch.cat(tensors_to_cat, dim=0))
        self.save_vector(params, self.model_val_path)

    def train(self, dataset):
        """
        One training iteration with accumulation of model change in model.accumulated_gradients.
        Goes through the entire dataset.

        Parameters
        ----------
        dataset : decentralizepy.datasets.Dataset
            The training dataset. Should implement get_trainset(batch_size, shuffle)

        """
        self.model.accumulated_gradients = []
        self.init_model = {
            k: v.data.clone().detach()
            for k, v in zip(self.model.state_dict(), self.model.parameters())
        }
        if self.accumulation:
            if self.model.accumulated_changes is None:
                flats = [v.data.flatten() for _, v in self.init_model.items()]
                flat = torch.cat(flats)
                self.model.accumulated_changes = torch.zeros_like(flat)
                self.prev = flat
            else:
                flats = [v.data.flatten() for _, v in self.init_model.items()]
                flat = torch.cat(flats)
                self.model.accumulated_changes += flat - self.prev
                self.prev = flat

        super().train(dataset)
        with torch.no_grad():
            change = {
                k: v.data.clone().detach() - self.init_model[k]
                for k, v in zip(self.model.state_dict(), self.model.parameters())
            }
            if self.accumulation:
                flats_change = [v.data.flatten() for _, v in change.items()]
                flat_change = torch.cat(flats_change)
                # flatten does not copy data if input is already flattened
                # however cat copies
                change = {"flat": self.model.accumulated_changes + flat_change}

            self.model.accumulated_gradients.append(change)

            if self.save_accumulated:
                self.save_change()
                self.save_model_params()

        self.communication_round += 1
