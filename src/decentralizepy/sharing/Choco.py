import logging
from collections import OrderedDict

import torch

from decentralizepy.sharing.Sharing import Sharing


def zeros_like_state_dict(state_dict):
    """
    Creates a new state dictionary such that it has same
    layers (name and size) as the input state dictionary, but all values
    are zero

    Parameters
    ----------
    state_dict: dict[str, torch.Tensor]

    """
    result_dict = OrderedDict()
    for tensor_name, tensor_values in state_dict.items():
        result_dict[tensor_name] = torch.zeros_like(tensor_values)
    return result_dict


def get_dict_keys_and_check_matching(dict_1, dict_2):
    """
    Checks if keys of the two dictionaries match and
    reutrns them if they do, otherwise raises ValueError

    Parameters
    ----------
    dict_1: dict
    dict_2: dict

    Raises
    ------
    ValueError
        If the keys of the dictionaries don't match

    """
    keys = dict_1.keys()
    if set(keys).difference(set(dict_2.keys())):
        raise ValueError("Dictionaries must have matching keys")
    return keys


def subtract_state_dicts(_1, _2):
    """
    Subtracts one state dictionary from another

    Parameters
    ----------
    _1: dict[str, torch.Tensor]
        Minuend state dictionary
    _2: dict[str, torch.Tensor]
        Subtrahend state dictionary

    Raises
    ------
    ValueError
        If the keys of the state dictionaries don't match

    """
    keys = get_dict_keys_and_check_matching(_1, _2)
    result_dict = OrderedDict()
    for key in keys:
        # Size checking is done by torch during the subtraction
        result_dict[key] = _1[key] - _2[key]
    return result_dict


def self_add_state_dict(_1, _2, constant=1.0):
    """
    Scales one state dictionary by a constant and
    adds it directly to another minimizing copies
    created. Equivalent to operation `_1 += constant * _2`

    Parameters
    ----------
    _1: dict[str, torch.Tensor]
        State dictionary
    _2: dict[str, torch.Tensor]
        State dictionary
    constant: float
        Constant to scale _2 with

    Raises
    ------
    ValueError
        If the keys of the state dictionaries don't match

    """
    keys = get_dict_keys_and_check_matching(_1, _2)
    for key in keys:
        # Size checking is done by torch during the subtraction
        _1[key] += constant * _2[key]


def flatten_state_dict(state_dict):
    """
    Transforms state dictionary into a flat tensor
    by flattening and concatenating tensors of the
    state dictionary.

    Note: changes made to the result won't affect state dictionary

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.tensor]
        A state dictionary to flatten

    """
    return torch.cat([tensor.flatten() for tensor in state_dict.values()], axis=0)


def unflatten_state_dict(flat_tensor, reference_state_dict):
    """
    Transforms a falt tensor into a state dictionary
    by using another state dictionary as a reference
    for size and names of the tensors. Assumes
    that the number of elements of the flat tensor
    is the same as the number of elements in the
    reference state dictionary.

    This operation is inverse operation to flatten_state_dict

    Note: changes made to the result will affect the flat tensor

    Parameters
    ----------
    flat_tensor : torch.tensor
        A 1-dim tensor
    reference_state_dict : OrderedDict[str, torch.tensor]
        A state dictionary used as a reference for tensor names
    and shapes of the result

    """
    result = OrderedDict()
    start_index = 0
    for tensor_name, tensor in reference_state_dict.items():
        end_index = start_index + tensor.numel()
        result[tensor_name] = flat_tensor[start_index:end_index].reshape(tensor.shape)
        start_index = end_index
    return result


def serialize_sparse_tensor(tensor):
    """
    Serializes sparse tensor by flattening it and
    returning values and indices of it that are not 0

    Parameters
    ----------
    tensor: torch.Tensor

    """
    flat = tensor.flatten()
    indices = flat.nonzero(as_tuple=True)[0]
    values = flat[indices]
    return values, indices


def deserialize_sparse_tensor(values, indices, shape):
    """
    Deserializes tensor from its non-zero values and indices
    in flattened form and original shape of the tensor.

    Parameters
    ----------
    values: torch.Tensor
        Non-zero entries of flattened original tensor
    indices: torch.Tensor
        Respective indices of non-zero entries of flattened original tensor
    shape: torch.Size or tuple[*int]
        Shape of the original tensor

    """
    result = torch.zeros(size=shape)
    if len(indices):
        flat_result = result.flatten()
        flat_result[indices] = values
    return result


def topk_sparsification_tensor(tensor, alpha):
    """
    Performs topk sparsification of a tensor and returns
    the same tensor from the input but transformed.

    Note: no copies are created, but input vector is directly changed

    Parameters
    ----------
    tensor : torch.tensor
        A tensor to perform the sparsification on
    alpha : float
        Percentage of topk values to keep

    """
    tensor_abs = tensor.abs()
    flat_abs_tensor = tensor_abs.flatten()
    numel_to_keep = round(alpha * flat_abs_tensor.numel())
    if numel_to_keep > 0:
        cutoff_value, _ = torch.kthvalue(-flat_abs_tensor, numel_to_keep)
        tensor[tensor_abs < -cutoff_value] = 0
    return tensor


def topk_sparsification(state_dict, alpha):
    """
    Performs topk sparsification of a state_dict
    applying it over all elements together.

    Note: the changes made to the result won't affect
    the input state dictionary

    Parameters
    ----------
    state_dict : OrderedDict[str, torch.tensor]
        A state dictionary to perform the sparsification on
    alpha : float
        Percentage of topk values to keep

    """
    flat_tensor = flatten_state_dict(state_dict)
    return unflatten_state_dict(
        topk_sparsification_tensor(flat_tensor, alpha), state_dict
    )


def serialize_sparse_state_dict(state_dict):
    with torch.no_grad():
        concatted_tensors = torch.cat(
            [tensor.flatten() for tensor in state_dict.values()], axis=0
        )
        return serialize_sparse_tensor(concatted_tensors)


def deserialize_sparse_state_dict(values, indices, reference_state_dict):
    with torch.no_grad():
        keys = []
        lens = []
        shapes = []
        for k, v in reference_state_dict.items():
            keys.append(k)
            shapes.append(v.shape)
            lens.append(v.numel())
        total_num_el = sum(lens)
        T = deserialize_sparse_tensor(values, indices, (total_num_el,))
        result_state_dict = OrderedDict()
        start_index = 0
        for i, k in enumerate(keys):
            end_index = start_index + lens[i]
            result_state_dict[k] = T[start_index:end_index].reshape(shapes[i])
            start_index = end_index
        return result_state_dict


class Choco(Sharing):
    """
    API defining who to share with and what, and what to do on receiving

    """

    def __init__(
        self,
        rank,
        machine_id,
        communication,
        mapping,
        graph,
        model,
        dataset,
        log_dir,
        step_size,
        alpha,
        compress=False,
        compression_package=None,
        compression_class=None,
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Local rank
        machine_id : int
            Global machine id
        communication : decentralizepy.communication.Communication
            Communication module used to send and receive messages
        mapping : decentralizepy.mappings.Mapping
            Mapping (rank, machine_id) -> uid
        graph : decentralizepy.graphs.Graph
            Graph reprensenting neighbors
        model : decentralizepy.models.Model
            Model to train
        dataset : decentralizepy.datasets.Dataset
            Dataset for sharing data. Not implemented yet! TODO
        log_dir : str
            Location to write shared_params (only writing for 2 procs per machine)
        step_size : float
            Step size from the formulation of Choco
        alpha : float
            Percentage of elements to keep during topk sparsification

        """
        super().__init__(
            rank,
            machine_id,
            communication,
            mapping,
            graph,
            model,
            dataset,
            log_dir,
            compress,
            compression_package,
            compression_class,
        )
        self.step_size = step_size
        self.alpha = alpha
        logging.debug(
            "type(step_size): %s, value: %s",
            str(type(self.step_size)),
            str(self.step_size),
        )
        logging.debug(
            "type(alpha): %s, value: %s", str(type(self.alpha)), str(self.alpha)
        )
        model_state_dict = model.state_dict()
        self.model_hat = zeros_like_state_dict(model_state_dict)
        self.s = zeros_like_state_dict(model_state_dict)
        self.my_q = None

    def compress_data(self, data):
        result = dict(data)
        if self.compress:
            if "indices" in result:
                result["indices"] = self.compressor.compress(result["indices"])
            if "params" in result:
                result["params"] = self.compressor.compress_float(result["params"])
        return result

    def decompress_data(self, data):
        if self.compress:
            if "indices" in data:
                data["indices"] = self.compressor.decompress(data["indices"])
            if "params" in data:
                data["params"] = self.compressor.decompress_float(data["params"])
        return data

    def _compress(self, x):
        return topk_sparsification(x, self.alpha)

    def _pre_step(self):
        """
        Called at the beginning of step.

        """
        with torch.no_grad():
            self.my_q = self._compress(
                subtract_state_dicts(self.model.state_dict(), self.model_hat)
            )

    def serialized_model(self):
        """
        Convert self q to a dictionary. Here we can choose how much to share

        Returns
        -------
        dict
            Model converted to dict

        """
        values, indices = serialize_sparse_state_dict(self.my_q)
        data = dict()
        data["params"] = values.numpy()
        data["indices"] = indices.numpy()
        data["send_partial"] = True
        return self.compress_data(data)

    def deserialized_model(self, m):
        """
        Convert received dict to state_dict.

        Parameters
        ----------
        m : dict
            received dict

        Returns
        -------
        state_dict
            state_dict of received

        """
        if "send_partial" not in m:
            return super().deserialized_model(m)

        with torch.no_grad():
            m = self.decompress_data(m)
            indices = torch.tensor(m["indices"], dtype=torch.long)
            values = torch.tensor(m["params"])
            return deserialize_sparse_state_dict(
                values, indices, self.model.state_dict()
            )

    def _averaging(self, peer_deques):
        """
        Averages the received model with the local model

        """
        with torch.no_grad():
            self_add_state_dict(self.model_hat, self.my_q)  # x_hat = q_self + x_hat
            weight_total = 0
            for i, n in enumerate(peer_deques):
                data = peer_deques[n].popleft()
                degree, iteration = data["degree"], data["iteration"]
                del data["degree"]
                del data["iteration"]
                del data["CHANNEL"]
                logging.debug(
                    "Averaging model from neighbor {} of iteration {}".format(
                        n, iteration
                    )
                )
                data = self.deserialized_model(data)
                # Metro-Hastings
                weight = 1 / (max(len(peer_deques), degree) + 1)
                weight_total += weight
                for key, value in data.items():
                    if key in self.s:
                        self.s[key] += value * weight
                    # else:
                    #     self.s[key] = value * weight

            for key, value in self.my_q.items():
                self.s[key] += (1 - weight_total) * value  # Metro-Hastings

            total = self.model.state_dict().copy()
            self_add_state_dict(
                total,
                subtract_state_dicts(self.s, self.model_hat),
                constant=self.step_size,
            )  # x = x + gamma * (s - x_hat)

        self.model.load_state_dict(total)
        self._post_step()
        self.communication_round += 1

    def _averaging_server(self, peer_deques):
        """
        Averages the received models of all working nodes

        """
        raise NotImplementedError()
