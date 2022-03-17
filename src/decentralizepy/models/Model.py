from torch import nn


class Model(nn.Module):
    """
    This class wraps the torch model
    More fields can be added here

    """

    def __init__(self):
        """
        Constructor

        """
        super().__init__()
        self.model_change = None
        self._param_count_ot = None
        self._param_count_total = None
        self.accumulated_changes = None

    def count_params(self, only_trainable=False):
        """
        Counts the total number of params

        Parameters
        ----------
        only_trainable : bool
            Counts only parameters with gradients when True

        Returns
        -------
        int
            Total number of parameters

        """
        if only_trainable:
            if not self._param_count_ot:
                self._param_count_ot = sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                )
            return self._param_count_ot
        else:
            if not self._param_count_total:
                self._param_count_total = sum(p.numel() for p in self.parameters())
            return self._param_count_total

    def rewind_accumulation(self, indices):
        """
        resets accumulated_changes at the given indices

        Parameters
        ----------
        indices : torch.Tensor
            Tensor that contains indices corresponding to the flatten model

        """
        if self.accumulated_changes is not None:
            self.accumulated_changes[indices] = 0.0
