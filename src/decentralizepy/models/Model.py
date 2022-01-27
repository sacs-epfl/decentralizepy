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
        self.accumulated_gradients = []
