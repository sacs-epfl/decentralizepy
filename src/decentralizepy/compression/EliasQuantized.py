from decentralizepy.compression.Elias import Elias
from decentralizepy.compression.Quantization import Quantization


class EliasQuantization(Elias, Quantization):
    """
    Compress metadata and quantize parameters

    """

    def __init__(self, k: int = 8, *args, **kwargs):
        """
        Constructor
        Parameters
        ----------
        k : int, optional
            Quantization parameter
        """
        super().__init__(k)
        self.k = k
