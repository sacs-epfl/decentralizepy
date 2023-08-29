from decentralizepy.compression.Elias import Elias
from decentralizepy.compression.Quantization import Quantization


class EliasQuantization(Elias, Quantization):
    """
    Compress metadata and quantize parameters

    """

    def __init__(self, float_precision: int = 2**15 - 1, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        float_precision : int, optional
            Quantization parameter
        """
        super().__init__(float_precision=float_precision, *args, **kwargs)
        self.k = float_precision
