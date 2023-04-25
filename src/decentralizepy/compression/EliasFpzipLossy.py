# elias implementation: taken from this stack overflow post:
# https://stackoverflow.com/questions/62843156/python-fast-compression-of-large-amount-of-numbers-with-elias-gamma
import fpzip

from decentralizepy.compression.Elias import Elias


class EliasFpzipLossy(Elias):
    """
    Compression API

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
        """

    def compress_float(self, arr):
        """
        compression function for float arrays

        Parameters
        ----------
        arr : np.ndarray
            Data to compress

        Returns
        -------
        bytearray
            encoded data as bytes

        """
        return fpzip.compress(arr, precision=18, order="C")

    def decompress_float(self, bytes):
        """
        decompression function for compressed float arrays

        Parameters
        ----------
        bytes :bytearray
            compressed data

        Returns
        -------
        arr : np.ndarray
            decompressed data as array

        """
        return fpzip.decompress(bytes, order="C").squeeze()
