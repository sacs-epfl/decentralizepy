import numpy as np


class Compression:
    """
    Compression API

    """

    def __init__(self):
        """
        Constructor
        """

    def compress(self, arr):
        """
        compression function

        Parameters
        ----------
        arr : np.ndarray
            Data to compress

        Returns
        -------
        bytearray
            encoded data as bytes

        """
        raise NotImplementedError

    def decompress(self, bytes):
        """
        decompression function

        Parameters
        ----------
        bytes :bytearray
            compressed data

        Returns
        -------
        arr : np.ndarray
            decompressed data as array

        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError
