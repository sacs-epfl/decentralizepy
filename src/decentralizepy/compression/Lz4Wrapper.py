import lz4.frame
import numpy as np

from decentralizepy.compression.Compression import Compression


class Lz4Wrapper(Compression):
    """
    Compression API

    """

    def __init__(self, compress_metadata=True, compress_data=False):
        """
        Constructor
        """
        self.compress_metadata = compress_metadata
        self.compress_data = compress_data

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
        if self.compress_metadata:
            arr.sort()
            diff = np.diff(arr, prepend=0).astype(np.int32)
            to_compress = diff.tobytes("C")
            return lz4.frame.compress(to_compress)
        return arr

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
        if self.compress_metadata:
            decomp = lz4.frame.decompress(bytes)
            return np.cumsum(np.frombuffer(decomp, dtype=np.int32))
        return bytes

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
        if self.compress_data:
            to_compress = arr.tobytes("C")
            return lz4.frame.compress(to_compress)
        return arr

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
        if self.compress_data:
            decomp = lz4.frame.decompress(bytes)
            return np.frombuffer(decomp, dtype=np.float32)
        return bytes
