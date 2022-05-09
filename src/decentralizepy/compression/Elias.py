# elias implementation: taken from this stack overflow post:
# https://stackoverflow.com/questions/62843156/python-fast-compression-of-large-amount-of-numbers-with-elias-gamma
import fpzip
import numpy as np

from decentralizepy.compression.Compression import Compression


class Elias(Compression):
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
        arr.sort()
        first = arr[0]
        arr = np.diff(arr).astype(np.int32)
        arr = arr.view(f"u{arr.itemsize}")
        l = np.log2(arr).astype("u1")
        L = ((l << 1) + 1).cumsum()
        out = np.zeros(int(L[-1] + 128), "u1")
        for i in range(l.max() + 1):
            out[L - i - 1] += (arr >> i) & 1

        s = np.array([out.size], dtype=np.int64)
        size = np.ndarray(8, dtype="u1", buffer=s.data)
        packed = np.packbits(out)
        packed[-8:] = size
        s = np.array([first], dtype=np.int64)
        size = np.ndarray(8, dtype="u1", buffer=s.data)
        packed[-16:-8] = size
        return packed

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
        n_arr = bytes[-8:]
        n = np.ndarray(1, dtype=np.int64, buffer=n_arr.data)[0]
        first = bytes[-16:-8]
        first = np.ndarray(1, dtype=np.int64, buffer=first.data)[0]
        b = bytes[:-16]
        b = np.unpackbits(b, count=n).view(bool)
        s = b.nonzero()[0]
        s = (s << 1).repeat(np.diff(s, prepend=-1))
        s -= np.arange(-1, len(s) - 1)
        s = s.tolist()  # list has faster __getitem__
        ns = len(s)

        def gen():
            idx = 0
            yield idx
            while idx < ns:
                idx = s[idx]
                yield idx

        offs = np.fromiter(gen(), int)
        sz = np.diff(offs) >> 1
        mx = sz.max() + 1
        out_fin = np.zeros(offs.size, int)
        out_fin[0] = first
        out = out_fin[1:]
        for i in range(mx):
            out[b[offs[1:] - i - 1] & (sz >= i)] += 1 << i
        out = np.cumsum(out_fin)
        return out

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
        return bytes
