# Quantize to [-k, k]

import pickle

import numpy as np

from decentralizepy.compression.Compression import Compression


class Quantization(Compression):
    """
    Compress metadata and quantize parameters

    """

    def __init__(self, float_precision: int = 2 ** 15 - 1, *args, **kwargs):
        """
        Constructor
        
        Parameters
        ----------
        float_precision : int, optional
            Quantization parameter
        """
        super().__init__(*args, **kwargs)
        self.k = float_precision

    def compress_float(self, x):
        """
        compression function for float arrays

        Parameters
        ----------
        x : np.ndarray
            Data to compress

        Returns
        -------
        bytearray
            encoded data as bytes

        """

        # Compute scale factor
        scale_factor = np.mean(np.abs(x)) / self.k
        # scale_factor = np.max(np.abs(x)) / self.k

        # Normalize x to [-k, k]
        norm_factor = np.max(np.abs(x)) / self.k
        x = x / norm_factor
        x = x.round().astype(np.int32)

        # Get the maximum absolute value from the input array
        max_abs = np.max(np.abs(x))

        # Get the nearest power of 2 greater than equal to max_abs
        nearest_pow_2 = 2 ** np.ceil(np.log2(max_abs))

        # Check if nearest_pow_2 is the same as max_abs
        if nearest_pow_2 == max_abs:
            nearest_pow_2 = nearest_pow_2 * 2

        # Calculate the number of bits required to represent the nearest power of 2
        num_bits = int(np.ceil(np.log2(nearest_pow_2))) + 1

        # Make all numbers of x positive
        x = x + nearest_pow_2 - 1

        x = np.asarray(x, dtype=np.uint32)

        # Create a numpy array of shape (x.shape, num_bits) and fill it with zeros
        bit_rep = np.zeros((x.shape[0], num_bits), dtype=np.uint8)

        # Iterate over x and convert each number to binary
        for i in range(len(x)):
            str_bit = np.binary_repr(x[i], width=num_bits)
            array_bit = np.array(list(str_bit), dtype=np.uint8)
            indices_with_1 = np.where(array_bit == 1)[0]
            bit_rep[i][indices_with_1] = 1

        bit_rep = bit_rep.reshape(-1)

        # Pack the bits into minimum number of bytes
        intermediate_rep = np.packbits(bit_rep, bitorder="little")
        padding = np.array([0], dtype=np.uint8)
        if bit_rep.shape[0] % 8:
            padding = np.array([8 - (bit_rep.shape[0] % 8)], dtype=np.uint8)
        num_bits = np.array([num_bits], dtype=np.uint8)
        to_send = np.concatenate((padding, num_bits, intermediate_rep), dtype=np.uint8)

        return pickle.dumps((scale_factor, to_send))

    def decompress_float(self, bytes):
        """
        decompression function for compressed float arrays

        Parameters
        ----------
        bytes :bytearray
            compressed data

        Returns
        -------
        np.ndarray
            decompressed data as array

        """
        # Extract scale_factor and x from bytes
        scale_factor, x = pickle.loads(bytes)

        # Extract padding and num_bits from x
        padding = -x[0].item() if x[0].item() else None
        num_bits = x[1].item()
        rest_of_x = x[2:].astype(np.uint8)

        # Unpack rest_of_x and reshape it
        received_x = np.unpackbits(rest_of_x, bitorder="little", count=padding)
        received_x = received_x.reshape((-1, num_bits)).astype(np.uint8)

        # Initialize an unit8 array with the same number of rows as received_x
        output = np.zeros(received_x.shape[0], dtype=np.int32)

        # Convert each row into an integer
        for i in range(received_x.shape[0]):
            output[i] = (
                int("".join(received_x[i].astype(str)), 2) - (2 ** (num_bits - 1)) + 1
            )

        # Denormalize the output
        output = output * scale_factor

        return output.astype(np.float32)
