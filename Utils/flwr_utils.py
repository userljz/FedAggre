
from io import BytesIO
from typing import cast

import numpy as np


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy.typing as npt
from functools import reduce

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
NDArrays = List[NDArray]

@dataclass
class Parameters:
    """Model parameters."""

    tensors: List[bytes]
    tensor_type: str



def ndarrays_to_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(NDArray, ndarray_deserialized)



def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime