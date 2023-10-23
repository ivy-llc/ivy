from typing import Union, Optional, Tuple
import mxnet as mx
import numpy as np
from mxnet.ndarray import NDArray

from ivy.utils.exceptions import IvyNotImplementedException


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def kaiser_bessel_derived_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def vorbis_window(
    window_length: Union[(None, mx.ndarray.NDArray)],
    *,
    dtype: None = np.float32,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def tril_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], ...)]:
    raise IvyNotImplementedException()


def blackman_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def polysub(poly1: NDArray, poly2: NDArray) -> NDArray:
    # Pad the coefficients to have the same length (necessary for subtraction)
    max_length = max(poly1.shape[0], poly2.shape[0])
    poly1 = mx.nd.pad(
        poly1,
        mode="constant",
        pad_width=(0, max_length - poly1.shape[0]),
        constant_value=0,
    )
    poly2 = mx.nd.pad(
        poly2,
        mode="constant",
        pad_width=(0, max_length - poly2.shape[0]),
        constant_value=0,
    )

    # Calculate the difference of the polynomials
    return poly1 - poly2
