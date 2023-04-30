from typing import Union, Optional, Tuple
import mxnet as mx
import numpy as np


def triu_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[Union[(None, mx.ndarray.NDArray)]]:
    raise NotImplementedError("mxnet.triu_indices Not Implemented")


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.kaiser_window Not Implemented")


def kaiser_bessel_derived_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.kaiser_bessel_derived_window Not Implemented")


def vorbis_window(
    window_length: Union[(None, mx.ndarray.NDArray)],
    *,
    dtype: None = np.float32,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.vorbis_window Not Implemented")


def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[None] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.hann_window Not Implemented")


def tril_indices(
    n_rows: int, n_cols: Optional[int] = None, k: int = 0, /, *, device: str
) -> Tuple[(Union[(None, mx.ndarray.NDArray)], ...)]:
    raise NotImplementedError("mxnet.tril_indices Not Implemented")


def frombuffer(
    buffer: bytes,
    dtype: Optional[None] = float,
    count: Optional[int] = (-1),
    offset: Optional[int] = 0,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.frombuffer Not Implemented")
