import mxnet as mx
from numbers import Number
from typing import Optional, Union, Tuple

import numpy as np

import ivy


def argmax(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: Optional[Union[(ivy.Dtype, ivy.NativeDtype)]] = None,
    select_last_index: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.argmax Not Implemented")


def argmin(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    output_dtype: Optional[Union[np.dtype, str]] = None,
    select_last_index: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.argmin Not Implemented")


def nonzero(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[(None, mx.ndarray.NDArray, Tuple[Union[(None, mx.ndarray.NDArray)]])]:
    raise NotImplementedError("mxnet.nonzero Not Implemented")


def where(
    condition: Union[(None, mx.ndarray.NDArray)],
    x1: Union[(None, mx.ndarray.NDArray)],
    x2: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.where Not Implemented")


def argwhere(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.argwhere Not Implemented")
