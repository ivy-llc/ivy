from typing import Union, Optional, Sequence
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException
import ivy


def all(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()


def any(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    keepdims: bool = False,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    x = mx.nd.array(x, dtype="bool")
    try:
        return mx.nd.any(x, axis=axis, keepdims=keepdims, out=out)
    except ValueError as error:
        raise ivy.utils.exceptions.IvyIndexError(error)
