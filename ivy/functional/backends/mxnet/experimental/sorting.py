from typing import Union, Optional
import mxnet as mx

from ivy.utils.exceptions import IvyNotImplementedException


def lexsort(
    keys: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    axis: int = (-1),
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
