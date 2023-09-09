import mxnet as mx
from typing import Union, Optional, Iterable, Any
import numpy as np

from ivy.utils.exceptions import IvyNotImplementedException


def current_backend_str() -> str:
    return "mxnet"


def is_native_array(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    exclusive: bool = False,
) -> bool:
    if exclusive:
        return isinstance(x, mx.ndarray.NDArray)
    else:
        return isinstance(x, mx.ndarray.NDArray) or isinstance(x, np.ndarray)

def all_equal(
    *xs: Iterable[Any], equality_matrix: bool = False
) -> Union[bool, mx.nd.NDArray]:  
    def equality_fn(a, b):
        if isinstance(a, mx.nd.NDArray) and isinstance(b, mx.nd.NDArray):
            return mx.nd.all(a == b)
        else:
            return a == b

    if equality_matrix:
        num_arrays = len(xs)
        mat = mx.nd.empty((num_arrays, num_arrays), ctx=mx.cpu(), dtype=bool)
        for i, xa in enumerate(xs):
            for j_, xb in enumerate(xs[i:]):
                j = j_ + i
                res = equality_fn(xa, xb)
                if isinstance(res, mx.nd.NDArray):
                    res = res.asscalar()
                mat[i][j] = res
                mat[j][i] = res
        return mat

    x0 = xs[0]
    for x in xs[1:]:
        if not equality_fn(x0, x):
            return False
    return True
def to_numpy(x: mx.ndarray.NDArray, /, *, copy: bool = True) -> np.ndarray:
    if copy:
        if x.shape == ():
            new_arr = x.asnumpy()
            return new_arr
        return x.copy().asnumpy()
    else:
        return x.asnumpy()


def itemsize(x: mx.ndarray.NDArray, /) -> int:
    return x.asnumpy().itemsize


def container_types():
    return []


def gather(
    x: mx.ndarray.NDArray,
    indices: mx.ndarray.NDArray,
    /,
    *,
    axis: int = -1,
    batch_dims: int = 0,
    out: Optional[mx.ndarray.NDArray] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise IvyNotImplementedException()
