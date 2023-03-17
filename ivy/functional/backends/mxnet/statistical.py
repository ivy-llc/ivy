# global
import mxnet as mx
from typing import Union, Optional, Sequence

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.jax import JaxArray
from . import backend_version

# Array API Standard #
# -------------------#


def min(
    x: mx.nd.NDArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> mx.ndarray:
    if axis is not None:
        x = mx.nd.min(x, axis=axis, keepdims=keepdims, out=out)
    else:
        x = mx.nd.min(x, out=out)
    if keepdims:
        if out is not None:
            raise Exception('Cannot use out parameter with keepdims=True.')
        x = x.reshape(tuple(np.insert(x.shape, axis, 1)))
    return ivy.mx.to_numpy(x)
