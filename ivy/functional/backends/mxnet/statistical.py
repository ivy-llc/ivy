import mxnet as mx
import ivy
import numpy as np
from ivy.core.container import Container
from ivy.core.framework_handler import get_framework_from_str
from ivy.framework_handler import current_framework

# Array API Standard #
# -------------------#

def min(
    x: mx.nd.NDArray,
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
    out: Optional[mx.nd.NDArray] = None,
) -> np.ndarray:
    if axis is not None:
        x = mx.nd.min(x, axis=axis, keepdims=keepdims, out=out)
    else:
        x = mx.nd.min(x, out=out)
    if keepdims:
        if out is not None:
            raise Exception('Cannot use out parameter with keepdims=True.')
        x = x.reshape(tuple(np.insert(x.shape, axis, 1)))
    return ivy.mx.to_numpy(x)
